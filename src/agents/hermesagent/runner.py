from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.agents.approval_posture import HERMES as HERMES_POSTURE, record as record_posture
from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.utils.docker_utils import (
    run_warmup,
    setup_skills,
    inject_lobster_workspace,
    TMP_WORKSPACE,
)
from src.utils.gateway_usage import TASK_ID_HEADER
from src.utils.grading import extract_usage_from_jsonl
from src.utils.transient_errors import (
    RESUME_ATTEMPTS,
    RESUME_BACKOFF_S,
    resumable_provider_error,
    unbounded_provider_error,
    unrecoverable_session_error,
)

load_dotenv()

logger = logging.getLogger(__name__)

# v0.5 is the official base; the tag names the hermes-agent release the image
# actually carries, which Dockerfile.hermesagent now swaps in from
# baselines/hermes-agent rather than inheriting from the base image.
HERMES_IMAGE = os.environ.get("HERMES_DOCKER_IMAGE", "").strip()
HERMES_HOME = "/root/.hermes"
HERMES_INSTALL_DIR = "/opt/hermes"
HERMES_VENV_PYTHON = "/opt/hermes/.venv/bin/python3"
# Ordinary same-session resumes retain the shared three-resume limit (see
# transient_errors.RESUME_ATTEMPTS). The encrypted-session and
# instant-inference quota signatures bypass it.
HERMES_RESUME_ATTEMPTS: int | None = RESUME_ATTEMPTS
HERMES_RETRY_DELAY_SECONDS = RESUME_BACKOFF_S
# The host-side docker exec timeout is not a graceful stop for the Python
# runner inside the container.  Allow Hermes to persist its session and log
# the last native response before escalating to TERM/KILL.
HERMES_TIMEOUT_FLUSH_SECONDS = 15.0
HERMES_AGENT_PID_PATH = "/tmp/wildclaw-hermes-agent.pid"

OPENCLAW_COMPAT_TRANSCRIPT_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"
BENCH_RUNNER_HOST_PATH = Path(__file__).with_name("bench_runner.py")
BENCH_CONFIG_CONTAINER_PATH = "/tmp/hermes_bench_config.json"
COMPAT_TRANSCRIPT_HOST_PATH = Path(__file__).with_name("compat_transcript.py")


class HermesAgentAgent(BaseAgent):
    def __init__(
        self,
        image: str | None = None,
        openrouter_api_key: str = "",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        brave_api_key: str = "",
    ) -> None:
        self.image = (image or HERMES_IMAGE).strip()
        if not self.image:
            raise ValueError(
                "HERMES_DOCKER_IMAGE must be set when no Hermes image is passed"
            )
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = openrouter_base_url
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY", "")

    @property
    def expects_gateway(self) -> bool:
        return False

    @property
    def transcript_container_path(self) -> str:
        return OPENCLAW_COMPAT_TRANSCRIPT_PATH

    def prepare_grading_transcript(self, task_id: str) -> str:
        self._write_compat_transcript(task_id)
        return self.transcript_container_path

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        elapsed_time = 0.0
        agent_proc = None
        start_time: float | None = None
        excluded_retry_time = 0.0

        try:
            api_key, base_url = self._resolve_runtime_provider(spec.model, spec.models_config)

            exec_path = os.path.join(spec.workspace_path, "exec")
            tmp_path = os.path.join(spec.workspace_path, "tmp")
            os.makedirs(exec_path, exist_ok=True)

            self._start_container(
                spec.task_id,
                exec_path,
                api_key=api_key,
                base_url=base_url,
                extra_env=spec.task.get("env", ""),
                tmp_path=tmp_path,
                lobster_env=spec.lobster.get("env") if spec.lobster else None,
            )
            if spec.lobster:
                inject_lobster_workspace(spec.task_id, spec.lobster["workspace"])

            self._prepare_workspace(spec.task_id)
            setup_skills(
                spec.task_id,
                spec.task.get("skills", ""),
                spec.task.get("skills_path", ""),
                container_skills_root=f"{HERMES_HOME}/skills",
            )
            run_warmup(spec.task_id, spec.task.get("warmup", ""))

            self._configure_hermes(spec.task_id, api_key, base_url, spec.model)
            # Read back from the container rather than from the string this
            # process just built: the config only counts once hermes-agent can
            # load it, and the point of the artifact is that a finished run can
            # be audited without re-deriving what the harness would have done.
            readback = subprocess.run(
                ["docker", "exec", spec.task_id, "/bin/bash", "-c",
                 f"sed -n '/^approvals:/,/^[^ ]/p' {HERMES_HOME}/config.yaml"],
                capture_output=True, text=True,
            )
            record_posture(
                spec.output_dir,
                HERMES_POSTURE,
                applied={
                    "delivered_as": f"{HERMES_HOME}/config.yaml (and hermes.yaml)",
                    "readback": (readback.stdout or "").strip()[:400],
                    "readback_returncode": readback.returncode,
                },
            )

            reasoning_config = self._map_thinking(spec.thinking)
            self._write_bench_runner(
                spec.task_id, spec.prompt, spec.model,
                api_key, base_url, reasoning_config,
            )

            start_time = time.perf_counter()
            resume_attempt = 0
            while True:
                log_offset = (
                    spec.output_dir.joinpath("agent.log").stat().st_size
                    if spec.output_dir.joinpath("agent.log").exists()
                    else 0
                )
                agent_proc = self._run_bench_runner_background(
                    task_id=spec.task_id,
                    log_path=spec.output_dir / "agent.log",
                    append=resume_attempt > 0,
                    resume=resume_attempt > 0,
                )
                counted_elapsed = time.perf_counter() - start_time - excluded_retry_time
                remaining = int(spec.timeout_seconds - counted_elapsed)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        cmd="Hermes task",
                        timeout=spec.timeout_seconds,
                    )
                remaining = max(1, remaining)
                logger.info("[%s] Waiting for hermes-agent to finish...", spec.task_id)
                attempt_started = time.perf_counter()
                provider_error_reason: str | None = None
                try:
                    deadline = time.perf_counter() + remaining
                    while True:
                        try:
                            agent_proc.wait(timeout=min(1.0, max(0.05, deadline - time.perf_counter())))
                            break
                        except subprocess.TimeoutExpired:
                            provider_error_reason = self._find_error_marker(
                                spec.output_dir / "agent.log", log_offset,
                                unrecoverable_session_error,
                            )
                            if provider_error_reason:
                                logger.warning(
                                    "[%s] Hermes provider error detected (%s); stopping current run for session resume",
                                    spec.task_id, provider_error_reason,
                                )
                                agent_proc.terminate()
                                try:
                                    agent_proc.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    agent_proc.kill()
                                    agent_proc.wait()
                                break
                            if time.perf_counter() >= deadline:
                                raise
                except subprocess.TimeoutExpired:
                    logger.info("[%s] hermes-agent timed out...", spec.task_id)
                    elapsed_time = float(spec.timeout_seconds)
                    self._flush_timed_out_run(spec.task_id, agent_proc)
                    self._cleanup_bench_config(spec.task_id)
                    return AgentExecution(
                        elapsed_time=elapsed_time,
                        error="Hermes run timed out",
                        gateway_proc=None,
                        agent_proc=agent_proc,
                        excluded_retry_time=excluded_retry_time,
                    )
                if not provider_error_reason:
                    provider_error_reason = self._find_error_marker(
                        spec.output_dir / "agent.log", log_offset
                    )
                attempt_elapsed = time.perf_counter() - attempt_started
                self._close_runner_streams(agent_proc)
                if agent_proc.returncode == 0 and not provider_error_reason:
                    # Retry time stays INSIDE elapsed_time, deliberately; see
                    # the note in the claudecode runner.  excluded_retry_time
                    # is still accumulated below because the *task budget*
                    # still refunds a resumed attempt -- only the reported
                    # wall clock stopped deducting it.
                    elapsed_time = time.perf_counter() - start_time
                    if excluded_retry_time:
                        logger.info(
                            "[%s] hermes-agent elapsed %.2fs, including %.2fs of wrapper retry time",
                            spec.task_id, elapsed_time, excluded_retry_time,
                        )
                    break
                if not provider_error_reason:
                    raise RuntimeError(
                        f"Hermes runner failed without a resumable provider error "
                        f"(rc={agent_proc.returncode})"
                    )
                excluded_retry_time += attempt_elapsed
                if (
                    HERMES_RESUME_ATTEMPTS is not None
                    and resume_attempt >= HERMES_RESUME_ATTEMPTS
                    and unbounded_provider_error(provider_error_reason) is None
                ):
                    raise RuntimeError(f"Hermes runner failed (rc={agent_proc.returncode})")
                if remaining <= 30:
                    raise RuntimeError("Hermes runner failed and no useful time remains for resume")
                if HERMES_RETRY_DELAY_SECONDS > 0:
                    delay_started = time.perf_counter()
                    time.sleep(HERMES_RETRY_DELAY_SECONDS)
                    excluded_retry_time += time.perf_counter() - delay_started
                resume_attempt += 1
                logger.warning(
                    "[%s] Hermes runner exited non-zero%s; retrying same session (%s/%s)",
                    spec.task_id,
                    f" after provider error ({provider_error_reason})" if provider_error_reason else "",
                    resume_attempt,
                    (
                        "unlimited"
                        if unbounded_provider_error(provider_error_reason)
                        else HERMES_RESUME_ATTEMPTS or "unlimited"
                    ),
                )
            self._close_runner_streams(agent_proc)

            logger.info("[%s] hermes-agent exit code: %s", spec.task_id, agent_proc.returncode)
            self._cleanup_bench_config(spec.task_id)

            return AgentExecution(
                elapsed_time=elapsed_time,
                error=None,
                gateway_proc=None,
                agent_proc=agent_proc,
                excluded_retry_time=excluded_retry_time,
            )
        except Exception as exc:
            if agent_proc is not None and agent_proc.poll() is None:
                # Settle a live child on every failure path, not only the
                # explicit wall-clock timeout path, before usage collection.
                self._flush_timed_out_run(spec.task_id, agent_proc)
            elif agent_proc is not None:
                self._close_runner_streams(agent_proc)
            self._cleanup_bench_config(spec.task_id)
            logger.error("[%s] hermes-agent execution error: %s", spec.task_id, exc)
            if start_time is not None:
                elapsed_time = max(0.0, time.perf_counter() - start_time)
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=str(exc),
                gateway_proc=None,
                agent_proc=agent_proc,
                excluded_retry_time=excluded_retry_time,
            )

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict[str, Any]:
        transcript_host = output_dir / "chat.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Keep usage collection independent from grading.  In particular, a
        # timed-out run may have no score path but can still have a native
        # session snapshot that the compat converter can expose.
        try:
            self._write_compat_transcript(task_id)
        except Exception as exc:
            logger.warning("[%s] Compat transcript before usage failed: %s", task_id, exc)
        r_cp = subprocess.run(
            ["docker", "cp", f"{task_id}:{self.transcript_container_path}", str(transcript_host)],
            capture_output=True,
            text=True,
        )
        if r_cp.returncode == 0 and transcript_host.exists():
            usage = extract_usage_from_jsonl(transcript_host)
        else:
            logger.warning("[%s] Transcript copy failed: %s", task_id, r_cp.stderr.strip())
            usage = self._extract_usage_from_session_logs(task_id)

        if self._usage_has_no_tokens(usage):
            log_usage = self._extract_usage_from_agent_log(output_dir / "agent.log")
            if not self._usage_has_no_tokens(log_usage):
                usage = log_usage

        self._copy_session_log(task_id, output_dir)

        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    # ------------------------------------------------------------------
    # Provider / thinking helpers
    # ------------------------------------------------------------------

    def _resolve_runtime_provider(self, model: str, models_config: dict | None) -> tuple[str, str]:
        api_key = self.openrouter_api_key
        base_url = self.openrouter_base_url
        config_key, config_base_url = self._resolve_provider_config(model, models_config)
        if config_key:
            api_key = config_key
        if config_base_url:
            base_url = config_base_url
        return api_key, base_url

    @staticmethod
    def _resolve_provider_config(model: str, models_config: dict | None) -> tuple[str, str]:
        """Extract api_key and base_url from *models_config* for *model*.

        Returns (api_key, base_url) — either or both may be empty strings
        if the config does not contain a matching provider.
        """
        if not models_config:
            return "", ""
        providers = models_config.get("providers", {})
        # Try exact model-id match first.
        for _prov_name, prov in providers.items():
            if not isinstance(prov, dict):
                continue
            for m in prov.get("models", []):
                if isinstance(m, dict) and m.get("id") == model:
                    return prov.get("apiKey", ""), prov.get("baseUrl", "")
        # No exact match — fall back to the first (usually only) provider.
        if providers:
            first = next(iter(providers.values()))
            if isinstance(first, dict):
                return first.get("apiKey", ""), first.get("baseUrl", "")
        return "", ""

    @staticmethod
    def _map_thinking(thinking: str | None) -> dict | None:
        """Map the benchmark ``thinking`` value to a Hermes *reasoning_config* dict."""
        if thinking is None:
            return None
        t = thinking.strip().lower()
        if t in ("off", "none", "disabled", "false"):
            return {"enabled": False}
        if t in ("on", "enabled", "medium", "true"):
            return {"enabled": True, "effort": "medium"}
        if t == "high":
            return {"enabled": True, "effort": "high"}
        if t in ("low", "minimal"):
            return {"enabled": True, "effort": "low"}
        return {"enabled": True, "effort": t}

    # ------------------------------------------------------------------
    # Container setup helpers
    # ------------------------------------------------------------------

    def _start_container(
        self,
        task_id: str,
        workspace_path: str,
        api_key: str,
        base_url: str,
        extra_env: str = "",
        tmp_path: str = "",
        lobster_env: list[str] | None = None,
    ) -> None:
        proxy_http = os.environ.get("HTTP_PROXY_INNER", "")
        proxy_https = os.environ.get("HTTPS_PROXY_INNER", "")
        env_args = [
            "-e", f"http_proxy={proxy_http}",
            "-e", f"https_proxy={proxy_https}",
            "-e", f"HTTP_PROXY={proxy_http}",
            "-e", f"HTTPS_PROXY={proxy_https}",
            "-e", f"OPENAI_API_KEY={api_key}",
            "-e", f"OPENAI_BASE_URL={base_url}",
            "-e", f"no_proxy={'' if not proxy_http else os.environ.get('NO_PROXY_INNER', '')}",
        ]
        if self.brave_api_key:
            env_args += ["-e", f"BRAVE_API_KEY={self.brave_api_key}"]
        for line in extra_env.splitlines():
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            value = os.environ.get(key, "")
            env_args += ["-e", f"{key}={value}"]

        for key in (lobster_env or []):
            value = os.environ.get(key, "")
            if not value:
                continue
            env_args += ["-e", f"{key}={value}"]

        cmd = [
            "docker", "run", "-d",
            "--name", task_id,
            *env_args,
            "-v", f"{workspace_path}:/app:ro",
            self.image,
            "/bin/bash", "-c", "tail -f /dev/null",
        ]
        logger.info("[%s] Starting hermes-agent container (image=%s)", task_id, self.image)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"hermes-agent container startup failed:\n{r.stderr}")
        logger.info("[%s] Container ID: %s", task_id, r.stdout.strip()[:12])

        if tmp_path and os.path.exists(tmp_path):
            subprocess.run(
                ["docker", "exec", task_id, "mkdir", "-p", "/tmp_workspace/tmp"],
                capture_output=True,
            )
            cp_r = subprocess.run(
                ["docker", "cp", f"{tmp_path}/.", f"{task_id}:/tmp_workspace/tmp/"],
                capture_output=True, text=True,
            )
            if cp_r.returncode != 0:
                logger.error("[%s] Temp file copy failed: %s", task_id, cp_r.stderr)

    def _prepare_workspace(self, task_id: str) -> None:
        r = subprocess.run(
            [
                "docker", "exec", task_id, "/bin/bash", "-c",
                f"mkdir -p {TMP_WORKSPACE} && cp -r /app/. {TMP_WORKSPACE} && chmod -R u+w {TMP_WORKSPACE}",
            ],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"hermes-agent workspace copy failed:\n{r.stderr}")

    @staticmethod
    def _yaml_quote(value: object) -> str:
        return json.dumps(str(value), ensure_ascii=True)

    @classmethod
    def _build_hermes_yaml(
        cls, model: str, api_key: str, base_url: str, task_id: str = ""
    ) -> str:
        """Build a config that pins the main and every auxiliary call together."""
        model_value = cls._yaml_quote(model)
        key_value = cls._yaml_quote(api_key)
        base_value = cls._yaml_quote(base_url)
        auxiliary_tasks = (
            "compression",
            "vision",
            "browser_vision",
            "web_extract",
            "session_search",
            "skills_hub",
            "approval",
            "mcp",
            "flush_memories",
            "title_generation",
        )
        lines = [
            "model:",
            f"  default: {model_value}",
            "  provider: custom",
            f"  base_url: {base_value}",
            f"  api_key: {key_value}",
            "  api_mode: codex_responses",
            "  context_length: 128000",
            "terminal:",
            f"  cwd: {TMP_WORKSPACE}",
            "auxiliary:",
        ]
        for task in auxiliary_tasks:
            lines.extend(
                [
                    f"  {task}:",
                    "    provider: custom",
                    f"    model: {model_value}",
                    f"    base_url: {base_value}",
                    f"    api_key: {key_value}",
                    "    api_mode: codex_responses",
                ]
            )
        lines.extend(
            [
                # The per-session JSON snapshot is opt-in as of hermes 0.21.0.
                # ``run_agent._save_session_log`` early-returns unless
                # ``sessions.write_json_snapshots`` is true
                # (``hermes_cli/config_defaults.py`` defaults it to False, read
                # into ``agent._session_json_enabled`` by
                # ``agent/agent_init.py``), because state.db is now canonical
                # and the snapshots have no in-tree consumer upstream. They do
                # have one here: ``~/.hermes/sessions/session_*.json`` is the
                # only thing ``compat_transcript.py`` can build the graded
                # openclaw-shaped transcript from, and without it grading, the
                # usage numbers, the saved artifacts and resume all read an
                # empty run. Payload shape and path are unchanged from 0.9.0 --
                # only the gate is new.
                "sessions:",
                "  write_json_snapshots: true",
                # Pinned, not inherited. hermes-agent's own default is
                # ``approvals.mode: "manual"``
                # (hermes_cli/config.py DEFAULT_CONFIG, deep-merged under any
                # user config), and every other WildClaw baseline states its
                # bypass outright: codex ``--dangerously-bypass-approvals-and-
                # sandbox``, pylm ``--sandbox-mode dangerous_skip``.
                #
                # This block is the *only* thing holding hermes' posture here,
                # and always has been. The container short-circuit at the top
                # of ``tools/approval.py:check_all_command_guards`` -- now
                # ``_should_skip_container_guards`` -- is not a second
                # mechanism backing it up: it keys on ``env_type``, which is
                # hermes' *terminal backend* (``TERMINAL_ENV``, default
                # ``local``), not on whether hermes itself happens to be
                # running inside a container. WildClaw never sets
                # ``TERMINAL_ENV`` and never sets ``terminal.backend``, so
                # ``env_type`` is ``local`` and that branch has never once been
                # taken on this path, at 0.9.0 or at 0.21.0. (The one place in
                # this repository that does take it is
                # ``eval_framework/backends/hermes_backend.py``, which sets
                # ``TERMINAL_ENV=docker`` for its macOS docker-terminal route.)
                # 0.21.0 also adds an unconditional hardline floor -- rm -rf /,
                # mkfs, dd to a raw device, fork bombs -- that runs before
                # ``mode: "off"`` is even read and cannot be bypassed by any
                # config; that is deliberate upstream policy, not a posture
                # this harness can or should declare around.
                #
                # Quoted because bare ``off`` is YAML 1.1 false; hermes
                # normalises that back to "off" (_normalize_approval_mode), but
                # the config should say what it means. Matches
                # eval_framework/backends/hermes_backend.py:2244 and
                # terrarium_agents/hermes_agent.py:978, which both already
                # write this block.
                "approvals:",
                # From src/agents/approval_posture.py, so the value the config
                # carries and the value the run records are one value.
                f'  mode: {cls._yaml_quote(HERMES_POSTURE.config["approvals.mode"])}',
                "tools:",
                "  profile: coding",
                "  web:",
                "    search:",
                "      enabled: true",
                "      provider: brave",
            ]
        )
        if task_id:
            # Stamps this task's correlation id on every request hermes'
            # own outbound client sends -- out-of-band, never inside message
            # content -- so the shared gateway can bucket request_count by
            # task instead of guessing from wall-clock window overlap (the
            # failure mode under ``--parallel`` > 1). Matched to the model's
            # base_url (hermes_cli/config.py's
            # ``get_custom_provider_extra_headers`` matches ``providers``/
            # ``custom_providers`` entries by normalized base_url, not by
            # name), so this needs no named provider id that ``model:``
            # would otherwise have to reference. Matches
            # ``eval_framework/backends/hermes_backend.py``'s use of
            # ``providers.<name>.extra_headers`` for the same purpose.
            lines.extend(
                [
                    "providers:",
                    "  wildclaw_gateway:",
                    f"    base_url: {base_value}",
                    "    extra_headers:",
                    f"      {TASK_ID_HEADER}: {cls._yaml_quote(task_id)}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _configure_hermes(
        self,
        task_id: str,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ) -> None:
        """Configure hermes-agent inside the container with one consistent provider config."""
        # ``task_id`` is already this task's stable, unique identity (it is
        # also the container name), so it doubles as the Gateway correlation
        # id -- no separate identity to invent or thread in.
        hermes_yaml = self._build_hermes_yaml(model, api_key, base_url, task_id=task_id)
        with tempfile.TemporaryDirectory(prefix="hermes_config_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            yaml_host = tmp_root / "hermes.yaml"
            yaml_host.write_text(hermes_yaml, encoding="utf-8")

            r_mkdir = subprocess.run(
                [
                    "docker",
                    "exec",
                    task_id,
                    "/bin/bash",
                    "-c",
                    f"mkdir -p {HERMES_HOME} && mkdir -p $(dirname {OPENCLAW_COMPAT_TRANSCRIPT_PATH})",
                ],
                capture_output=True,
                text=True,
            )
            if r_mkdir.returncode != 0:
                raise RuntimeError(f"hermes-agent config mkdir failed:\n{r_mkdir.stderr}")

            for src, dst in (
                (yaml_host, f"{HERMES_HOME}/hermes.yaml"),
                (yaml_host, f"{HERMES_HOME}/config.yaml"),
            ):
                copied = subprocess.run(
                    ["docker", "cp", str(src), f"{task_id}:{dst}"],
                    capture_output=True,
                    text=True,
                )
                if copied.returncode != 0:
                    raise RuntimeError(f"hermes-agent config copy failed ({dst}):\n{copied.stderr}")

            r_link = subprocess.run(
                ["docker", "exec", task_id, "ln", "-sfn", TMP_WORKSPACE, f"{HERMES_HOME}/workspace"],
                capture_output=True,
                text=True,
            )
            if r_link.returncode != 0:
                raise RuntimeError(f"hermes-agent workspace link failed:\n{r_link.stderr}")

        logger.info("[%s] hermes-agent configured", task_id)

    def _write_bench_runner(
        self,
        task_id: str,
        prompt: str,
        model: str,
        api_key: str,
        base_url: str,
        reasoning_config: dict | None,
    ) -> None:
        """Write the bench runner config into the container."""
        config_payload = {
            "config": {
                "model": model,
                "api_key": api_key,
                "provider": "custom",
                "base_url": base_url,
                "api_mode": "codex_responses",
                "max_iterations": 90,
                "reasoning_config": reasoning_config,
                "session_id": f"wildclaw-{task_id}",
            },
            "prompt": prompt,
        }

        config_tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8",
            ) as f:
                json.dump(config_payload, f, ensure_ascii=False)
                config_tmp = f.name

            r = subprocess.run(
                ["docker", "cp", config_tmp, f"{task_id}:{BENCH_CONFIG_CONTAINER_PATH}"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"Failed to copy {BENCH_CONFIG_CONTAINER_PATH} into container:\n{r.stderr}")
        finally:
            for p in (config_tmp,):
                if p:
                    Path(p).unlink(missing_ok=True)

    def _run_bench_runner_background(
        self, task_id: str, log_path: Path, append: bool = False, resume: bool = False
    ) -> subprocess.Popen[str]:
        if not BENCH_RUNNER_HOST_PATH.exists():
            raise RuntimeError(f"Hermes bench runner script not found: {BENCH_RUNNER_HOST_PATH}")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a" if append else "w", encoding="utf-8")
        script_file = BENCH_RUNNER_HOST_PATH.open("r", encoding="utf-8")
        proc = subprocess.Popen(
            [
                "docker",
                "exec",
                "-i",
                task_id,
                "/bin/bash",
                "-c",
                f"cd {HERMES_INSTALL_DIR} && "
                f"echo $$ > {shlex.quote(HERMES_AGENT_PID_PATH)} && "
                f"WILDCLAW_HERMES_RESUME={'1' if resume else ''} "
                f"exec {HERMES_VENV_PYTHON} -",
            ],
            stdin=script_file,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        proc._log_file = log_file  # type: ignore[attr-defined]
        proc._script_file = script_file  # type: ignore[attr-defined]
        logger.info("[%s] Started Hermes bench runner PID=%s -> %s", task_id, proc.pid, log_path)
        return proc

    @staticmethod
    def _close_runner_streams(proc: subprocess.Popen[str] | None) -> None:
        if proc is None:
            return
        stream = getattr(proc, "_script_file", None)
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        log_file = getattr(proc, "_log_file", None)
        if log_file is not None and not log_file.closed:
            try:
                log_file.close()
            except Exception:
                pass

    @staticmethod
    def _signal_hermes_process(task_id: str, signal_name: str) -> None:
        """Signal the exact in-container Hermes bench runner when possible."""
        if signal_name not in {"INT", "TERM", "KILL"}:
            raise ValueError(f"unsupported signal: {signal_name}")
        pid_file = shlex.quote(HERMES_AGENT_PID_PATH)
        script = f"""
pid_file={pid_file}
if test -s "$pid_file" && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    kill -{signal_name} "$(cat "$pid_file")" 2>/dev/null || true
else
    pkill -{signal_name} -f '[p]ython3 -' 2>/dev/null || true
fi
"""
        try:
            subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "[%s] Hermes timeout signal %s errored: %s",
                task_id,
                signal_name,
                exc,
            )

    def _flush_timed_out_run(
        self, task_id: str, agent_proc: subprocess.Popen | None
    ) -> None:
        """Give Hermes time to persist native session usage before stopping it."""
        if agent_proc is None or agent_proc.poll() is not None:
            if agent_proc is not None:
                self._close_runner_streams(agent_proc)
            return

        self._signal_hermes_process(task_id, "INT")
        try:
            agent_proc.wait(timeout=HERMES_TIMEOUT_FLUSH_SECONDS)
            self._close_runner_streams(agent_proc)
            return
        except subprocess.TimeoutExpired:
            logger.warning(
                "[%s] Hermes did not exit during the %.1fs usage flush grace period",
                task_id,
                HERMES_TIMEOUT_FLUSH_SECONDS,
            )

        self._signal_hermes_process(task_id, "TERM")
        try:
            agent_proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            self._signal_hermes_process(task_id, "KILL")
            try:
                agent_proc.kill()
            except (OSError, ProcessLookupError):
                pass
            try:
                agent_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("[%s] Hermes docker exec did not exit after KILL", task_id)
        finally:
            self._close_runner_streams(agent_proc)

    @classmethod
    def _find_error_marker(
        cls,
        log_path: Path,
        offset: int,
        matcher: Callable[[str], str | None] = resumable_provider_error,
    ) -> str | None:
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as log:
                log.seek(offset)
                text = log.read().lower()
        except OSError:
            return None

        for line in text.splitlines():
            # Hermes reports optional tool capability probes as
            # ``tools.registry ... unavailable (check failed)``.  They are
            # normal in the benchmark image and must not trigger a resume.
            if "tools.registry" in line and "unavailable (check failed)" in line:
                continue
            marker = matcher(line)
            if marker:
                return marker
        return None

    @staticmethod
    def _cleanup_bench_config(task_id: str) -> None:
        subprocess.run(
            ["docker", "exec", task_id, "rm", "-f", BENCH_CONFIG_CONTAINER_PATH],
            capture_output=True,
            text=True,
        )

    # ------------------------------------------------------------------
    # Transcript conversion (all sessions merged)
    # ------------------------------------------------------------------

    def _write_compat_transcript(self, task_id: str) -> None:
        """Convert Hermes session logs to OpenClaw-compatible JSONL for grading."""
        if not COMPAT_TRANSCRIPT_HOST_PATH.exists():
            logger.warning(
                "[%s] Compat transcript script not found: %s",
                task_id,
                COMPAT_TRANSCRIPT_HOST_PATH,
            )
            return

        with COMPAT_TRANSCRIPT_HOST_PATH.open("r", encoding="utf-8") as script_file:
            r = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    task_id,
                    HERMES_VENV_PYTHON,
                    "-",
                ],
                stdin=script_file,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        if r.returncode != 0:
            logger.warning("[%s] Compat transcript write failed: %s", task_id, r.stderr)
        else:
            logger.info("[%s] Compat transcript written to %s", task_id, OPENCLAW_COMPAT_TRANSCRIPT_PATH)

    # ------------------------------------------------------------------
    # Usage extraction (all sessions merged)
    # ------------------------------------------------------------------

    def _extract_usage_from_session_logs(self, task_id: str) -> dict[str, Any]:
        """Fallback: extract usage from copied Hermes session JSON files."""
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }

        with tempfile.TemporaryDirectory(prefix="hermes_usage_") as tmp_dir:
            sessions_host = Path(tmp_dir) / "sessions"
            sessions_host.mkdir(parents=True, exist_ok=True)
            copied = subprocess.run(
                ["docker", "cp", f"{task_id}:{HERMES_HOME}/sessions/.", str(sessions_host)],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                return usage

            total_requests = 0
            for session_file in sorted(sessions_host.glob("session_*.json"), key=lambda p: p.stat().st_mtime):
                try:
                    payload = json.loads(session_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                messages = payload.get("messages", [])
                if not isinstance(messages, list):
                    continue
                total_requests += sum(
                    1 for item in messages if isinstance(item, dict) and item.get("role") == "assistant"
                )
            usage["request_count"] = total_requests

        return usage

    def _usage_has_no_tokens(self, usage: dict[str, Any]) -> bool:
        return (
            usage.get("input_tokens", 0) == 0
            and usage.get("output_tokens", 0) == 0
            and usage.get("total_tokens", 0) == 0
        )

    def _extract_usage_from_agent_log(self, log_path: Path) -> dict[str, Any]:
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }
        if not log_path.exists():
            return usage

        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "API Response received" not in line or (
                "CompletionUsage(" not in line and "ResponseUsage(" not in line
            ):
                continue
            usage["request_count"] += 1
            usage["input_tokens"] += (
                self._extract_int_from_log(line, "prompt_tokens")
                or self._extract_int_from_log(line, "input_tokens")
            )
            usage["output_tokens"] += (
                self._extract_int_from_log(line, "completion_tokens")
                or self._extract_int_from_log(line, "output_tokens")
            )
            usage["total_tokens"] += self._extract_int_from_log(line, "total_tokens")
            usage["cache_read_tokens"] += self._extract_int_from_log(line, "cached_tokens")
            usage["cache_write_tokens"] += self._extract_int_from_log(line, "cache_write_tokens")
            usage["cost_usd"] += self._extract_float_from_log(line, "cost")

        usage["cost_usd"] = round(usage["cost_usd"], 6)
        return usage

    def _extract_int_from_log(self, line: str, field: str) -> int:
        match = re.search(rf"\b{re.escape(field)}=(\d+)", line)
        return int(match.group(1)) if match else 0

    def _extract_float_from_log(self, line: str, field: str) -> float:
        match = re.search(rf"\b{re.escape(field)}=([0-9.eE+-]+)", line)
        return float(match.group(1)) if match else 0.0

    def _copy_session_log(self, task_id: str, output_dir: Path) -> None:
        """Copy *all* hermes session logs from the container to the output directory."""
        hermes_log_dest = output_dir / "hermes_session"
        hermes_log_dest.mkdir(parents=True, exist_ok=True)

        r = subprocess.run(
            ["docker", "cp", f"{task_id}:{HERMES_HOME}/sessions/.", str(hermes_log_dest)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            logger.info("[%s] Hermes session logs copied to %s", task_id, hermes_log_dest)
        else:
            logger.warning("[%s] Hermes session log copy failed: %s", task_id, r.stderr.strip())
