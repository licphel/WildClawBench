from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.utils.grading import extract_usage_from_jsonl
from src.utils.transient_errors import resumable_provider_error
from src.utils.docker_utils import (
    close_proc_log,
    inject_lobster_workspace,
    inject_openclaw_models,
    run_background,
    run_warmup,
    setup_skills,
    setup_workspace,
    start_container,
)

load_dotenv()

logger = logging.getLogger(__name__)

OPENCLAW_RESUME_ATTEMPTS = int(os.environ.get("OPENCLAW_RESUME_ATTEMPTS", "0"))
OPENCLAW_RETRY_DELAY_SECONDS = float(os.environ.get("OPENCLAW_RETRY_DELAY_SECONDS", "2"))
OPENCLAW_RESUME_PREFIX = (
    "A previous attempt of this same task was interrupted by a transient "
    "provider error. Continue from the current workspace, preserve and "
    "verify completed work, and finish every required output. Do not "
    "restart the task or discuss the interruption. The original task is:\n\n"
)


class OpenClawAgent(BaseAgent):
    def __init__(
        self,
        gateway_port: int,
        openrouter_api_key: str = "",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        image_model: str | None = None,
    ) -> None:
        self.gateway_port = gateway_port
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = openrouter_base_url
        self.image_model = image_model if image_model is not None else os.environ.get("OPENCLAW_IMAGE_MODEL", "").strip()

    @property
    def expects_gateway(self) -> bool:
        return True

    @property
    def transcript_container_path(self) -> str:
        return "/root/.openclaw/agents/main/sessions/chat.jsonl"

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        gateway_proc = None
        agent_proc = None
        elapsed_time = float(spec.timeout_seconds)

        try:
            exec_path = os.path.join(spec.workspace_path, "exec")
            tmp_path = os.path.join(spec.workspace_path, "tmp")
            os.makedirs(exec_path, exist_ok=True)

            start_container(
                spec.task_id,
                exec_path,
                extra_env=spec.task.get("env", ""),
                tmp_path=tmp_path,
                lobster_env=spec.lobster.get("env") if spec.lobster else None,
            )
            if spec.lobster:
                inject_lobster_workspace(spec.task_id, spec.lobster["workspace"])

            setup_workspace(spec.task_id, thinking=spec.thinking)
            setup_skills(spec.task_id, spec.task.get("skills", ""), spec.task.get("skills_path", ""))
            run_warmup(spec.task_id, spec.task.get("warmup", ""))

            if spec.models_config:
                inject_openclaw_models(spec.task_id, spec.models_config)

            self._set_model(spec.task_id, spec.model)
            self._inject_openrouter_key(spec.task_id)
            image_model = self.image_model or spec.model
            self._set_image_model(spec.task_id, image_model)

            gateway_proc = run_background(
                spec.task_id,
                bash_cmd=(
                    f"export OPENROUTER_API_KEY='{self.openrouter_api_key}' && "
                    f"export OPENROUTER_BASE_URL='{self.openrouter_base_url}' && "
                    f"openclaw gateway --port {self.gateway_port}"
                ),
                log_path=spec.output_dir / "gateway.log",
            )
            logger.info("[%s] Waiting for gateway to be ready (2s)...", spec.task_id)
            time.sleep(2)

            safe_prompt = spec.prompt.replace("'", "'\\''")
            safe_resume_prompt = (OPENCLAW_RESUME_PREFIX + spec.prompt).replace(
                "'", "'\\''"
            )
            agent_log = spec.output_dir / "agent.log"
            start_time = time.perf_counter()
            excluded_retry_time = 0.0
            resume_attempt = 0
            while True:
                log_offset = agent_log.stat().st_size if agent_log.exists() else 0
                counted_elapsed = time.perf_counter() - start_time - excluded_retry_time
                remaining = max(1, int(spec.timeout_seconds - counted_elapsed))
                message = safe_prompt if resume_attempt == 0 else safe_resume_prompt
                agent_proc = run_background(
                    spec.task_id,
                    bash_cmd=f"openclaw agent --session-id chat --timeout {remaining} --message '{message}'",
                    log_path=agent_log,
                    append=resume_attempt > 0,
                )

                logger.info("[%s] Waiting for agent to finish...", spec.task_id)
                attempt_started = time.perf_counter()
                try:
                    agent_proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    logger.info("[%s] Agent timed out...", spec.task_id)
                    elapsed_time = float(spec.timeout_seconds)
                    agent_proc.kill()
                    agent_proc.wait()
                    break
                attempt_elapsed = time.perf_counter() - attempt_started
                # Retry time stays INSIDE elapsed_time, deliberately; see the
                # note in the claudecode runner.  This runner could never
                # deduct the openclaw CLI's own reconnects (MAX_RETRIES = 5
                # with 1s/2s/4s/8s/16s backoff, in
                # baselines/openclaw/src/agents/openai-ws-connection.ts)
                # anyway, so deducting only the wrapper's half made the number
                # neither inclusive nor exclusive.  excluded_retry_time is
                # still accumulated: the task budget still refunds a resumed
                # attempt.
                elapsed_time = time.perf_counter() - start_time
                if agent_proc.returncode == 0:
                    logger.info(
                        "[%s] Agent finished successfully, elapsed: %.2f seconds",
                        spec.task_id,
                        elapsed_time,
                    )
                    break
                provider_error_reason = self._find_error_marker(agent_log, log_offset)
                if provider_error_reason is None:
                    # Unchanged from before this loop existed: a non-zero exit
                    # carrying no provider signature is the agent's own failure,
                    # and the workspace it left behind is still the measurement.
                    break
                excluded_retry_time += attempt_elapsed
                if OPENCLAW_RESUME_ATTEMPTS > 0 and resume_attempt >= OPENCLAW_RESUME_ATTEMPTS:
                    raise RuntimeError(
                        f"OpenClaw agent failed after a provider error "
                        f"({provider_error_reason}) with no resume attempts left "
                        f"(rc={agent_proc.returncode})"
                    )
                if remaining <= 30:
                    raise RuntimeError(
                        f"OpenClaw agent failed after a provider error "
                        f"({provider_error_reason}) and no useful time remains for resume"
                    )
                if OPENCLAW_RETRY_DELAY_SECONDS > 0:
                    delay_started = time.perf_counter()
                    time.sleep(OPENCLAW_RETRY_DELAY_SECONDS)
                    excluded_retry_time += time.perf_counter() - delay_started
                close_proc_log(agent_proc)
                resume_attempt += 1
                logger.warning(
                    "[%s] OpenClaw agent exited non-zero after a provider error (%s); "
                    "retrying the same session (%s/%s)",
                    spec.task_id,
                    provider_error_reason,
                    resume_attempt,
                    OPENCLAW_RESUME_ATTEMPTS or "unlimited",
                )

            logger.info("[%s] Agent exit code: %s", spec.task_id, agent_proc.returncode)
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=None,
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )
        except Exception as exc:
            logger.error("[%s] Execution error: %s", spec.task_id, exc)
            return AgentExecution(
                elapsed_time=float(spec.timeout_seconds),
                error=str(exc),
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )

    @staticmethod
    def _find_error_marker(log_path: Path, offset: int) -> str | None:
        """Return the provider-failure signature this attempt logged, if any."""
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as log:
                log.seek(offset)
                return resumable_provider_error(log.read())
        except OSError:
            return None

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict:
        transcript_host = output_dir / "chat.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        r_cp = subprocess.run(
            ["docker", "cp", f"{task_id}:{self.transcript_container_path}", str(transcript_host)],
            capture_output=True,
            text=True,
        )
        if r_cp.returncode == 0 and transcript_host.exists():
            usage = extract_usage_from_jsonl(transcript_host)
        else:
            logger.warning("[%s] Transcript copy failed: %s", task_id, r_cp.stderr.strip())
            usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "request_count": 0,
            }
        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    def _set_model(self, task_id: str, model: str) -> None:
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", f"openclaw models set '{model}'"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Model setup failed:\n{r.stderr}")
        logger.info("[%s] Model set: %s", task_id, model)

    def _inject_openrouter_key(self, task_id: str) -> None:
        if not self.openrouter_api_key:
            return

        auth_profile_path = "/root/.openclaw/agents/main/agent/auth-profiles.json"
        inject_cmd = f"""python3 - <<'PY'
import json
import pathlib

p = pathlib.Path("{auth_profile_path}")
d = json.loads(p.read_text()) if p.exists() else {{"version": 1, "profiles": {{}}}}
d.setdefault("profiles", {{}})["openrouter:default"] = {{
    "type": "api_key",
    "provider": "openrouter",
    "key": {json.dumps(self.openrouter_api_key)}
}}
p.write_text(json.dumps(d, indent=2))
PY"""
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", inject_cmd],
            capture_output=True,
            text=True,
        )
        logger.info("[%s] Injected OPENROUTER_API_KEY into auth-profiles.json", task_id)

    def _set_image_model(self, task_id: str, model: str) -> None:
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", f"openclaw config set agents.defaults.imageModel.primary '{model}'"],
            capture_output=True,
            text=True,
        )
        logger.info("[%s] imageModel set: %s", task_id, model)
