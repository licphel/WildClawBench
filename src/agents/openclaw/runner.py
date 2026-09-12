from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from src.agents.approval_posture import OPENCLAW as OPENCLAW_POSTURE, record as record_posture
from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.utils.grading import extract_usage_from_jsonl
from src.utils.transient_errors import (
    RESUME_ATTEMPTS,
    RESUME_BACKOFF_S,
    resumable_provider_error,
    unbounded_provider_error,
)
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

OPENCLAW_HOME = "/root/.openclaw"
#: The one path every WildClaw baseline's graded transcript lives at. OpenClaw
#: no longer writes it itself (see ``prepare_grading_transcript``), so this is
#: the name of the file the runner puts there, not the name of a file OpenClaw
#: promises.
OPENCLAW_TRANSCRIPT_PATH = f"{OPENCLAW_HOME}/agents/main/sessions/chat.jsonl"

# Ordinary same-session resumes retain the shared three-resume limit. The
# encrypted-session and instant-inference quota signatures bypass it.
OPENCLAW_RESUME_ATTEMPTS: int | None = RESUME_ATTEMPTS
OPENCLAW_RETRY_DELAY_SECONDS = RESUME_BACKOFF_S
OPENCLAW_GATEWAY_STARTUP_TIMEOUT_SECONDS = 120.0
# ``docker exec`` timing out only kills the host-side client.  Let the
# in-container OpenClaw process handle SIGINT and flush its native transcript
# before the runner escalates to TERM/KILL.
OPENCLAW_TIMEOUT_FLUSH_SECONDS = 15.0
OPENCLAW_AGENT_PID_PATH = "/tmp/wildclaw-openclaw-agent.pid"
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
        self.image_model = image_model or ""

    def _resolve_gateway_port(self, task_id: str) -> int:
        """Return a valid free gateway port inside the task container.

        ``run_batch.py`` passes ``0`` as the dynamic-port sentinel.  OpenClaw's
        CLI does not accept ``--port 0`` (unlike Python's socket API), so the
        allocation has to happen before the CLI starts and in the container's
        own network namespace.  Each task has its own container; a short
        bind-and-release probe is therefore sufficient and avoids sharing a
        host-side port with another task.
        """
        if self.gateway_port > 0:
            return self.gateway_port
        probe = subprocess.run(
            [
                "docker",
                "exec",
                task_id,
                "python3",
                "-c",
                (
                    "import socket; "
                    "sock = socket.socket(); "
                    "sock.bind(('127.0.0.1', 0)); "
                    "print(sock.getsockname()[1]); "
                    "sock.close()"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "").strip()
            raise RuntimeError(
                f"Could not allocate an OpenClaw gateway port in the container "
                f"(rc={probe.returncode}): {detail[-500:]}"
            )
        raw_port = (probe.stdout or "").strip().splitlines()
        try:
            port = int(raw_port[-1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(
                f"Container returned an invalid OpenClaw gateway port: "
                f"{probe.stdout!r}"
            ) from exc
        if not 1 <= port <= 65535:
            raise RuntimeError(f"Container returned an out-of-range gateway port: {port}")
        logger.info("[%s] Allocated OpenClaw gateway port inside container: %s", task_id, port)
        return port

    @property
    def expects_gateway(self) -> bool:
        return True

    @property
    def transcript_container_path(self) -> str:
        return OPENCLAW_TRANSCRIPT_PATH

    def prepare_grading_transcript(self, task_id: str) -> str:
        """Guarantee ``chat.jsonl`` exists, whichever way OpenClaw stored it.

        OpenClaw wrote one JSONL file per session under
        ``agents/<id>/sessions/`` up to 2026.3.x. From 2026.9.1 there is no such
        file: the transcript is rows in the ``transcript_events`` table of
        ``<state>/agents/<id>/agent/openclaw-agent.sqlite``, and ``sessionFile``
        survives only as a deprecated "compatibility token; returns the session
        key, not a file path" (``src/agents/sessions/agent-session-base.ts``).
        Every consumer here reads the file: the grader loads it through
        ``transcript_loader.load_transcript`` inside the container, and
        ``collect_usage`` copies it out and runs ``extract_usage_from_jsonl``
        over it. A missing file is not an error anywhere -- grading sees an
        empty transcript and usage reports all zeros -- so the break would have
        been silent.

        The stored events are byte-identical to the lines the JSONL carried, so
        this is a container change and not a format change: the rows are
        emitted in ``(session_id, seq)`` order and nothing is reshaped. The
        existing file wins when there is one, which keeps this a no-op on an
        image still carrying an OpenClaw that writes JSONL -- as the WildClaw
        images do today -- and makes it the source once they are rebuilt.
        """

        self._materialize_transcript(task_id)
        return self.transcript_container_path

    def _materialize_transcript(self, task_id: str) -> None:
        script = f"""python3 - <<'PY'
import glob
import json
import os
import sqlite3

out = {json.dumps(OPENCLAW_TRANSCRIPT_PATH)}
if os.path.exists(out) and os.path.getsize(out) > 0:
    raise SystemExit(0)

rows = []
for db in sorted(glob.glob({json.dumps(OPENCLAW_HOME)} + "/**/openclaw-agent.sqlite", recursive=True)):
    try:
        conn = sqlite3.connect("file:" + db + "?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        continue
    try:
        rows.extend(
            conn.execute(
                "select session_id, seq, event_json from transcript_events "
                "order by session_id, seq"
            ).fetchall()
        )
    except sqlite3.Error:
        # A run that never reached the agent leaves the table absent.
        pass
    finally:
        conn.close()

os.makedirs(os.path.dirname(out), exist_ok=True)
written = 0
with open(out, "w", encoding="utf-8") as handle:
    for _session_id, _seq, event_json in rows:
        try:
            event = json.loads(event_json)
        except (TypeError, ValueError):
            continue
        handle.write(json.dumps(event, ensure_ascii=False) + "\\n")
        written += 1
print("materialized %d transcript event(s) from SQLite" % written)
PY"""
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", script],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            # Loud, never fatal: a transcript this could not build leaves the
            # run exactly where it already was, and grading still runs.
            logger.warning(
                "[%s] Could not materialize the openclaw transcript: %s",
                task_id,
                (r.stderr or "").strip()[:400],
            )
        elif (r.stdout or "").strip():
            logger.info("[%s] %s", task_id, (r.stdout or "").strip())

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        gateway_proc = None
        agent_proc = None
        elapsed_time = 0.0
        start_time: float | None = None

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

            self._apply_approval_posture(spec.task_id, spec.output_dir)
            self._set_model(spec.task_id, spec.model)
            self._inject_openrouter_key(spec.task_id)
            image_model = self.image_model or spec.model
            self._set_image_model(spec.task_id, image_model)

            gateway_port = self._resolve_gateway_port(spec.task_id)

            gateway_proc = run_background(
                spec.task_id,
                bash_cmd=(
                    f"export OPENROUTER_API_KEY='{self.openrouter_api_key}' && "
                    f"export OPENROUTER_BASE_URL='{self.openrouter_base_url}' && "
                    f"export OPENCLAW_GATEWAY_PORT='{gateway_port}' && "
                    "for attempt in 1 2; do "
                    f"openclaw gateway run --port {gateway_port} "
                    "--bind loopback --allow-unconfigured; "
                    "status=$?; "
                    "if [ $status -eq 0 ] || [ $attempt -eq 2 ]; then "
                    "exit $status; fi; "
                    "sleep 1; "
                    "done"
                ),
                log_path=spec.output_dir / "gateway.log",
            )
            logger.info(
                "[%s] Waiting for OpenClaw gateway health (timeout=%ss)...",
                spec.task_id,
                OPENCLAW_GATEWAY_STARTUP_TIMEOUT_SECONDS,
            )
            gateway_deadline = (
                time.monotonic() + OPENCLAW_GATEWAY_STARTUP_TIMEOUT_SECONDS
            )
            while True:
                if gateway_proc.poll() is not None:
                    gateway_log = spec.output_dir / "gateway.log"
                    detail = ""
                    if gateway_log.exists():
                        detail = gateway_log.read_text(
                            encoding="utf-8", errors="replace"
                        )[-2000:]
                    raise RuntimeError(
                        "OpenClaw gateway exited before becoming ready "
                        f"(rc={gateway_proc.returncode}):\n{detail}"
                    )
                health = subprocess.run(
                    [
                        "docker",
                        "exec",
                        spec.task_id,
                        "/bin/bash",
                        "-c",
                        f"OPENCLAW_GATEWAY_PORT='{gateway_port}' openclaw health",
                    ],
                    capture_output=True,
                    text=True,
                )
                if health.returncode == 0:
                    logger.info(
                        "[%s] OpenClaw gateway is ready: %s",
                        spec.task_id,
                        (health.stdout or "").strip()[:300],
                    )
                    break
                if time.monotonic() >= gateway_deadline:
                    detail = (health.stderr or health.stdout or "").strip()
                    raise RuntimeError(
                        "OpenClaw gateway did not become ready within "
                        f"{OPENCLAW_GATEWAY_STARTUP_TIMEOUT_SECONDS}s: {detail}"
                    )
                time.sleep(0.5)

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
                remaining = int(spec.timeout_seconds - counted_elapsed)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        cmd="OpenClaw task",
                        timeout=spec.timeout_seconds,
                    )
                remaining = max(1, remaining)
                message = safe_prompt if resume_attempt == 0 else safe_resume_prompt
                agent_proc = run_background(
                    spec.task_id,
                    bash_cmd=(
                        # The gateway is allocated per container; keep the
                        # agent CLI on that port instead of its default 18789.
                        f"export OPENCLAW_GATEWAY_PORT='{gateway_port}' && "
                        f"echo $$ > {shlex.quote(OPENCLAW_AGENT_PID_PATH)} && "
                        f"exec openclaw agent --session-id chat --timeout {remaining} "
                        f"--message '{message}'"
                    ),
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
                    self._flush_timed_out_run(spec.task_id, agent_proc)
                    return AgentExecution(
                        elapsed_time=elapsed_time,
                        error="OpenClaw run timed out",
                        gateway_proc=gateway_proc,
                        agent_proc=agent_proc,
                    )
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
                    raise RuntimeError(
                        "OpenClaw agent failed without a resumable provider error "
                        f"(rc={agent_proc.returncode})"
                    )
                excluded_retry_time += attempt_elapsed
                if (
                    OPENCLAW_RESUME_ATTEMPTS is not None
                    and resume_attempt >= OPENCLAW_RESUME_ATTEMPTS
                    and unbounded_provider_error(provider_error_reason) is None
                ):
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
                    (
                        "unlimited"
                        if unbounded_provider_error(provider_error_reason)
                        else OPENCLAW_RESUME_ATTEMPTS or "unlimited"
                    ),
                )

            logger.info("[%s] Agent exit code: %s", spec.task_id, agent_proc.returncode)
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=None,
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )
        except subprocess.TimeoutExpired:
            # This covers a deadline reached between retry attempts, before a
            # new child was started.  If a live child exists, flush it using
            # the same path as the direct wait timeout.
            elapsed_time = float(spec.timeout_seconds)
            self._flush_timed_out_run(spec.task_id, agent_proc)
            return AgentExecution(
                elapsed_time=elapsed_time,
                error="OpenClaw run timed out",
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )
        except Exception as exc:
            logger.error("[%s] Execution error: %s", spec.task_id, exc)
            if agent_proc is not None and agent_proc.poll() is None:
                # Any unexpected runner failure must still leave the native
                # transcript in a settled state before collect_usage runs.
                self._flush_timed_out_run(spec.task_id, agent_proc)
            if start_time is not None:
                elapsed_time = max(0.0, time.perf_counter() - start_time)
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=str(exc),
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )

    @staticmethod
    def _signal_openclaw_process(task_id: str, signal_name: str) -> None:
        """Signal the exact in-container OpenClaw CLI when possible."""
        if signal_name not in {"INT", "TERM", "KILL"}:
            raise ValueError(f"unsupported signal: {signal_name}")
        pid_file = shlex.quote(OPENCLAW_AGENT_PID_PATH)
        script = f"""
pid_file={pid_file}
if test -s "$pid_file" && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    kill -{signal_name} "$(cat "$pid_file")" 2>/dev/null || true
else
    pkill -{signal_name} -f '[o]penclaw agent' 2>/dev/null || true
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
                "[%s] OpenClaw timeout signal %s errored: %s",
                task_id,
                signal_name,
                exc,
            )

    def _flush_timed_out_run(
        self, task_id: str, agent_proc: subprocess.Popen | None
    ) -> None:
        """Give OpenClaw time to persist native usage before force stopping it."""
        if agent_proc is None or agent_proc.poll() is not None:
            if agent_proc is not None:
                close_proc_log(agent_proc)
            return

        self._signal_openclaw_process(task_id, "INT")
        try:
            agent_proc.wait(timeout=OPENCLAW_TIMEOUT_FLUSH_SECONDS)
            close_proc_log(agent_proc)
            return
        except subprocess.TimeoutExpired:
            logger.warning(
                "[%s] OpenClaw did not exit during the %.1fs usage flush grace period",
                task_id,
                OPENCLAW_TIMEOUT_FLUSH_SECONDS,
            )

        self._signal_openclaw_process(task_id, "TERM")
        try:
            agent_proc.wait(timeout=3)
            close_proc_log(agent_proc)
            return
        except subprocess.TimeoutExpired:
            self._signal_openclaw_process(task_id, "KILL")
            try:
                agent_proc.kill()
            except (OSError, ProcessLookupError):
                pass
            try:
                agent_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("[%s] OpenClaw docker exec did not exit after KILL", task_id)
        finally:
            close_proc_log(agent_proc)

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
        # New OpenClaw versions keep transcript_events in SQLite.  Materialize
        # it here as well as in the grading path: a timeout/error must not lose
        # native usage merely because grading was skipped or failed.
        try:
            self._materialize_transcript(task_id)
        except Exception as exc:
            logger.warning("[%s] Transcript materialization before usage failed: %s", task_id, exc)
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
        """Save the OpenRouter key where the running OpenClaw will look for it.

        Two stores, one per era. Up to 2026.3.x the credential was a JSON file
        at ``agents/<id>/agent/auth-profiles.json`` and writing it was the whole
        job. From 2026.9.1 auth is SQLite, and that file is not merely ignored:
        it is *detected* by name and makes OpenClaw refuse to start at all --
        ``AuthProfileMigrationRequiredError``, "requires legacy credential
        migration; run openclaw doctor --fix"
        (``src/agents/auth-profiles/legacy-source-diagnostic.ts``). So writing
        it unconditionally would take the whole baseline down once the image is
        rebuilt.

        ``models auth paste-api-key`` is the CLI that owns the current store; it
        reads the key from stdin, so the key never appears in an argv a
        ``docker exec`` would log, and it writes both the secret and the
        non-secret ``auth.profiles`` descriptor in config. It does not exist on
        2026.3.11 (that CLI has ``paste-token`` only), which is exactly what
        selects the legacy write for an image that still needs it.
        """

        if not self.openrouter_api_key:
            return

        modern = subprocess.run(
            [
                "docker", "exec", "-i", task_id, "/bin/bash", "-c",
                "openclaw models auth paste-api-key "
                "--provider openrouter --profile-id openrouter:default",
            ],
            input=self.openrouter_api_key + "\n",
            capture_output=True,
            text=True,
        )
        if modern.returncode == 0:
            logger.info(
                "[%s] Saved OPENROUTER_API_KEY via openclaw models auth paste-api-key",
                task_id,
            )
            return
        logger.info(
            "[%s] openclaw models auth paste-api-key unavailable (%s); "
            "writing the legacy auth-profiles.json instead",
            task_id,
            (modern.stderr or modern.stdout or "").strip().splitlines()[-1:] or "",
        )

        auth_profile_path = f"{OPENCLAW_HOME}/agents/main/agent/auth-profiles.json"
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

    def _apply_approval_posture(self, task_id: str, output_dir: Path) -> None:
        """Write openclaw's approval posture from the harness, not the image.

        OpenClaw has no bypass flag, so the declaration has to be config -- and
        it used to be config baked into a Docker layer
        (``/root/.openclaw/openclaw.json``), which meant the only way to learn
        what policy an openclaw run had was to open the image. These are the
        same three keys ``eval_framework/backends/openclaw_backend.py`` writes
        on the outer path. What is written, and whether each write
        succeeded, is recorded next to the task's other artifacts.
        """

        applied: dict[str, object] = {"config": {}, "files": {}}
        # 2026.9.1 treats this JSON as a legacy-migration marker and refuses
        # agent requests while it exists. Remove it defensively so an older
        # image cannot invalidate the declared config posture.
        legacy_path = "/root/.openclaw/exec-approvals.json"
        remove_legacy = subprocess.run(
            ["docker", "exec", task_id, "rm", "-f", legacy_path],
            capture_output=True,
            text=True,
        )
        applied["legacy_exec_approvals_absent"] = {
            "path": legacy_path,
            "returncode": remove_legacy.returncode,
            "stderr": (remove_legacy.stderr or "").strip()[:400],
        }
        for key, value in OPENCLAW_POSTURE.config.items():
            if isinstance(value, (list, dict)):
                cli_value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            elif isinstance(value, bool):
                cli_value = "true" if value else "false"
            else:
                cli_value = str(value)
            r = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c",
                 f"openclaw config set {shlex.quote(key)} {shlex.quote(cli_value)}"],
                capture_output=True, text=True,
            )
            applied["config"][key] = {
                "value": value,
                "returncode": r.returncode,
                "stderr": (r.stderr or "").strip()[:400],
            }
            if r.returncode != 0:
                # Loud, but not fatal: the image still carries the same values,
                # so a failed write leaves the run in the state it was already
                # in rather than in an undeclared one. The artifact says which.
                logger.warning(
                    "[%s] openclaw config set %s failed: %s",
                    task_id, key, (r.stderr or "").strip(),
                )
        for path, payload in OPENCLAW_POSTURE.files.items():
            body = json.dumps(payload, ensure_ascii=False)
            r = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c",
                 f"mkdir -p {shlex.quote(str(Path(path).parent))} && "
                 f"printf %s {shlex.quote(body)} > {shlex.quote(path)}"],
                capture_output=True, text=True,
            )
            applied["files"][path] = {
                "returncode": r.returncode,
                "stderr": (r.stderr or "").strip()[:400],
            }
            if r.returncode != 0:
                logger.warning(
                    "[%s] writing %s failed: %s", task_id, path, (r.stderr or "").strip()
                )
        readback = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c",
             "test ! -e /root/.openclaw/exec-approvals.json; "
             "echo legacy_exec_approvals_absent=$?; "
             "openclaw config get tools.exec.security 2>&1; "
             "openclaw config get tools.exec.ask 2>&1; "
             "openclaw config get tools.deny 2>&1"],
            capture_output=True, text=True,
        )
        applied["readback"] = (readback.stdout or "").strip()[:2000]
        record_posture(output_dir, OPENCLAW_POSTURE, applied=applied)

    def _set_image_model(self, task_id: str, model: str) -> None:
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", f"openclaw config set agents.defaults.imageModel.primary '{model}'"],
            capture_output=True,
            text=True,
        )
        logger.info("[%s] imageModel set: %s", task_id, model)
