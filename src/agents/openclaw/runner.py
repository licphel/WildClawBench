from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.utils.grading import extract_usage_from_jsonl
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
from src.utils.transient_errors import is_transient_error

load_dotenv()

logger = logging.getLogger(__name__)

# `openclaw agent` exits 0 even when the embedded run itself failed (e.g. the
# upstream model provider returning a transient error) — it only logs
# `isError=true ... error=<message>` to gateway.log and ends the session with
# no output. Left unchecked, run_batch.py records this as a clean, low/zero
# score with no error, which also makes it invisible to the transient-error
# retries below. Scan gateway.log for that marker so the failure surfaces as
# a real execution error instead.
_EMBEDDED_RUN_ERROR_RE = re.compile(r"embedded run agent end:.*isError=true.*?error=(.+)$", re.MULTILINE)

MAX_INNER_RETRIES = 2


def _embedded_run_error_in(text: str) -> str | None:
    match = _EMBEDDED_RUN_ERROR_RE.search(text)
    return match.group(1).strip() if match else None


def _read_new_content(path: Path, from_offset: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(from_offset)
            return f.read()
    except OSError:
        return ""


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

            gateway_log_path = spec.output_dir / "gateway.log"
            safe_prompt = spec.prompt.replace("'", "'\\''")
            start_time = time.perf_counter()
            deadline = start_time + spec.timeout_seconds
            total_attempts = MAX_INNER_RETRIES + 1
            attempt = 1
            embedded_error = None
            timed_out = False
            while True:
                remaining = max(1, int(deadline - time.perf_counter()))
                agent_log_path = spec.output_dir / (
                    "agent.log" if attempt == 1 else f"agent.retry{attempt}.log"
                )
                offset_before = gateway_log_path.stat().st_size if gateway_log_path.exists() else 0
                agent_proc = run_background(
                    spec.task_id,
                    bash_cmd=f"openclaw agent --session-id chat --timeout {remaining} --message '{safe_prompt}'",
                    log_path=agent_log_path,
                )

                logger.info("[%s] Waiting for agent to finish (attempt %d/%d)...", spec.task_id, attempt, total_attempts)
                try:
                    agent_proc.wait(timeout=remaining)
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(
                        "[%s] Agent finished, elapsed: %.2f seconds (attempt %d)",
                        spec.task_id, elapsed_time, attempt,
                    )
                except subprocess.TimeoutExpired:
                    logger.info("[%s] Agent timed out...", spec.task_id)
                    elapsed_time = float(spec.timeout_seconds)
                    agent_proc.kill()
                    agent_proc.wait()
                    timed_out = True

                close_proc_log(agent_proc)
                logger.info("[%s] Agent exit code: %s (attempt %d)", spec.task_id, agent_proc.returncode, attempt)

                if timed_out:
                    break

                embedded_error = _embedded_run_error_in(_read_new_content(gateway_log_path, offset_before))
                if not embedded_error:
                    break
                if not is_transient_error(embedded_error):
                    logger.warning(
                        "[%s] Embedded run failed with a non-transient error, not retrying in-container: %s",
                        spec.task_id, embedded_error,
                    )
                    break
                if attempt >= total_attempts or time.perf_counter() >= deadline:
                    logger.warning(
                        "[%s] Transient embedded-run error, out of in-container retries/time budget: %s",
                        spec.task_id, embedded_error,
                    )
                    break
                logger.warning(
                    "[%s] Transient embedded-run error (attempt %d/%d), retrying in the same container: %s",
                    spec.task_id, attempt, total_attempts, embedded_error,
                )
                attempt += 1

            if embedded_error and not timed_out:
                logger.warning(
                    "[%s] Embedded run failed despite exit code 0: %s",
                    spec.task_id, embedded_error,
                )
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=embedded_error if not timed_out else None,
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
