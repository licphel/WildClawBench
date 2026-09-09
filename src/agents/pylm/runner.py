from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from src.agents.approval_posture import PERDURA as PERDURA_POSTURE, record as record_posture
from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent


class PyLMAgent(BaseAgent):
    """WildClawBench ``run_batch`` backend for the PyLM CLI.

    The container-side entrypoint remains the stable PyLM CLI contract.  This
    class only adapts its lifecycle to WildClawBench's official BaseAgent
    interface so that task parsing, the common runner prompt, grading, usage
    collection, and cleanup all happen in ``eval/run_batch.py``.
    """

    transcript_path = "/root/.openclaw/agents/main/sessions/chat.jsonl"

    def __init__(self, image: str | None = None) -> None:
        self.image = (image or os.environ.get("WILDCLAW_PERDURA_IMAGE", "")).strip()
        if not self.image:
            raise ValueError(
                "WILDCLAW_PERDURA_IMAGE must be set when no Perdura image is passed"
            )
        self._summaries: dict[str, dict] = {}
        self._staging_dirs: dict[str, str] = {}

    @property
    def expects_gateway(self) -> bool:
        return False

    @property
    def transcript_container_path(self) -> str:
        return self.transcript_path

    @staticmethod
    def _repo_root() -> Path:
        # .../PyLM_Eval/benchmarks/WildClawBench/src/agents/pylm/runner.py
        return Path(__file__).resolve().parents[5]

    @classmethod
    def _import_cli_runner(cls):
        root = cls._repo_root()
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        import eval_framework.wildclaw_cli_runner as cli_runner

        return cli_runner

    @staticmethod
    def _workspace_dirs(workspace_path: str) -> tuple[str, str, str | None]:
        canonical = Path(workspace_path)
        canonical_exec = canonical / "exec"
        canonical_tmp = canonical / "tmp"
        try:
            canonical_exec.mkdir(parents=True, exist_ok=True)
            canonical_tmp.mkdir(parents=True, exist_ok=True)
            return str(canonical_exec), str(canonical_tmp), None
        except PermissionError:
            staging = tempfile.mkdtemp(prefix="wildclaw_pylm_ws_")
            exec_path = Path(staging) / "exec"
            tmp_path = Path(staging) / "tmp"
            exec_path.mkdir(parents=True, exist_ok=True)
            tmp_path.mkdir(parents=True, exist_ok=True)
            if canonical_exec.is_dir():
                shutil.copytree(canonical_exec, exec_path, dirs_exist_ok=True)
            if canonical_tmp.is_dir():
                shutil.copytree(canonical_tmp, tmp_path, dirs_exist_ok=True)
            return str(exec_path), str(tmp_path), staging

    @staticmethod
    def _skill_paths(task: dict) -> tuple[str, ...]:
        paths: list[str] = []
        for line in str(task.get("skills") or "").splitlines():
            line = line.strip()
            if not line:
                continue
            paths.append(f"/tmp_workspace/skills/{line.replace(chr(92), '/').strip('/').split('/')[-1]}")
        return tuple(paths)

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        self._summaries.pop(spec.task_id, None)
        cli_runner = self._import_cli_runner()
        from src.utils import docker_utils
        from src.utils.docker_utils import (
            run_warmup,
            setup_skills,
            setup_workspace,
            start_container,
        )

        # docker_utils reads DOCKER_IMAGE at import time.  run_batch imports
        # that module before constructing this backend, so update the same
        # module object rather than relying on a late environment variable.
        docker_utils.DOCKER_IMAGE = self.image

        exec_path, tmp_path, staging = self._workspace_dirs(
            spec.workspace_path
        )
        if staging is not None:
            self._staging_dirs[spec.task_id] = staging

        try:
            cli_runner._start_container_with_live_mounts(
                start_container,
                spec.task_id,
                exec_path,
                extra_env=str(spec.task.get("env") or ""),
                tmp_path=tmp_path,
                image=self.image,
                enable_nested_isolation=os.environ.get(
                    "WILDCLAW_PYLM_NESTED_ISOLATION", ""
                ).lower() in {"1", "true", "yes"},
                pylm_store_host_dir=spec.output_dir / "pylm_store",
            )
            setup_workspace(spec.task_id)
            setup_skills(
                spec.task_id,
                str(spec.task.get("skills") or ""),
                str(spec.task.get("skills_path") or ""),
                container_skills_root="/tmp_workspace/skills",
            )
            run_warmup(
                spec.task_id,
                str(spec.task.get("warmup") or ""),
                detach_background=True,
            )

            # Default from src/agents/approval_posture.py rather than a
            # literal here, so the declared posture and the launched one cannot
            # drift apart. The env var stays as the override.
            sandbox_mode = os.environ.get(
                "WILDCLAW_PYLM_SANDBOX_MODE", PERDURA_POSTURE.argv[1]
            )
            confirm_dangerous_skip = sandbox_mode == "dangerous_skip"
            non_interactive = "--non-interactive" in PERDURA_POSTURE.argv
            # Written twice on purpose. Once here, so a task that dies
            # mid-run still leaves its intended posture on disk; then again
            # below with what the CLI actually accepted, which is the value
            # worth having and the one that can differ -- the container entry
            # resolves --sandbox-mode against a live `perdura run --help`, and
            # the two CLI generations answer differently.
            posture_intent = {
                "delivered_as": "perdura CLI argv",
                "sandbox_mode": sandbox_mode,
                "overridden_by_env": sandbox_mode != PERDURA_POSTURE.argv[1],
                # wildclaw_cli_container_entry.py probes `perdura run --help`
                # and only appends this when the build still accepts it;
                # current perdura does not (sandbox_confirmation.py: "no
                # separate confirmation flag exists").
                "confirm_dangerous_skip_requested": confirm_dangerous_skip,
                "non_interactive_requested": non_interactive,
                "resolved": "pending: container has not reported yet",
            }
            record_posture(spec.output_dir, PERDURA_POSTURE, applied=posture_intent)
            summary, elapsed, error = cli_runner._run_container_cli(
                task_id=spec.task_id,
                prompt=spec.prompt,
                model=spec.model,
                benchmark_task_name="task",
                reasoning_effort=spec.thinking,
                sandbox_mode=sandbox_mode,
                confirm_dangerous_skip=confirm_dangerous_skip,
                non_interactive=non_interactive,
                timeout_seconds=spec.timeout_seconds,
                output_dir=spec.output_dir,
                plugin_paths=self._skill_paths(spec.task),
            )
            # `sandbox` is the container entry's own account of what
            # `perdura run` was actually given (wildclaw_cli_container_entry.py
            # -> cli_result.json). Absent only when the run died before the
            # entry returned, and that absence is itself worth recording.
            posture_intent["resolved"] = summary.get("sandbox") or {
                "note": "container reported no sandbox block; see cli_result.json",
            }
            record_posture(spec.output_dir, PERDURA_POSTURE, applied=posture_intent)
            self._summaries[spec.task_id] = summary
            return AgentExecution(
                elapsed_time=elapsed,
                error=error,
                gateway_proc=None,
                agent_proc=None,
            )
        except Exception as exc:
            self._summaries[spec.task_id] = {}
            return AgentExecution(
                elapsed_time=float(spec.timeout_seconds),
                error=str(exc),
                gateway_proc=None,
                agent_proc=None,
            )

    def collect_usage(
        self, task_id: str, output_dir: Path, elapsed_time: float
    ) -> dict:
        cli_runner = self._import_cli_runner()
        summary = self._summaries.get(task_id, {})
        result_path = output_dir / "cli_result.json"
        if not summary and result_path.is_file():
            try:
                loaded = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary = loaded
            except (OSError, json.JSONDecodeError):
                pass

        # ``retry_count`` is part of the shared CLI summary contract; the
        # normalizer preserves it when known and leaves it unknown on timeout.
        return cli_runner.usage_from_cli_summary(summary, elapsed_time)

    def cleanup_staging(self, task_id: str) -> None:
        staging = self._staging_dirs.pop(task_id, None)
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
