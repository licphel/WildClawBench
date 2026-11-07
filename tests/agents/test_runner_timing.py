"""The runners emit the uniform ``timing`` block (src/utils/timing_spans.py).

Setup before the clock is uncounted env_setup; the agent run is counted; a
timeout keeps its real elapsed time and says ``timed_out`` -- the legacy
``elapsed_time`` keeps reporting the budget on that path.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from src.agents.base import AgentTaskSpec
from src.agents.codex import runner as codex_runner
from src.utils.timing_spans import validate_timing_record

_ROOT = Path(__file__).resolve().parents[2]


def _spec(tmp_path: Path, timeout: int) -> AgentTaskSpec:
    return AgentTaskSpec(
        task_id="t1",
        task={},
        workspace_path=str(tmp_path / "ws"),
        prompt="do it",
        timeout_seconds=timeout,
        output_dir=tmp_path / "out",
        model="m",
    )


def _stub_setup(monkeypatch: pytest.MonkeyPatch, agent) -> None:
    def slow_container(*args, **kwargs):
        time.sleep(0.2)  # container start: never counted

    monkeypatch.setattr(agent, "_start_container", slow_container)
    monkeypatch.setattr(agent, "_prepare_workspace", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_write_codex_config", lambda **k: None)
    monkeypatch.setattr(agent, "_should_enable_image_helper", lambda *a: False)
    monkeypatch.setattr(agent, "_install_openclaw_transcript_shim", lambda *a: None)
    for name in ("record_posture", "setup_skills", "run_warmup", "snapshot_workspace_state"):
        monkeypatch.setattr(codex_runner, name, lambda *a, **k: None)
    monkeypatch.setattr(codex_runner, "load_skill_documents", lambda *a, **k: [])


@pytest.mark.parametrize("timed_out", [False, True])
def test_codex_runner_timing_counts_the_run_not_the_setup(tmp_path, monkeypatch, timed_out):
    monkeypatch.setenv("DOCKER_IMAGE_CODEX", "img")
    agent = codex_runner.CodexAgent()
    _stub_setup(monkeypatch, agent)

    def run_prompt(**kwargs):
        time.sleep(0.3)
        if timed_out:
            raise subprocess.TimeoutExpired(cmd="codex", timeout=0)
        return 0.0

    monkeypatch.setattr(agent, "_run_prompt", run_prompt)
    execution = agent.run_task(_spec(tmp_path, timeout=0 if timed_out else 600))

    record = execution.timing
    assert validate_timing_record(record) == []
    spans = record["spans"]
    assert spans["env_setup"]["seconds"] >= 0.2
    assert spans["agent"]["seconds"] >= 0.3
    assert record["boot_included_in"] == "agent"
    assert record["timed_out"] is timed_out
    # Real elapsed (>= 0.3s) even when the budget (0s) was exceeded, and
    # none of the 0.2s container start.
    assert 0.3 <= record["execution_seconds"] < 0.45
    if timed_out:
        assert execution.elapsed_time == 0.0  # legacy: the budget, unchanged


def test_run_batch_records_judge_export_and_gateway_teardown():
    """run_batch.py is import-heavy (judge credentials); pin the wiring by source."""

    source = (_ROOT / "eval" / "run_batch.py").read_text(encoding="utf-8")
    assert "timing.absorb(execution.timing)" in source
    assert 'with timing.span("judge"):\n                result = grade_the_task(' in source
    assert 'with timing.span("export"):\n                usage = backend.collect_usage(' in source
    assert 'with timing.span("teardown"):\n                try:\n                    gateway_proc.terminate()' in source
    assert 'usage["timing"] = timing.to_record()' in source
    assert "_restamp_usage_timing(output_dir, result[\"timing\"])" in source


@pytest.mark.parametrize(
    "runner", ["codex", "claudecode", "hermesagent", "openclaw", "pylm"]
)
def test_every_runner_hands_back_a_timing_record(runner):
    source = (_ROOT / "src" / "agents" / runner / "runner.py").read_text(encoding="utf-8")
    run_task = source.split("    def run_task(", 1)[1].split("\n    def ", 1)[0]
    returns = run_task.count("return AgentExecution(")
    assert returns and run_task.count("timing=") >= returns
