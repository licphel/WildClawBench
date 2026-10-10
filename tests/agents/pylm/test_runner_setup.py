from __future__ import annotations

import time
import types
from pathlib import Path

import pytest

from src.agents.base import AgentTaskSpec
from src.agents.pylm.runner import PyLMAgent
from src.utils import docker_utils

MARKER = "wildclaw_setup_incomplete"
BUDGET = 0.05


def _spec(tmp_path: Path) -> AgentTaskSpec:
    return AgentTaskSpec(
        task_id="t1",
        task={},
        workspace_path=str(tmp_path / "ws"),
        prompt="p",
        timeout_seconds=60,
        output_dir=tmp_path / "out",
        model="m",
    )


def _agent(monkeypatch, *, prepare_seconds: float, warmup_seconds: float):
    seen: dict = {}

    def prepare(image):
        time.sleep(prepare_seconds)
        return "shared-venv"

    def start(_start_container, task_id, exec_path, **kwargs):
        seen["shared_venv"] = kwargs.get("shared_venv")

    def run_cli(**kwargs):
        seen["agent_started"] = True
        return {}, 1.0, None

    cli_runner = types.SimpleNamespace(
        WILDCLAW_HOST_SETUP_MAX_SECONDS=BUDGET,
        WILDCLAW_SETUP_INCOMPLETE=MARKER,
        prepare_pylm_task_runtime=prepare,
        _start_container_with_live_mounts=start,
        _run_container_cli=run_cli,
    )
    monkeypatch.setattr(PyLMAgent, "_import_cli_runner", classmethod(lambda cls: cli_runner))
    monkeypatch.setattr(docker_utils, "setup_workspace", lambda task_id: None)
    monkeypatch.setattr(docker_utils, "setup_skills", lambda *a, **k: None)
    monkeypatch.setattr(
        docker_utils, "run_warmup", lambda *a, **k: time.sleep(warmup_seconds)
    )
    return PyLMAgent(image="img:v1"), seen


def test_shared_runtime_build_is_not_charged_to_the_task(tmp_path, monkeypatch) -> None:
    agent, seen = _agent(monkeypatch, prepare_seconds=BUDGET * 3, warmup_seconds=0)

    execution = agent.run_task(_spec(tmp_path))

    assert execution.error is None
    assert seen == {"shared_venv": "shared-venv", "agent_started": True}


def test_setup_overrun_is_marked_as_an_unstarted_agent(tmp_path, monkeypatch) -> None:
    agent, seen = _agent(monkeypatch, prepare_seconds=0, warmup_seconds=BUDGET * 3)

    execution = agent.run_task(_spec(tmp_path))

    assert MARKER in execution.error
    assert "agent_started" not in seen


def test_setup_failure_is_marked_as_an_unstarted_agent(tmp_path, monkeypatch) -> None:
    agent, seen = _agent(monkeypatch, prepare_seconds=0, warmup_seconds=0)

    def boom(*a, **k):
        raise RuntimeError("docker exec failed")

    monkeypatch.setattr(docker_utils, "setup_workspace", boom)

    execution = agent.run_task(_spec(tmp_path))

    assert MARKER in execution.error and "docker exec failed" in execution.error
    assert "agent_started" not in seen
