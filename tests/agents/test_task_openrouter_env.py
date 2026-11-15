"""Every baseline's task container gets the real OpenRouter account.

``resolve_judge_env`` / ``ensure_wildclaw_judge_env`` overwrite process
``OPENROUTER_*`` with the LLM-judge credential (the run's own gateway or the
GLM relay) after snapshotting the real account into
``WILDCLAW_TASK_OPENROUTER_*``. Claude Code, Codex, Hermes and OpenClaw used
to read process ``OPENROUTER_*`` when building their own container env, so
their agents saw the judge endpoint instead of openrouter.ai.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.agents.claudecode.runner import ClaudeCodeAgent
from src.agents.codex.runner import CodexAgent
from src.agents.hermesagent.runner import HermesAgentAgent
from src.utils import docker_utils

REAL_KEY = "sk-or-real"
REAL_BASE = "https://openrouter.ai/api/v1"
JUDGE_KEY = "judge-key"
JUDGE_BASE = "http://172.31.9.69:4000/v1"
TASK_ENV = "OPENROUTER_API_KEY\nOPENROUTER_BASE_URL\nJUDGE_MODEL\n"


@pytest.fixture(autouse=True)
def _judge_overwritten_env(monkeypatch):
    monkeypatch.setenv("WILDCLAW_TASK_OPENROUTER_API_KEY", REAL_KEY)
    monkeypatch.setenv("WILDCLAW_TASK_OPENROUTER_BASE_URL", REAL_BASE)
    monkeypatch.setenv("OPENROUTER_API_KEY", JUDGE_KEY)
    monkeypatch.setenv("OPENROUTER_BASE_URL", JUDGE_BASE)
    monkeypatch.setenv("JUDGE_MODEL", "glm-5.3-flash")
    monkeypatch.setenv("HTTP_PROXY_INNER", "http://10.0.2.2:17899")
    monkeypatch.setenv("HTTPS_PROXY_INNER", "http://10.0.2.2:17899")


class _Recorder:
    def __init__(self) -> None:
        self.runs: list[list[str]] = []

    def __call__(self, cmd, *args, **kwargs):
        self.runs.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")

    def docker_run_env(self) -> dict[str, str]:
        run = next(c for c in self.runs if c[:2] == ["docker", "run"])
        env: dict[str, str] = {}
        for flag, value in zip(run, run[1:]):
            if flag == "-e":
                key, _, val = value.partition("=")
                env[key] = val  # docker: the last -e for a key wins
        return env


def test_task_openrouter_env_only_for_declaring_tasks():
    assert docker_utils.task_openrouter_env(TASK_ENV) == {
        "OPENROUTER_API_KEY": REAL_KEY,
        "OPENROUTER_BASE_URL": REAL_BASE,
    }
    assert docker_utils.task_openrouter_env("JUDGE_MODEL\n") == {}
    assert docker_utils.task_openrouter_env("") == {}


def test_claudecode_container_gets_real_openrouter(monkeypatch, tmp_path: Path):
    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    agent = ClaudeCodeAgent(image="img", anthropic_api_key="k", anthropic_base_url="http://gw")
    agent._start_container("t1", str(tmp_path), small_fast_model="m", task_env=TASK_ENV)
    env = rec.docker_run_env()
    assert env["OPENROUTER_API_KEY"] == REAL_KEY
    assert env["OPENROUTER_BASE_URL"] == REAL_BASE

    rec.runs.clear()
    agent._start_container("t2", str(tmp_path), small_fast_model="m", task_env="")
    env = rec.docker_run_env()
    assert "OPENROUTER_API_KEY" not in env
    assert "OPENROUTER_BASE_URL" not in env


def test_codex_container_gets_real_openrouter(monkeypatch, tmp_path: Path):
    (tmp_path / "exec").mkdir()
    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    agent = CodexAgent(image="img", agent_api_key="k", agent_base_url="http://gw/v1")
    agent._start_container("t1", str(tmp_path), {"env": TASK_ENV}, None)
    env = rec.docker_run_env()
    assert env["OPENROUTER_API_KEY"] == REAL_KEY
    assert env["OPENROUTER_BASE_URL"] == REAL_BASE
    assert env["JUDGE_MODEL"] == "glm-5.3-flash"


def test_hermes_container_gets_real_openrouter(monkeypatch, tmp_path: Path):
    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    agent = HermesAgentAgent(image="img", agent_api_key="k", agent_base_url="http://gw/v1")
    agent._start_container("t1", str(tmp_path), api_key="k", base_url="http://gw/v1",
                           extra_env=TASK_ENV)
    env = rec.docker_run_env()
    assert env["OPENROUTER_API_KEY"] == REAL_KEY
    assert env["OPENROUTER_BASE_URL"] == REAL_BASE
