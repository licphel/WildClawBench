"""OpenClaw's browser/web_fetch runtime configuration (infrastructure only)."""

import json
import subprocess
from pathlib import Path

import pytest

from src.agents.openclaw import runner as openclaw_runner
from src.agents.openclaw.runner import OpenClawAgent, openclaw_dns_args
from src.utils.docker_utils import container_dns_args


def test_container_dns_args_parses_and_dedupes() -> None:
    env = {"X_DNS": "223.5.5.5, 119.29.29.29 223.5.5.5"}
    assert container_dns_args("X_DNS", env) == [
        "--dns", "223.5.5.5", "--dns", "119.29.29.29",
    ]
    assert container_dns_args("X_DNS", {}) == []


def test_container_dns_args_rejects_hostnames() -> None:
    with pytest.raises(ValueError):
        container_dns_args("X_DNS", {"X_DNS": "dns.example"})


def test_openclaw_dns_prefers_own_setting_then_perdura() -> None:
    assert openclaw_dns_args({"WILDCLAW_PERDURA_DOCKER_DNS": "1.1.1.1"}) == [
        "--dns", "1.1.1.1",
    ]
    assert openclaw_dns_args({
        "WILDCLAW_OPENCLAW_DOCKER_DNS": "8.8.8.8",
        "WILDCLAW_PERDURA_DOCKER_DNS": "1.1.1.1",
    }) == ["--dns", "8.8.8.8"]
    assert openclaw_dns_args({}) == []


def _capture_config_sets(monkeypatch, proxy_env):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(openclaw_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(openclaw_runner, "container_proxy_env", lambda: proxy_env)
    monkeypatch.setattr(openclaw_runner, "openclaw_dns_args", lambda: [])
    return calls


def test_browser_runtime_with_proxy(monkeypatch, tmp_path: Path) -> None:
    calls = _capture_config_sets(monkeypatch, {
        "HTTPS_PROXY": "http://10.0.2.2:17899",
        "NO_PROXY": "localhost,127.0.0.1,::1",
    })
    agent = OpenClawAgent.__new__(OpenClawAgent)
    agent._configure_browser_runtime("task", tmp_path)

    sets = [c[-1] for c in calls if "config set" in c[-1]]
    browser_set = next(c for c in sets if "config set browser " in c)
    assert "noSandbox" in browser_set and "headless" in browser_set
    assert "--proxy-server=http://10.0.2.2:17899" in browser_set
    assert "--proxy-bypass-list=localhost;127.0.0.1;::1" in browser_set
    assert "dangerouslyAllowPrivateNetwork" in browser_set
    assert any("tools.web.fetch.useTrustedEnvProxy true" in c for c in sets)
    assert any("tools.exec.pathPrepend" in c and "/root/miniconda3/envs/eval/bin" in c for c in sets)

    record = json.loads((tmp_path / "browser_runtime.json").read_text())
    assert record["config"]["browser"]["value"]["noSandbox"] is True


def test_browser_runtime_without_proxy(monkeypatch, tmp_path: Path) -> None:
    calls = _capture_config_sets(monkeypatch, {})
    agent = OpenClawAgent.__new__(OpenClawAgent)
    agent._configure_browser_runtime("task", tmp_path)

    sets = [c[-1] for c in calls if "config set" in c[-1]]
    browser_set = next(c for c in sets if "config set browser " in c)
    assert "proxy-server" not in browser_set
    assert not any("useTrustedEnvProxy" in c for c in sets)
