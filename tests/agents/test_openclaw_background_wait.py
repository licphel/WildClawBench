"""OpenClaw: grading waits for detached media tasks and their completion turn."""

import json
import time
from pathlib import Path

from src.agents.openclaw import runner as openclaw_runner
from src.agents.openclaw.runner import OpenClawAgent

NOW_MS = 1_800_000_000_000


def _probe(tasks, events=10, role="assistant", stop="stop", ts=NOW_MS):
    return {"tasks": tasks, "events": events, "last_role": role,
            "last_stop": stop, "last_ts": ts}


def _task(status, ended_at=None):
    return {"task_id": "t", "runtime": "cli", "task_kind": "image_generation",
            "status": status, "ended_at": ended_at}


def test_state_none_without_tasks() -> None:
    assert OpenClawAgent._background_state(_probe([]))[0] == "none"


def test_ledger_noise_is_not_background_work() -> None:
    noise = [
        {"runtime": "cli", "task_kind": None, "status": "running"},  # the CLI run
        {"runtime": "cron", "task_kind": "automation_run", "status": "failed",
         "ended_at": NOW_MS},
    ]
    assert OpenClawAgent._background_state(_probe(noise))[0] == "none"


def test_state_active_while_task_running() -> None:
    assert OpenClawAgent._background_state(_probe([_task("running")]))[0] == "active"
    assert OpenClawAgent._background_state(
        _probe([_task("succeeded", NOW_MS), _task("queued")])
    )[0] == "active"


def test_state_awaits_turn_until_final_assistant_after_task_end() -> None:
    done = [_task("succeeded", NOW_MS)]
    # Last message predates the task's end: the completion turn has not run.
    assert OpenClawAgent._background_state(_probe(done, ts=NOW_MS - 1))[0] == "awaiting_turn"
    # Completion turn still mid tool call.
    assert OpenClawAgent._background_state(
        _probe(done, stop="toolUse", ts=NOW_MS + 5)
    )[0] == "awaiting_turn"
    assert OpenClawAgent._background_state(_probe(done, ts=NOW_MS + 5)) == ("settled", NOW_MS)


def _run_wait(monkeypatch, tmp_path, probes, budget=600.0):
    clock = {"t": 0.0}
    seq = iter(probes)
    last = {"p": None}

    def fake_probe(self, task_id):
        try:
            last["p"] = next(seq)
        except StopIteration:
            pass
        return last["p"]

    monkeypatch.setattr(OpenClawAgent, "_probe_background_work", fake_probe)
    monkeypatch.setattr(openclaw_runner.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(openclaw_runner.time, "time", lambda: NOW_MS / 1000 + clock["t"])
    monkeypatch.setattr(
        openclaw_runner.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s),
    )
    agent = OpenClawAgent.__new__(OpenClawAgent)
    agent._wait_for_background_work("task", tmp_path, budget_seconds=budget)
    return json.loads((tmp_path / "background_tasks.json").read_text()), clock["t"]


def test_no_background_tasks_returns_immediately(monkeypatch, tmp_path: Path) -> None:
    record, waited = _run_wait(monkeypatch, tmp_path, [_probe([])])
    assert record["outcome"] == "no_background_tasks"
    assert waited == 0.0


def test_waits_through_task_and_completion_turn(monkeypatch, tmp_path: Path) -> None:
    probes = [
        _probe([_task("running")], events=10, ts=NOW_MS - 10_000),
        _probe([_task("running")], events=10, ts=NOW_MS - 10_000),
        _probe([_task("succeeded", NOW_MS)], events=10, ts=NOW_MS - 10_000),
        _probe([_task("succeeded", NOW_MS)], events=14, stop="toolUse", ts=NOW_MS + 3_000),
        _probe([_task("succeeded", NOW_MS)], events=16, ts=NOW_MS + 8_000),
    ]
    record, waited = _run_wait(monkeypatch, tmp_path, probes)
    assert record["outcome"] == "settled"
    # Waited for the quiet window after the final assistant message.
    assert waited >= openclaw_runner.OPENCLAW_BACKGROUND_QUIET_SECONDS


def test_budget_bounds_the_wait(monkeypatch, tmp_path: Path) -> None:
    record, waited = _run_wait(
        monkeypatch, tmp_path, [_probe([_task("running")])], budget=20.0,
    )
    assert record["outcome"] == "budget_exhausted_while_active"
    assert waited <= 20.0


def test_gives_up_when_no_completion_turn(monkeypatch, tmp_path: Path) -> None:
    record, waited = _run_wait(
        monkeypatch, tmp_path,
        [_probe([_task("succeeded", NOW_MS)], ts=NOW_MS - 1)], budget=3600.0,
    )
    assert record["outcome"] == "no_completion_turn"
    assert waited < 3600.0
