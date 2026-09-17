"""Regression tests for the Hermes WildClaw runner's retry classification."""

from pathlib import Path

from src.agents.hermesagent.runner import HermesAgentAgent


def test_successful_runner_does_not_scan_task_tool_429(tmp_path: Path) -> None:
    """A task API's 429 must not turn a clean Hermes completion into a retry."""
    log_path = tmp_path / "agent.log"
    log_path.write_text(
        'tool output: {"status": 429, "message": "Too Many Requests"}\n',
        encoding="utf-8",
    )

    # The string matcher necessarily sees the marker; it has no knowledge of
    # whether the line came from the model provider or a task-owned tool.
    assert HermesAgentAgent._find_error_marker(log_path, 0) == "too many requests"
    # A zero exit is bench_runner's completed=True contract, so the runner
    # must not use that ambiguous transcript text as a provider failure.
    assert HermesAgentAgent._find_post_exit_provider_error(
        log_path, 0, 0,
    ) is None


def test_failed_runner_still_scans_for_provider_error(tmp_path: Path) -> None:
    """A non-zero runner exit retains provider-resume detection."""
    log_path = tmp_path / "agent.log"
    log_path.write_text(
        "provider request failed: HTTP 429 Too Many Requests\n",
        encoding="utf-8",
    )

    assert HermesAgentAgent._find_post_exit_provider_error(
        log_path, 0, 1,
    ) == "too many requests"
