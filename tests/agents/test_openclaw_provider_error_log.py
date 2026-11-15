"""OpenClaw reports a provider failure in its own log, not on the CLI's stdout.

2026-11-15 combined re-run, Terra OpenClaw R2 task_5/task_10/task_11: the
upstream cut three streams at once.  ``agent.log`` said only "request timed
out" / "Agent couldn't generate a response", the runner raised "failed without
a resumable provider error", and all three were banked as 0.0.  The signature
was in ``gateway.log`` the whole time.
"""

from __future__ import annotations

from src.agents.openclaw.runner import OpenClawAgent

_TRUNCATED = (
    "2026-11-15T06:36:18.903+00:00 [openai-transport] [responses] error "
    "provider=proxy api=openai-responses model=gpt-5.6-terra "
    "name=ResponsesStreamFailure status=undefined code=server_error "
    "type=undefined causeName=undefined causeCode=undefined "
    "message=server_error: upstream_stream_truncated: upstream stream ended early\n"
    "2026-11-15T06:36:18.938+00:00 [agent/embedded] embedded run agent end: "
    "runId=6fd92a10 isError=true model=gpt-5.6-terra provider=proxy "
    "error=proxy/gpt-5.6-terra request failed (request timed out). This is "
    "usually temporary -- try again shortly. rawError=server_error: "
    "upstream_stream_truncated: upstream stream ended early\n"
)
_HANDSHAKE = (
    "2026-11-15T06:40:01.000+00:00 [openai-transport] [responses] error "
    "provider=proxy api=openai-responses model=gpt-5.6-terra name=Error "
    "status=500 code=500 type=upstream_error causeName=undefined "
    "causeCode=undefined message=500 ConnectTimeout: _ssl.c:1015: The "
    "handshake operation timed out\n"
)
_AGENT_STDOUT = (
    "proxy/gpt-5.6-terra request failed (request timed out). This is usually "
    "temporary -- try again shortly.\n"
)


def test_agent_stdout_alone_carries_no_signature(tmp_path):
    agent_log = tmp_path / "agent.log"
    agent_log.write_text(_AGENT_STDOUT)
    assert OpenClawAgent._find_error_marker(agent_log, 0) is None


def test_the_truncation_in_openclaws_own_log_is_found(tmp_path):
    log = tmp_path / "gateway.log"
    log.write_text("2026-11-15T06:34:19.987+00:00 [gateway] ready\n" + _TRUNCATED)
    assert OpenClawAgent._find_provider_error_in_openclaw_log(log, 0)


def test_the_handshake_timeout_in_openclaws_own_log_is_found(tmp_path):
    log = tmp_path / "gateway.log"
    log.write_text(_HANDSHAKE)
    assert OpenClawAgent._find_provider_error_in_openclaw_log(log, 0)


def test_only_this_attempts_part_of_the_log_is_read(tmp_path):
    log = tmp_path / "gateway.log"
    log.write_text(_TRUNCATED)
    offset = log.stat().st_size
    with log.open("a") as handle:
        handle.write("2026-11-15T06:40:00.000+00:00 [gateway] agent run started\n")
    assert OpenClawAgent._find_provider_error_in_openclaw_log(log, offset) is None


def test_a_tool_failure_quoting_a_provider_error_is_not_one(tmp_path):
    """The same log carries what a fetched page said; that is task content."""
    log = tmp_path / "gateway.log"
    log.write_text(
        "2026-11-15T06:35:00.000+00:00 [tools] web_fetch failed: SECURITY "
        "NOTICE: ... HTTP 500 ConnectTimeout upstream_stream_truncated\n"
        "2026-11-15T06:35:01.000+00:00 [provider-transport-fetch] [model-fetch] "
        "response provider=proxy status=200\n"
    )
    assert OpenClawAgent._find_provider_error_in_openclaw_log(log, 0) is None
