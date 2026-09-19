from __future__ import annotations

import json
import zipfile
from pathlib import Path

from src.agents.pylm.transcript import (
    convert_pylm_trajectory_zip_to_openclaw_jsonl,
    perdura_channel_messages_to_openclaw_messages,
)


def _channel_view(
    *,
    message_id: str,
    sender_is_user: bool,
    data: dict,
    reply_parent: dict | None = None,
    created_at: float = 1.0,
) -> dict:
    """A trimmed ``source: "channel"`` entry: a ``ChannelMessageView.model_dump``
    with ``source`` added (see baselines/PyReduce apps/perdura-cli/src/perdura/
    cli/composition/trajectory.py::_tagged / _merged_chat_entries, and
    apps/perdura-cli/tests/unit/test_trajectory_export.py::
    test_trajectory_export_chat_jsonl_covers_chat_and_interaction_in_order for
    the underlying view shape). Only the fields this converter reads are
    included.
    """
    return {
        "source": "channel",
        "message_id": message_id,
        "channel": "channel-1",
        "meta": {
            "name": "message",
            "sender_endpoint_id": "endpoint-1",
            "sender_name": "user" if sender_is_user else "task",
            "sender_is_user": sender_is_user,
            "sender_identity": {},
            "step_id": None,
            "reply_parent": reply_parent,
            "reply_summary": None,
        },
        "data": data,
        "attachments": [],
        "task_inbox_deliveries": [],
        "created_at": created_at,
    }


def _task_description_entry(
    *,
    task_id: str = "task-1",
    created_at: float = 0.0,
    name: str = "summarize_report",
    arguments_excerpt: str = 'path="report.pdf"',
) -> dict:
    """A ``source: "task_description"`` entry, always first
    (trajectory.py::_task_description_entry)."""
    return {
        "source": "task_description",
        "task_id": task_id,
        "created_at": created_at,
        "name": name,
        "arguments_excerpt": arguments_excerpt,
    }


def _assistant_entry(
    *,
    operation_id: str = "op-assistant-1",
    task_id: str = "task-1",
    requested_at: float = 2.0,
    assistant_text: str | None = "Here is the summary.",
    code_blocks: list[str] | None = None,
    finish_reason: str = "stop",
) -> dict:
    """A ``source: "assistant"`` entry (one per ``model.infer`` completion;
    trajectory.py::_completion_record)."""
    return {
        "source": "assistant",
        "operation_id": operation_id,
        "step_id": "step-1",
        "task_id": task_id,
        "provider": "openai",
        "model": "gpt-5.5",
        "requested_at": requested_at,
        "request": {"messages": [{"role": "user", "content": "summarize"}]},
        "response": {"content": assistant_text},
        "error": None,
        "assistant_text": assistant_text,
        "code_blocks": code_blocks or [],
        "reasoning": None,
        "finish_reason": finish_reason,
    }


def _tool_call_entry(
    *,
    operation_id: str = "op-tool-1",
    task_id: str = "task-1",
    requested_at: float = 3.0,
    name: str = "file.write",
    arguments: dict | None = None,
    outcome: str | None = "succeeded",
    phase: str = "settled",
    result_value: object = "wrote 128 bytes",
    error: dict | None = None,
) -> dict:
    """A ``source: "tool_call"`` entry -- an ``OperationRecord.model_dump``
    with ``source`` added (perdura.abi.execution.syscall.OperationRecord;
    trajectory.py::_merged_chat_entries excludes only ``model.infer``)."""
    entry: dict = {
        "source": "tool_call",
        "operation_id": operation_id,
        "task_id": task_id,
        "origin": {"kind": "step", "step_id": "step-1"},
        "request": {
            "name": name,
            "tool_name": name,
            "invocation_key": None,
            "arguments": arguments or {"path": "report.txt"},
        },
        "request_message_id": "msg-req-1",
        "deadline_at": None,
        "requested_at": requested_at,
        "phase": phase,
        "outcome": outcome,
        "terminal_message_id": "msg-term-1" if phase == "settled" else None,
        "terminal_at": requested_at + 0.5 if phase == "settled" else None,
        "result": None,
        "error": None,
        "delivery_message_id": None,
        "delivery_confirmed_at": None,
    }
    if outcome == "succeeded":
        entry["result"] = {"value": result_value, "committed_message_ids": []}
    elif outcome is not None:
        entry["error"] = error or {
            "category": "external",
            "code": "sandbox_error",
            "summary": "write failed",
            "agent_sentence": "The write failed.",
            "details": {},
        }
    return entry


def _task_return_entry(
    *,
    task_id: str = "task-1",
    ended_at: float = 5.0,
    return_value: object = "Summary written to report.txt",
    terminal_status: str = "succeeded",
) -> dict:
    """A ``source: "task_return"`` entry, always last when present
    (trajectory.py::_task_return_entry)."""
    return {
        "source": "task_return",
        "task_id": task_id,
        "ended_at": ended_at,
        "return_value": return_value,
        "terminal_status": terminal_status,
    }


def test_plain_chat_messages_round_trip_role_and_text() -> None:
    entries = [
        _channel_view(
            message_id="msg-user",
            sender_is_user=True,
            data={"kind": "text", "text": "Please summarize this."},
            created_at=6.0,
        ),
        _channel_view(
            message_id="msg-assistant",
            sender_is_user=False,
            data={"kind": "text", "text": "Here is the summary."},
            created_at=7.0,
        ),
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 2
    user_row, assistant_row = converted
    assert user_row == {
        "type": "message",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "Please summarize this."}],
        },
    }
    assert assistant_row["message"]["role"] == "assistant"
    assert assistant_row["message"]["content"] == [
        {"type": "text", "text": "Here is the summary."}
    ]


def test_empty_text_message_is_skipped() -> None:
    entries = [_channel_view(message_id="m", sender_is_user=True, data={"kind": "text", "text": ""})]

    assert perdura_channel_messages_to_openclaw_messages(entries) == []


def test_media_message_renders_caption_as_text() -> None:
    entries = [
        _channel_view(
            message_id="m",
            sender_is_user=False,
            data={"kind": "media", "caption": "a chart", "media_type": "image/png"},
        )
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 1
    block = converted[0]["message"]["content"][0]
    assert block["type"] == "text"
    assert "a chart" in block["text"]


def test_channel_entry_without_source_field_is_still_handled() -> None:
    """Back-compat: a zip exported before the merged-stream format existed
    has no ``source`` key on its (bare ChannelMessageView) chat.jsonl rows."""
    entry = _channel_view(message_id="m", sender_is_user=True, data={"kind": "text", "text": "hi"})
    del entry["source"]

    converted = perdura_channel_messages_to_openclaw_messages([entry])

    assert len(converted) == 1
    assert converted[0]["message"]["role"] == "user"


def test_interaction_request_becomes_a_tool_use_block_without_crashing() -> None:
    entries = [
        _channel_view(
            message_id="chat-approval-request",
            sender_is_user=False,
            data={
                "kind": "interaction_request",
                "task_id": "task-1",
                "request": {
                    "intent": "approval",
                    "title": "Approve deploy",
                    "permissions": [
                        {
                            "operation": "process.execute",
                            "selectors": {"command": "deploy.sh"},
                            "reason": "release window is open",
                        }
                    ],
                },
                "lifecycle": {
                    "state": "answered",
                    "delivery_id": "d1",
                    "provider_message_id": "p1",
                    "error": None,
                },
            },
        )
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 1
    message = converted[0]["message"]
    assert message["role"] == "assistant"
    block = message["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "interaction_request"
    assert block["id"] == "chat-approval-request"
    assert block["input"]["lifecycle_state"] == "answered"
    assert block["input"]["request"]["title"] == "Approve deploy"


def test_interaction_response_becomes_a_tool_result_threaded_to_its_request() -> None:
    entries = [
        _channel_view(
            message_id="chat-approval-response",
            sender_is_user=True,
            data={
                "kind": "interaction_response",
                "response": {"intent": "approval", "decision": "approve"},
            },
            reply_parent={
                "message_id": "chat-approval-request",
                "sender_display_name": "task",
                "preview": "Approve deploy",
            },
        )
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 1
    message = converted[0]["message"]
    assert message["role"] == "user"
    block = message["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "chat-approval-request"
    assert json.loads(block["content"]) == {"intent": "approval", "decision": "approve"}


def test_full_chat_and_interaction_sequence_preserves_order() -> None:
    """Mirrors the PyReduce trajectory-export unit test's own fixture ordering:
    user text, assistant reply, an answered approval request/response pair,
    then an awaiting form request -- oldest to newest."""
    entries = [
        _channel_view(message_id="msg-user-text", sender_is_user=True, data={"kind": "text", "text": "Please summarize this."}, created_at=6.0),
        _channel_view(message_id="msg-assistant-reply", sender_is_user=False, data={"kind": "text", "text": "Here is the summary."}, created_at=7.0),
        _channel_view(
            message_id="chat-approval-request",
            sender_is_user=False,
            data={
                "kind": "interaction_request",
                "task_id": "task-1",
                "request": {"intent": "approval", "title": "Approve deploy"},
                "lifecycle": {"state": "answered", "delivery_id": "", "provider_message_id": "", "error": None},
            },
            created_at=8.0,
        ),
        _channel_view(
            message_id="chat-approval-response",
            sender_is_user=True,
            data={"kind": "interaction_response", "response": {"intent": "approval", "decision": "approve"}},
            reply_parent={"message_id": "chat-approval-request", "sender_display_name": "task", "preview": "Approve deploy"},
            created_at=9.0,
        ),
        _channel_view(
            message_id="chat-form-request",
            sender_is_user=False,
            data={
                "kind": "interaction_request",
                "task_id": "task-1",
                "request": {"intent": "collect_input", "title": "Release notes"},
                "lifecycle": {"state": "awaiting_response", "delivery_id": "", "provider_message_id": "", "error": None},
            },
            created_at=10.0,
        ),
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 5
    roles = [row["message"]["role"] for row in converted]
    assert roles == ["user", "assistant", "assistant", "user", "assistant"]
    kinds = [row["message"]["content"][0]["type"] for row in converted]
    assert kinds == ["text", "text", "tool_use", "tool_result", "tool_use"]
    assert converted[4]["message"]["content"][0]["input"]["lifecycle_state"] == "awaiting_response"


def test_assistant_entry_becomes_a_plain_assistant_text_message() -> None:
    entries = [_assistant_entry(assistant_text="The answer is 42.")]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert converted == [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "The answer is 42."}],
            },
        }
    ]


def test_assistant_entry_with_no_text_is_skipped() -> None:
    entries = [_assistant_entry(assistant_text=None)]

    assert perdura_channel_messages_to_openclaw_messages(entries) == []


def test_assistant_entry_does_not_double_emit_code_blocks() -> None:
    """``code_blocks`` is already inside ``assistant_text`` (regex-extracted);
    only ``assistant_text`` should be rendered."""
    text = "Here you go:\n```python\nprint('hi')\n```\n"
    entries = [_assistant_entry(assistant_text=text, code_blocks=["print('hi')\n"])]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 1
    assert converted[0]["message"]["content"] == [{"type": "text", "text": text}]


def test_succeeded_tool_call_becomes_a_tool_use_and_tool_result_pair() -> None:
    entries = [
        _tool_call_entry(
            operation_id="op-write-1",
            name="file.write",
            arguments={"path": "report.txt", "content": "hello"},
            outcome="succeeded",
            result_value="wrote 5 bytes",
        )
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 2
    use_message, result_message = converted
    assert use_message["message"]["role"] == "assistant"
    use_block = use_message["message"]["content"][0]
    assert use_block == {
        "type": "tool_use",
        "name": "file.write",
        "id": "op-write-1",
        "input": {"path": "report.txt", "content": "hello"},
    }
    assert result_message["message"]["role"] == "user"
    result_block = result_message["message"]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "op-write-1"
    assert json.loads(result_block["content"]) == "wrote 5 bytes"


SECRET = "sk-ant-9rfiwe-q3wef9fiwe-kfj39f8wefnlKJ29fjwfiw"


def _assistant_tool_inputs(converted: list[dict]) -> str:
    chunks: list[str] = []
    for entry in converted:
        msg = entry.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            chunks.append(json.dumps(block.get("input"), ensure_ascii=False, sort_keys=True))
    return "\n".join(chunks)


def test_tool_result_syscall_does_not_put_outcome_in_tool_use_input() -> None:
    """Kernel ``tool.result`` delivers the previous tool's return in
    ``arguments.outcome``. That must not be copied into assistant tool_use
    input, or leaked-API graders treat git-diff stdout as disclosure."""
    entries = [
        _tool_call_entry(
            operation_id="op-tool-result-1",
            name="tool.result",
            arguments={
                "outcome": {
                    "kind": "returned",
                    "value": {"exit_code": 0, "stdout": f"---STAT---\n{SECRET}\n"},
                }
            },
            result_value={
                "exit_code": 0,
                "stdout": f"---STAT---\n{SECRET}\n",
            },
        )
    ]
    converted = perdura_channel_messages_to_openclaw_messages(entries)
    use_block = converted[0]["message"]["content"][0]
    assert use_block["type"] == "tool_use"
    assert use_block["name"] == "tool.result"
    assert SECRET not in json.dumps(use_block["input"])
    assert use_block["input"] == {}
    result_block = converted[1]["message"]["content"][0]
    assert result_block["type"] == "tool_result"
    assert SECRET in result_block["content"]
    assert SECRET not in _assistant_tool_inputs(converted)


def test_observation_show_does_not_put_preview_text_in_tool_use_input() -> None:
    entries = [
        _tool_call_entry(
            operation_id="op-obs-1",
            name="observation.show",
            arguments={"preview": f"...{SECRET[:12]}...", "text": f"diff {SECRET}"},
            result_value={"preview": f"diff {SECRET}", "type": "value"},
        )
    ]
    converted = perdura_channel_messages_to_openclaw_messages(entries)
    use_block = converted[0]["message"]["content"][0]
    assert use_block["input"] == {}
    assert SECRET not in _assistant_tool_inputs(converted)
    assert "text" not in use_block["input"]
    assert "preview" not in use_block["input"]
    assert SECRET in converted[1]["message"]["content"][0]["content"]


def test_shell_invocation_keeps_command_arguments_on_tool_use() -> None:
    """A real agent-authored exec still exposes its command to graders."""
    entries = [
        _tool_call_entry(
            operation_id="op-shell-1",
            name="shell.invocation",
            arguments={"identity": {"command": ["git", "diff"]}},
            result_value={"stdout": f"{SECRET}\n"},
        )
    ]
    converted = perdura_channel_messages_to_openclaw_messages(entries)
    use_block = converted[0]["message"]["content"][0]
    assert use_block["input"]["identity"]["command"] == ["git", "diff"]
    assert SECRET not in _assistant_tool_inputs(converted)
    assert SECRET in converted[1]["message"]["content"][0]["content"]


def test_failed_tool_call_surfaces_the_error_as_the_tool_result() -> None:
    entries = [
        _tool_call_entry(
            operation_id="op-write-2",
            name="file.write",
            outcome="failed",
            error={
                "category": "external",
                "code": "disk_full",
                "summary": "no space left on device",
                "agent_sentence": "The disk is full.",
                "details": {},
            },
        )
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 2
    result_block = converted[1]["message"]["content"][0]
    assert result_block["type"] == "tool_result"
    payload = json.loads(result_block["content"])
    assert payload["code"] == "disk_full"
    assert payload["agent_sentence"] == "The disk is full."


def test_unsettled_tool_call_emits_only_the_tool_use_block() -> None:
    entries = [
        _tool_call_entry(
            operation_id="op-write-3",
            name="file.write",
            outcome=None,
            phase="executing",
        )
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 1
    assert converted[0]["message"]["content"][0]["type"] == "tool_use"


def test_task_description_entry_becomes_a_leading_user_message() -> None:
    entries = [_task_description_entry(name="summarize_report", arguments_excerpt='path="report.pdf"')]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 1
    message = converted[0]["message"]
    assert message["role"] == "user"
    text = message["content"][0]["text"]
    assert "summarize_report" in text
    assert "report.pdf" in text


def test_task_return_entry_becomes_a_trailing_assistant_message() -> None:
    entries = [
        _task_return_entry(
            return_value="Summary written to report.txt",
            terminal_status="succeeded",
        )
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 1
    message = converted[0]["message"]
    assert message["role"] == "assistant"
    text = message["content"][0]["text"]
    assert "succeeded" in text
    assert "Summary written to report.txt" in text


def test_task_return_entry_with_no_ended_at_or_status_is_skipped_when_empty() -> None:
    entries = [
        {
            "source": "task_return",
            "task_id": "task-1",
            "ended_at": None,
            "return_value": None,
            "terminal_status": None,
        }
    ]

    assert perdura_channel_messages_to_openclaw_messages(entries) == []


def test_full_merged_stream_preserves_task_description_first_and_task_return_last() -> None:
    """Mirrors trajectory.py::_merged_chat_entries: task_description always
    first (structural, not by timestamp), then channel/assistant/tool_call
    merged and sorted by (requested_at or created_at, tie-break id), then
    task_return always last."""
    entries = [
        _task_description_entry(created_at=0.0, name="summarize_report"),
        _channel_view(message_id="msg-user", sender_is_user=True, data={"kind": "text", "text": "go"}, created_at=1.0),
        _assistant_entry(operation_id="op-a1", requested_at=2.0, assistant_text="Working on it."),
        _tool_call_entry(operation_id="op-t1", requested_at=3.0, name="file.write", outcome="succeeded", result_value="ok"),
        _task_return_entry(ended_at=5.0, return_value="done", terminal_status="succeeded"),
    ]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    # task_description -> 1 message, channel -> 1, assistant -> 1,
    # tool_call -> 2 (tool_use + tool_result), task_return -> 1 = 6 total.
    assert len(converted) == 6
    assert converted[0]["message"]["role"] == "user"
    assert "summarize_report" in converted[0]["message"]["content"][0]["text"]
    assert converted[-1]["message"]["role"] == "assistant"
    assert "done" in converted[-1]["message"]["content"][0]["text"]
    tool_use_block = converted[3]["message"]["content"][0]
    assert tool_use_block["type"] == "tool_use"
    assert tool_use_block["name"] == "file.write"


def test_unrecognized_source_falls_back_to_a_raw_text_block() -> None:
    entries = [{"source": "future_thing", "payload": {"x": 1}}]

    converted = perdura_channel_messages_to_openclaw_messages(entries)

    assert len(converted) == 1
    block = converted[0]["message"]["content"][0]
    assert block["type"] == "text"
    assert json.loads(block["text"]) == entries[0]


def test_convert_zip_with_no_chat_history_writes_an_empty_transcript(tmp_path: Path) -> None:
    zip_path = tmp_path / "pylm-trajectory.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("canonical/chat.jsonl", "")
        archive.writestr("manifest.json", "{}")
    output_path = tmp_path / "chat.jsonl"

    count = convert_pylm_trajectory_zip_to_openclaw_jsonl(zip_path, output_path)

    assert count == 0
    assert output_path.read_text(encoding="utf-8") == ""


def test_convert_zip_extracts_and_converts_real_chat_history(tmp_path: Path) -> None:
    zip_path = tmp_path / "pylm-trajectory.zip"
    rows = [
        _channel_view(message_id="m1", sender_is_user=True, data={"kind": "text", "text": "hello"}),
        _channel_view(message_id="m2", sender_is_user=False, data={"kind": "text", "text": "hi there"}),
    ]
    chat_jsonl = "\n".join(json.dumps(row) for row in rows) + "\n"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("canonical/chat.jsonl", chat_jsonl)
    output_path = tmp_path / "nested" / "chat.jsonl"

    count = convert_pylm_trajectory_zip_to_openclaw_jsonl(zip_path, output_path)

    assert count == 2
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["message"]["role"] == "user"
    assert written[1]["message"]["role"] == "assistant"


def test_convert_zip_extracts_and_converts_a_non_interactive_task(tmp_path: Path) -> None:
    """A non-interactive Task has no operator Channel history at all -- the
    exact failure mode this feature exists to fix (06_Safety_Alignment
    task_9/task_10 previously scored 0.0 because the judge only ever saw a
    bare file path). The merged stream still carries task_description,
    assistant and tool_call entries, so the compat transcript is non-empty."""
    zip_path = tmp_path / "pylm-trajectory.zip"
    rows = [
        _task_description_entry(name="fix_config", arguments_excerpt='path="config.yaml"'),
        _assistant_entry(operation_id="op-a1", requested_at=1.0, assistant_text="Reading config."),
        _tool_call_entry(operation_id="op-t1", requested_at=2.0, name="file.read", outcome="succeeded", result_value="key: value"),
        _task_return_entry(ended_at=3.0, return_value="Config fixed.", terminal_status="succeeded"),
    ]
    chat_jsonl = "\n".join(json.dumps(row) for row in rows) + "\n"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("canonical/chat.jsonl", chat_jsonl)
    output_path = tmp_path / "chat.jsonl"

    count = convert_pylm_trajectory_zip_to_openclaw_jsonl(zip_path, output_path)

    assert count == 5
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    roles_and_types = [
        (row["message"]["role"], row["message"]["content"][0]["type"]) for row in written
    ]
    assert roles_and_types == [
        ("user", "text"),
        ("assistant", "text"),
        ("assistant", "tool_use"),
        ("user", "tool_result"),
        ("assistant", "text"),
    ]


def test_convert_missing_zip_degrades_to_empty_transcript_without_raising(tmp_path: Path) -> None:
    output_path = tmp_path / "chat.jsonl"

    count = convert_pylm_trajectory_zip_to_openclaw_jsonl(tmp_path / "does-not-exist.zip", output_path)

    assert count == 0
    assert output_path.read_text(encoding="utf-8") == ""
