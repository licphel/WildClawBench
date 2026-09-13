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
    """A trimmed ChannelMessageView shape, matching what
    `_model_dump(ChannelMessageView)` actually serializes (see
    baselines/PyReduce apps/perdura-cli/tests/unit/test_trajectory_export.py
    ::test_trajectory_export_chat_jsonl_covers_chat_and_interaction_in_order).
    Only the fields this converter reads are included.
    """
    return {
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


def test_convert_missing_zip_degrades_to_empty_transcript_without_raising(tmp_path: Path) -> None:
    output_path = tmp_path / "chat.jsonl"

    count = convert_pylm_trajectory_zip_to_openclaw_jsonl(tmp_path / "does-not-exist.zip", output_path)

    assert count == 0
    assert output_path.read_text(encoding="utf-8") == ""
