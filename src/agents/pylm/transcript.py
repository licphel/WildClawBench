from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


def _text_from_data(data: dict[str, Any]) -> str:
    """Render a plain chat/media ChannelMessageData block as display text."""
    kind = data.get("kind")
    if kind == "text":
        return str(data.get("text", ""))
    if kind == "media":
        media_type = str(data.get("media_type", "") or "media")
        caption = str(data.get("caption", ""))
        return f"[{media_type}] {caption}".strip()
    return ""


def _interaction_request_block(data: dict[str, Any], message_id: Any) -> dict[str, Any]:
    """Represent a ChannelInteractionRequestData as a tool_use block.

    Perdura's operator Channel treats "ask the operator something and wait"
    (an approval, a form, ...) as its own message kind rather than a plain
    chat message. The openclaw compat schema has no equivalent message kind
    of its own, but every other converter here already represents an agent
    invoking something external as a `tool_use` content block (see
    claudecode/transcript.py, hermesagent/compat_transcript.py) -- an
    interaction request is exactly that: the agent invoking the interaction
    system and (eventually) getting an answer back. Reusing the shape means
    a grade() function's existing `block.get("type") in ("tool_use",
    "toolCall")` scan already picks these up for free.
    """
    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    lifecycle = data.get("lifecycle") if isinstance(data.get("lifecycle"), dict) else {}
    return {
        "type": "tool_use",
        "name": "interaction_request",
        "id": str(message_id or ""),
        "input": {
            "request": request,
            "lifecycle_state": lifecycle.get("state"),
        },
    }


def _interaction_response_block(data: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Represent a ChannelInteractionResponseData as a tool_result block.

    Symmetric with `_interaction_request_block`: the operator's answer to an
    interaction request is the "result" side of that tool call, so it takes
    the matching `tool_result` shape, threaded back to the request via
    `tool_use_id` (from the response message's own `reply_parent`).
    """
    response = data.get("response") if isinstance(data.get("response"), dict) else {}
    reply_parent = meta.get("reply_parent") if isinstance(meta.get("reply_parent"), dict) else {}
    return {
        "type": "tool_result",
        "tool_use_id": str(reply_parent.get("message_id", "")),
        "content": json.dumps(response, ensure_ascii=False),
    }


def perdura_channel_messages_to_openclaw_messages(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map perdura's exported operator-Channel rows to the shared openclaw
    compat message shape every WildClaw grade() function reads.

    ``entries`` is the already-parsed ``canonical/chat.jsonl`` member of a
    `pylm export trajectory` zip: one ``_model_dump(ChannelMessageView)`` dict
    per line, oldest-to-newest (see baselines/PyReduce
    apps/perdura-cli/src/perdura/cli/composition/trajectory.py::
    _channel_messages_oldest_first). Each view's ``meta.sender_is_user``
    decides the openclaw role, and ``data.kind`` decides how the content is
    represented:

    - ``text`` / ``media`` -> a plain ``text`` content block (analogous to
      claudecode/transcript.py's ``_normalize_role_content_message``).
    - ``interaction_request`` -> a ``tool_use`` block (see
      ``_interaction_request_block``).
    - ``interaction_response`` -> a ``tool_result`` block (see
      ``_interaction_response_block``).

    Unknown kinds fall back to a text block holding the raw ``data`` payload
    rather than being silently dropped. A row with nothing renderable (e.g. an
    empty text message) is skipped, mirroring every other converter here.
    """
    converted: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
        kind = data.get("kind")
        role = "user" if meta.get("sender_is_user") else "assistant"

        content: list[dict[str, Any]]
        if kind in ("text", "media"):
            text = _text_from_data(data)
            content = [{"type": "text", "text": text}] if text else []
        elif kind == "interaction_request":
            content = [_interaction_request_block(data, entry.get("message_id"))]
        elif kind == "interaction_response":
            content = [_interaction_response_block(data, meta)]
        elif data:
            content = [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]
        else:
            content = []

        if not content:
            continue
        converted.append({"type": "message", "message": {"role": role, "content": content}})
    return converted


def convert_pylm_trajectory_zip_to_openclaw_jsonl(zip_path: Path, output_path: Path) -> int:
    """Extract ``canonical/chat.jsonl`` from a pylm trajectory zip and convert it.

    Writes the openclaw compat transcript to ``output_path`` (creating parent
    directories as needed) and returns how many messages it contains.

    A Task whose operator Channel has no history exports an empty
    ``canonical/chat.jsonl`` member by design (a real, documented outcome --
    see the trajectory-export commit this converter is paired with -- not a
    failure of this function), and a missing/unreadable zip degrades the same
    way any other converter here does on a missing source file: write an
    empty transcript and return 0, rather than raising.
    """
    entries: list[dict[str, Any]] = []
    raw = ""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                raw = archive.read("canonical/chat.jsonl").decode("utf-8", errors="ignore")
            except KeyError:
                raw = ""
    except (OSError, zipfile.BadZipFile):
        raw = ""

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)

    converted = perdura_channel_messages_to_openclaw_messages(entries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = ""
    if converted:
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in converted) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    return len(converted)
