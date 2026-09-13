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


def _channel_entry_to_message(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Map one ``source: "channel"`` entry -- a ``ChannelMessageView.model_dump``
    with ``source`` added -- to the shared openclaw compat message shape.

    This is the original (pre-merged-stream) converter logic, unchanged: each
    view's ``meta.sender_is_user`` decides the openclaw role, and
    ``data.kind`` decides how the content is represented:

    - ``text`` / ``media`` -> a plain ``text`` content block (analogous to
      claudecode/transcript.py's ``_normalize_role_content_message``).
    - ``interaction_request`` -> a ``tool_use`` block (see
      ``_interaction_request_block``).
    - ``interaction_response`` -> a ``tool_result`` block (see
      ``_interaction_response_block``).

    Unknown kinds fall back to a text block holding the raw ``data`` payload
    rather than being silently dropped. Returns ``None`` for a row with
    nothing renderable (e.g. an empty text message), mirroring every other
    converter here.
    """
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
        return None
    return {"type": "message", "message": {"role": role, "content": content}}


def _assistant_entry_to_message(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Map one ``source: "assistant"`` entry (one per ``model.infer``
    completion, see ``_completion_record`` in baselines/PyReduce
    apps/perdura-cli/.../trajectory.py) to a plain assistant text message.

    ``assistant_text`` is already the completion's rendered response text,
    and ``code_blocks`` is a regex-extracted list of fenced code blocks
    already inside that same text -- emitting both would duplicate content,
    so only ``assistant_text`` is used here, exactly as claudecode's
    ``_normalize_role_content_message`` and hermesagent's ``_assistant_entry``
    both represent a plain model text turn: a ``message`` row whose content is
    a single ``text`` block. A completion with no renderable text (a bare
    tool-only turn, or a failed/errored ``model.infer``) is skipped, mirroring
    every other converter here.
    """
    text = entry.get("assistant_text")
    if not text:
        return None
    return {
        "type": "message",
        "message": {"role": "assistant", "content": [{"type": "text", "text": str(text)}]},
    }


def _tool_call_use_block(entry: dict[str, Any]) -> dict[str, Any]:
    """Represent a ``source: "tool_call"`` entry's request half as a
    ``tool_use`` block, analogous to ``_interaction_request_block``: the
    entry's own ``request.name`` / ``request.arguments`` (an
    ``OperationRecord.model_dump`` with ``source`` added -- see
    ``perdura.abi.execution.syscall.OperationRecord``) become the tool name
    and input.
    """
    request = entry.get("request") if isinstance(entry.get("request"), dict) else {}
    return {
        "type": "tool_use",
        "name": str(request.get("name", "")),
        "id": str(entry.get("operation_id", "")),
        "input": request.get("arguments", {}),
    }


def _tool_call_result_content(entry: dict[str, Any]) -> str | None:
    """Render a ``tool_call`` entry's settled outcome as the ``tool_result``
    block's ``content`` string, or ``None`` when the operation never settled
    (``phase`` still ``request_committed``/``executing``, so both ``result``
    and ``error`` are absent -- nothing to report yet).
    """
    outcome = entry.get("outcome")
    if outcome == "succeeded":
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        return json.dumps(result.get("value"), ensure_ascii=False)
    if outcome in ("failed", "timed_out", "cancelled", "outcome_unknown"):
        error = entry.get("error") if isinstance(entry.get("error"), dict) else {}
        return json.dumps(error, ensure_ascii=False)
    return None


def _tool_call_entry_to_messages(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Map one ``source: "tool_call"`` entry -- every non-``model.infer``
    ``OperationRecord`` (sandbox/tool operation) -- to a ``tool_use``/
    ``tool_result`` message pair, symmetric with
    ``_interaction_request_block``/``_interaction_response_block``. Unlike
    the Channel interaction request/response pair (two separate source rows),
    one ``OperationRecord`` carries both its request and its settled outcome,
    so this emits both halves from the single entry: a ``tool_use`` block in
    an assistant message, then -- only once the operation has settled -- a
    matching ``tool_result`` block in a user message, threaded back via
    ``tool_use_id`` = the operation's own ``operation_id``. Reusing the
    tool_use/tool_result shape means every WildClaw ``grade()`` function's
    existing `block.get("type") in ("tool_use", "toolCall")` scan picks these
    up uniformly regardless of which baseline produced them.
    """
    messages = [
        {
            "type": "message",
            "message": {"role": "assistant", "content": [_tool_call_use_block(entry)]},
        }
    ]
    result_content = _tool_call_result_content(entry)
    if result_content is not None:
        messages.append(
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(entry.get("operation_id", "")),
                            "content": result_content,
                        }
                    ],
                },
            }
        )
    return messages


def _task_description_message(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Map the single ``source: "task_description"`` entry (always first) to
    a user message carrying the Task's own definition.

    Perdura never posts the Task's prompt to the operator Channel the way an
    interactive session would -- the definition is CLI-driven, not a chat
    message -- so without this, a non-interactive Task's transcript would
    give a grader no visibility at all into what was asked, unlike every
    other WildClawBench baseline, whose native session transcript naturally
    opens with the user's task prompt (``eval/run_batch.py`` builds that same
    prompt text and hands it to every baseline's CLI/session, this one
    included; only the other baselines' own transcripts echo it back).
    Emitting it here is fairness with those baselines' grading inputs, not an
    addition to what the model was shown: the model already received this
    text as its task prompt before this exporter ever runs.
    """
    name = entry.get("name")
    arguments_excerpt = entry.get("arguments_excerpt")
    parts = [str(part) for part in (name, arguments_excerpt) if part]
    text = "\n\n".join(parts)
    if not text:
        return None
    return {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}


def _task_return_message(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Map the single ``source: "task_return"`` entry (always last, only
    present when the Task produced a result) to an assistant message
    carrying that return value.

    This is the Task's actual final output -- the thing being graded -- and
    nothing else currently carries it into the transcript, so it is always
    worth surfacing (unlike ``task_description``, there is no duplication
    concern: no other path re-states the return value).
    """
    terminal_status = entry.get("terminal_status")
    return_value = entry.get("return_value")
    parts = []
    if terminal_status:
        parts.append(f"[task_return] terminal_status={terminal_status}")
    if return_value is not None:
        parts.append(json.dumps(return_value, ensure_ascii=False))
    text = "\n".join(parts)
    if not text:
        return None
    return {
        "type": "message",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def perdura_channel_messages_to_openclaw_messages(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map perdura's exported merged chat stream to the shared openclaw
    compat message shape every WildClaw grade() function reads.

    ``entries`` is the already-parsed ``canonical/chat.jsonl`` member of a
    `pylm export trajectory` zip: one JSON object per line, oldest-to-newest,
    each tagged with a top-level ``source`` field (see baselines/PyReduce
    apps/perdura-cli/src/perdura/cli/composition/trajectory.py::
    _merged_chat_entries / ADR-0416). ``source`` decides how each entry is
    represented:

    - ``"channel"`` (or missing, for zips exported before this merged-stream
      format existed) -- the original Channel-only conversion; see
      ``_channel_entry_to_message``.
    - ``"assistant"`` -- one ``model.infer`` completion's plain response
      text; see ``_assistant_entry_to_message``.
    - ``"tool_call"`` -- every other (non-``model.infer``) operation; see
      ``_tool_call_entry_to_messages``.
    - ``"task_description"`` -- the Task's own definition, always first; see
      ``_task_description_message``.
    - ``"task_return"`` -- the Task's return value, always last when present;
      see ``_task_return_message``.

    An entry with an unrecognized ``source`` falls back to a text block
    holding its raw JSON payload rather than being silently dropped, matching
    this file's "unknown kind" fallback for Channel entries. A row with
    nothing renderable is skipped, mirroring every other converter here.
    """
    converted: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")

        if source in (None, "channel"):
            message = _channel_entry_to_message(entry)
            if message is not None:
                converted.append(message)
        elif source == "assistant":
            message = _assistant_entry_to_message(entry)
            if message is not None:
                converted.append(message)
        elif source == "tool_call":
            converted.extend(_tool_call_entry_to_messages(entry))
        elif source == "task_description":
            message = _task_description_message(entry)
            if message is not None:
                converted.append(message)
        elif source == "task_return":
            message = _task_return_message(entry)
            if message is not None:
                converted.append(message)
        else:
            converted.append(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": json.dumps(entry, ensure_ascii=False)}],
                    },
                }
            )
    return converted


def convert_pylm_trajectory_zip_to_openclaw_jsonl(zip_path: Path, output_path: Path) -> int:
    """Extract ``canonical/chat.jsonl`` from a pylm trajectory zip and convert it.

    Writes the openclaw compat transcript to ``output_path`` (creating parent
    directories as needed) and returns how many messages it contains.

    A Task whose merged chat stream carries only its ``task_description`` /
    ``task_return`` bookends (no Channel history, no completions, no tool
    calls) still exports a non-empty ``canonical/chat.jsonl`` -- that is the
    normal shape for a non-interactive Task, and is exactly the case this
    converter exists to make visible to a grader (see the trajectory-export
    commit this converter is paired with). A missing/unreadable zip degrades
    the same way any other converter here does on a missing source file:
    write an empty transcript and return 0, rather than raising.
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
