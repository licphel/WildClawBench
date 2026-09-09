from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BENCH_CONFIG_PATH = "/tmp/hermes_bench_config.json"
HERMES_INSTALL_DIR = "/opt/hermes"
HERMES_WORKSPACE = "/tmp_workspace"
HERMES_SESSION_DIR = "/root/.hermes/sessions"


def _sanitize_resume_history(messages: list[dict]) -> list[dict]:
    """Keep only complete OpenAI-compatible tool-call rounds."""
    history: list[dict] = []
    pending_ids: set[str] = set()
    pending_start: int | None = None

    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system", "user", "assistant", "tool",
        }:
            continue

        restored = dict(message)
        role = restored.get("role")
        if role == "assistant":
            for field in (
                "reasoning",
                "reasoning_content",
                "reasoning_details",
                "codex_reasoning_items",
                "finish_reason",
            ):
                restored.pop(field, None)

        if pending_ids:
            tool_call_id = restored.get("tool_call_id")
            if role != "tool" or tool_call_id not in pending_ids:
                return history[:pending_start]
            history.append(restored)
            pending_ids.remove(tool_call_id)
            if not pending_ids:
                pending_start = None
            continue

        if role == "tool":
            continue

        history.append(restored)
        if role != "assistant" or not restored.get("tool_calls"):
            continue

        call_ids = [
            call.get("id")
            for call in restored["tool_calls"]
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        ]
        if len(call_ids) != len(restored["tool_calls"]) or len(set(call_ids)) != len(call_ids):
            history.pop()
            break
        pending_ids = set(call_ids)
        pending_start = len(history) - 1

    if pending_ids and pending_start is not None:
        return history[:pending_start]
    return history


def _load_resume_history(session_id: str) -> list[dict]:
    """Load safe chat history for a same-session retry, if one was persisted."""
    session_path = Path(HERMES_SESSION_DIR) / f"session_{session_id}.json"
    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []

    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        return []

    return _sanitize_resume_history(messages)


def main() -> int:
    sys.path.insert(0, HERMES_INSTALL_DIR)
    os.chdir(HERMES_WORKSPACE)

    from run_agent import AIAgent  # imported after install dir is added to sys.path

    data = json.loads(open(BENCH_CONFIG_PATH, encoding="utf-8").read())
    cfg = data["config"]
    prompt = data["prompt"]
    conversation_history = None
    if os.environ.get("WILDCLAW_HERMES_RESUME"):
        restored_history = _load_resume_history(cfg.get("session_id", ""))
        if restored_history:
            conversation_history = restored_history
            prompt = (
                "Continue the interrupted task in the current workspace. Preserve "
                "and verify completed work, finish every required output, and do "
                "not restart the task or discuss the interruption."
            )
        else:
            prompt = (
                "A previous attempt of this same task was interrupted by a transient "
                "provider error. Continue from the current workspace, preserve and "
                "verify completed work, and finish every required output. Do not "
                "restart the task or discuss the interruption. The original task is:\n\n"
                f"{prompt}"
            )

    agent = AIAgent(
        model=cfg["model"],
        api_key=cfg.get("api_key") or None,
        provider=cfg.get("provider", "custom"),
        base_url=cfg.get("base_url", ""),
        api_mode=cfg.get("api_mode", "codex_responses"),
        max_iterations=cfg.get("max_iterations", 90),
        save_trajectories=True,
        verbose_logging=True,
        reasoning_config=cfg.get("reasoning_config"),
        session_id=cfg.get("session_id"),
    )
    result = agent.run_conversation(prompt, conversation_history=conversation_history)
    print("Completed:", result.get("completed"))
    print("API calls:", result.get("api_calls"))
    return 0 if result.get("completed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
