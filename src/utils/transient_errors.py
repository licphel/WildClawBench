"""Known-transient infra error signatures.

Shared by openclaw's in-container retry (src/agents/openclaw/runner.py) and
eval/run_batch.py's task-level retry, so the two layers can't drift out of
sync on what counts as "worth retrying".
"""

TRANSIENT_ERROR_PATTERNS = (
    "Upstream service temporarily unavailable",
    "Upstream error",  # covers e.g. "HTTP 400: Upstream error: 400" from the relay
    "ECONNRESET",
    "network aborted",
    "ETIMEDOUT",
    "EAI_AGAIN",
    "502 Bad Gateway",
    "503 Service Unavailable",
)


def is_transient_error(error: str | None) -> bool:
    if not error:
        return False
    return any(pattern in error for pattern in TRANSIENT_ERROR_PATTERNS)
