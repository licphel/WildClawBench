"""One token counter for all five WildClaw baselines: the inference Gateway.

Why this exists
---------------
WildClaw's five baselines each self-report token usage and they do not agree on
what the numbers mean.  Measured on real ``usage.json`` records in this repo
(``output*/<backend>/*/*/*/usage.json``, records with ``total_tokens > 0``):

===========  =======  ==========================  ===============================
backend      records  ``total_tokens`` equals     convention
===========  =======  ==========================  ===============================
claudecode         1  input + output + cache_read cache read counted *separately*
openclaw          73  input + output + cache_read cache read counted *separately*
codex             12  input + output              cache read is *inside* input
hermesagent        2  input + output              cache read is *inside* input
pylm              44  input + output              cache read is *inside* input
===========  =======  ==========================  ===============================

A consumer that does not know which convention a record follows silently
double-counts the prompt side, or halves the apparent cache rate.  Worse, the
self-reported numbers have been wrong outright: claudecode reported
``request_count: 1`` for a task where codex reported 13 on the same work.

The fix is not to repair five reporters, it is to stop having five.  WildClaw
now runs through the inference Gateway, and the Gateway sees every request on
the wire.  This module reads the Gateway's own count and folds it into the same
``usage.json`` the harness already writes, alongside -- never instead of -- what
the backend said about itself.

Relationship to ``eval_framework/sentinel_gateway.py``
-----------------------------------------------------
Sentinel solved this problem first; ``fetch_gateway_usage`` /
``collect_gateway_usage`` there are the reference.  The normalized payload
below is deliberately the same shape (same keys, same ``status`` /``source``
strings) so a reader who knows one knows the other.

The *retrieval* differs, and has to.  Sentinel's Gateway runs inside each trial
container, so it reads ``/terrarium/inference_gateway/usage.json`` off that
container's filesystem and gets a counter whose whole lifetime is one trial.
WildClaw's Gateway is a host-side singleton shared by the entire batch (it must
be: pacing has to be coherent across clients, and an OAuth refresh token is
single-use).  There is no per-task file to read and no per-client partition in
the Gateway, so a task's slice is a *delta* of the shared cumulative counter,
read over HTTP from ``GET /v1/gateway/provenance``.

That delta is exact only if nothing else was talking to the Gateway during the
window.  ``GatewayUsageWindow`` therefore tracks how many task windows were
open concurrently and stamps every record with the answer: ``attribution:
"exclusive"`` (trustworthy, promoted to the authoritative top-level numbers) or
``attribution: "overlapped"`` (recorded, but the self-reported numbers stay
authoritative because the delta cannot be attributed to one task) --
*unless* the caller also sent a per-task correlation id (``TASK_ID_HEADER``)
and the Gateway confirmed it understood task-scoping, in which case the
counter this window read was never shared with any other task's window to
begin with and the record says ``attribution: "task_scoped"`` instead: the
concurrent-window count is beside the point when nothing else could have
landed in this task's own bucket.

Elapsed time is NOT consolidated the same way
---------------------------------------------
The gateway cannot supply it.  It sees request latency; ``elapsed_time`` is the
agent's wall clock, which also covers container startup, tool execution and
whatever else the harness put inside its own timer.  So each baseline keeps its
own number -- and the record says which definition produced it, because the
definitions still differ on scope.  Read off the runners:

===========  =============================================================
baseline     the clock
===========  =============================================================
claudecode   opened at ``run_task`` entry, so container start, workspace
             prep, skills and warmup are inside it
codex        same
hermesagent  opened *after* container start, prep, skills, warmup and the
             hermes config write -- agent only
openclaw     opened after all setup and the gateway's 2s readiness sleep --
             agent only
pylm         the whole ``docker exec`` of the container entrypoint, so it
             excludes container start but includes the entrypoint's own
             provider setup and trajectory export
===========  =============================================================

Two axes.  One of them has been unified; the other has not, and must not be
read as if it had.

*Retries: unified, on "included".*  Every baseline retries, but only three of
the five retry somewhere their Python wrapper can see.  openclaw reconnects
inside the openclaw CLI (``MAX_RETRIES = 5`` with 1s/2s/4s/8s/16s backoff, in
``baselines/openclaw/src/agents/openai-ws-connection.ts``) and perdura retries
inside its own runtime -- it reports ``retry_count`` and nothing more -- so
neither wrapper can deduct that time however much it wants to.  "Every clock
includes retry time" is therefore the only convention all five can actually
satisfy; "every clock subtracts it" is not on the menu, and a column where two
bars quietly mean something different from the other three is worse than a
column that is uniformly inclusive.  So the three wrappers that used to
subtract their own retry attempts no longer do:

    claudecode   includes wrapper retry time
    codex        includes
    hermesagent  includes
    openclaw     includes (and never could reach the CLI's own reconnects)
    pylm         includes (and never could reach perdura's own retries)

Each wrapper still *measures* the time its own retry attempts and backoff
cost, because the **task budget** still refunds it: a run resumed after a
transient provider error gets its full timeout of real work.  That refund is a
fairness knob about how long the agent may run; it is not, any more, a
subtraction from the number that gets reported.

*Scope: not unified.*  claudecode and codex time the whole task including
container startup; hermesagent, openclaw and pylm time the agent alone.  This
axis is what ``runtime_semantics`` still discriminates, so nobody plots five
baselines' runtimes against each other without knowing that two of the bars
carry container start, workspace prep, skills and warmup that the other three
do not.

Vocabulary
----------
``usage_source`` and ``cache_semantics`` mirror ``eval_framework/run_layout.py``
exactly -- that module is the source of truth for the vocabulary and this one
cannot import it (WildClawBench is a separate checkout with its own
interpreter).  ``eval_framework/test_wildclaw_gateway_usage.py`` asserts the two
lists stay identical, so a rename in one is a test failure rather than a silent
vocabulary fork.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Vocabulary -- keep identical to eval_framework/run_layout.py
# --------------------------------------------------------------------------- #

USAGE_SOURCE_GATEWAY = "gateway"
USAGE_SOURCE_BACKEND = "backend"
USAGE_SOURCE_MIXED = "mixed"
USAGE_SOURCE_NONE = "none"
USAGE_SOURCE_VALUES = (
    USAGE_SOURCE_GATEWAY,
    USAGE_SOURCE_BACKEND,
    USAGE_SOURCE_MIXED,
    USAGE_SOURCE_NONE,
)

CACHE_DISJOINT = "disjoint"
CACHE_SUBSET = "subset"
CACHE_UNKNOWN = "unknown"
CACHE_SEMANTICS_VALUES = (CACHE_DISJOINT, CACHE_SUBSET, CACHE_UNKNOWN)

#: The Gateway accumulates in the Anthropic key shape -- ``note_usage`` is fed
#: ``_anthropic_usage(...)`` or ``_anthropic_usage_from_responses(...)``, and the
#: latter subtracts the cached bucket out of the Responses prompt total
#: (``input_tokens = total_input - cached_input``) precisely so the two buckets
#: are disjoint.  So a Gateway-observed record is ``disjoint`` for *every*
#: baseline, including the three whose own reporting is ``subset``.  The
#: convention is a property of who counted, not of which agent ran.
GATEWAY_CACHE_SEMANTICS = CACHE_DISJOINT

#: What each baseline's own ``collect_usage`` means by ``cache_read_tokens``.
#: Verified against the record counts in the module docstring.
SELF_REPORTED_CACHE_SEMANTICS = {
    "claudecode": CACHE_DISJOINT,
    "openclaw": CACHE_DISJOINT,
    "codex": CACHE_SUBSET,
    "hermesagent": CACHE_SUBSET,
    "pylm": CACHE_SUBSET,
}

#: How each baseline defines ``elapsed_time``.  Verified against
#: ``src/agents/<backend>/runner.py`` and, for pylm, against
#: ``eval_framework/wildclaw_cli_runner.py::_run_container_cli``.
#: The three ``*_including_retries`` ids are what the runners produce now.
#: The three ``*_minus_wrapper_retries`` ids are retired but NOT deleted:
#: ``usage.json`` records already on disk carry them, and a reader that cannot
#: resolve the id a record was stamped with learns less than one that can.
RUNTIME_TASK_WITH_RETRIES = "task_wall_clock_including_retries"
RUNTIME_AGENT_WITH_RETRIES = "agent_wall_clock_including_retries"
RUNTIME_CONTAINER_CLI_WITH_RETRIES = "container_cli_wall_clock_including_retries"
RUNTIME_TASK_MINUS_RETRIES = "task_wall_clock_minus_wrapper_retries"
RUNTIME_AGENT_MINUS_RETRIES = "agent_wall_clock_minus_wrapper_retries"
RUNTIME_CONTAINER_CLI_MINUS_RETRIES = "container_cli_wall_clock_minus_wrapper_retries"
RUNTIME_UNKNOWN = "unknown"
RUNTIME_SEMANTICS_VALUES = (
    RUNTIME_TASK_WITH_RETRIES,
    RUNTIME_AGENT_WITH_RETRIES,
    RUNTIME_CONTAINER_CLI_WITH_RETRIES,
    RUNTIME_TASK_MINUS_RETRIES,
    RUNTIME_AGENT_MINUS_RETRIES,
    RUNTIME_CONTAINER_CLI_MINUS_RETRIES,
    RUNTIME_UNKNOWN,
)

#: The ids no baseline reports any more.  Kept resolvable for records written
#: before the retry axis was unified; a *new* record carrying one of these is
#: a bug, not history.
RUNTIME_RETIRED_SEMANTICS = frozenset(
    {
        RUNTIME_TASK_MINUS_RETRIES,
        RUNTIME_AGENT_MINUS_RETRIES,
        RUNTIME_CONTAINER_CLI_MINUS_RETRIES,
    }
)

#: ``includes_container_setup``
#:     whether container start, workspace prep, skills and warmup sit inside
#:     the clock.  This is the discriminating field: two baselines open the
#:     clock before the container exists, three after.
#: ``excludes_wrapper_retry_time``
#:     whether the Python wrapper subtracted the time its own retry attempts
#:     and backoff cost.  False for every baseline in current use -- two of
#:     the five could never have subtracted anything (their retries are inside
#:     the agent process), so "included" is the only convention all five can
#:     satisfy.  True only on the retired ids.
#: ``includes_in_container_retry_time``
#:     true for every baseline.  A wrapper times a process; retries inside that
#:     process are inside the number and cannot be removed after the fact.
#: ``retry_sites``
#:     where the retries this number does or does not count actually happen.
#: ``retired``
#:     present and true on the ids no runner produces any more.  They resolve
#:     so that records already on disk stay readable.
RUNTIME_DEFINITIONS = {
    RUNTIME_TASK_WITH_RETRIES: {
        "includes_container_setup": True,
        "excludes_wrapper_retry_time": False,
        "includes_in_container_retry_time": True,
        "retry_sites": ["wrapper", "in-agent"],
        "note": "clock opens at run_task entry, before the container starts, "
                "so container start, workspace prep, skills and warmup are "
                "inside it; nothing is subtracted -- the wrapper's own resumed "
                "attempts and their backoff are inside the number, as are the "
                "agent's internal retries.  The wrapper still measures its own "
                "retry time, but only to refund the task budget",
    },
    RUNTIME_AGENT_WITH_RETRIES: {
        "includes_container_setup": False,
        "excludes_wrapper_retry_time": False,
        "includes_in_container_retry_time": True,
        "retry_sites": ["wrapper", "in-agent"],
        "note": "clock opens after container start, prep, skills and warmup, "
                "just before the agent process, and nothing is subtracted: "
                "the wrapper's own resumed attempts are inside the number, and "
                "so are the agent's internal ones -- for openclaw that is the "
                "CLI reconnecting inside the container (MAX_RETRIES = 5, "
                "1s/2s/4s/8s/16s backoff), which no wrapper ever saw",
    },
    RUNTIME_CONTAINER_CLI_WITH_RETRIES: {
        "includes_container_setup": False,
        "excludes_wrapper_retry_time": False,
        "includes_in_container_retry_time": True,
        "retry_sites": ["wrapper", "in-agent"],
        "note": "the whole docker exec of the container entrypoint: excludes "
                "container start but includes the entrypoint's provider setup "
                "and trajectory export, the attempts _run_container_cli "
                "resumed after a transient provider error, and perdura's own "
                "internal retries (reported as retry_count, never subtractable)",
    },
    RUNTIME_TASK_MINUS_RETRIES: {
        "includes_container_setup": True,
        "excludes_wrapper_retry_time": True,
        "includes_in_container_retry_time": True,
        "retry_sites": ["wrapper", "in-agent"],
        "retired": True,
        "note": "RETIRED.  clock opens at run_task entry, before the container "
                "starts; the wrapper's own retry attempts and backoff are "
                "subtracted, the agent's internal ones are not.  claudecode "
                "and codex reported this until the retry axis was unified on "
                "'included'; kept so those records still resolve",
    },
    RUNTIME_AGENT_MINUS_RETRIES: {
        "includes_container_setup": False,
        "excludes_wrapper_retry_time": True,
        "includes_in_container_retry_time": True,
        "retry_sites": ["wrapper", "in-agent"],
        "retired": True,
        "note": "RETIRED.  clock opens after container start, prep, skills and "
                "warmup; the wrapper's own retry attempts and backoff are "
                "subtracted.  hermesagent and openclaw reported this until the "
                "retry axis was unified on 'included'; kept so those records "
                "still resolve",
    },
    RUNTIME_CONTAINER_CLI_MINUS_RETRIES: {
        "includes_container_setup": False,
        "excludes_wrapper_retry_time": True,
        "includes_in_container_retry_time": True,
        "retry_sites": ["wrapper", "in-agent"],
        "retired": True,
        "note": "RETIRED.  the docker exec of the container entrypoint, minus "
                "the attempts _run_container_cli resumed after a transient "
                "provider error; perdura's own internal retries were still "
                "inside the number (reported as retry_count).  pylm reported "
                "this until the retry axis was unified on 'included'; kept so "
                "those records still resolve",
    },
    RUNTIME_UNKNOWN: {
        "includes_container_setup": None,
        "excludes_wrapper_retry_time": None,
        "includes_in_container_retry_time": None,
        "retry_sites": [],
        "note": "unrecognised baseline; the definition was not read off a runner",
    },
}

RUNTIME_SEMANTICS_BY_BASELINE = {
    "claudecode": RUNTIME_TASK_WITH_RETRIES,
    "codex": RUNTIME_TASK_WITH_RETRIES,
    "hermesagent": RUNTIME_AGENT_WITH_RETRIES,
    "openclaw": RUNTIME_AGENT_WITH_RETRIES,
    "pylm": RUNTIME_CONTAINER_CLI_WITH_RETRIES,
}

#: run_batch.py builds the agent object before it knows it will need a name for
#: it, and HermesAgentAgent is imported lazily, so map by class name rather than
#: by isinstance.
BACKEND_NAME_BY_CLASS = {
    "ClaudeCodeAgent": "claudecode",
    "CodexAgent": "codex",
    "OpenClawAgent": "openclaw",
    "HermesAgentAgent": "hermesagent",
    "PyLMAgent": "pylm",
}

_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "request_count",
)

#: Gateway provenance key -> our key.  The Gateway speaks Anthropic's names.
_GATEWAY_KEY_MAP = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_input_tokens": "cache_read_tokens",
    "cache_creation_input_tokens": "cache_write_tokens",
    "request_count": "request_count",
}

_PROBE_TIMEOUT_S = float(os.environ.get("WILDCLAW_GATEWAY_USAGE_TIMEOUT", "5"))

#: Same header name as ``eval_framework/backends/inference_gateway.py``'s
#: ``TASK_ID_HEADER``.  Not imported from there: WildClawBench is its own
#: checkout with its own interpreter and does not have that module on its
#: path (see the module docstring's "Vocabulary" section for the same
#: constraint on ``usage_source``/``cache_semantics``).  Kept identical by
#: convention.  Sent on the provenance GET so the Gateway can bucket its
#: answer to just this task's requests instead of the whole process's --
#: the fix for concurrent WildClaw workers (``--parallel`` > 1) whose windows
#: used to overlap in wall-clock time and come back unattributable.
TASK_ID_HEADER = "X-PyLM-Task-Id"


def backend_name(backend: Any) -> str:
    """The CLI backend name for an agent object (``codex``, ``pylm``, ...)."""
    raw = backend if isinstance(backend, str) else type(backend).__name__
    # Accepts either an agent object, its class name, or the CLI backend name
    # itself -- run_batch.py has the object, tests and callers have the name.
    raw = BACKEND_NAME_BY_CLASS.get(raw, raw)
    return re.sub(r"[^a-z0-9]", "", raw.lower())


def self_reported_cache_semantics(backend: Any) -> str:
    """Convention a baseline's own numbers follow, or ``"unknown"``.

    Never guesses: an unrecognised backend is ``"unknown"`` rather than a
    default, because a wrong convention is worse than a declared absence.
    """
    return SELF_REPORTED_CACHE_SEMANTICS.get(backend_name(backend), CACHE_UNKNOWN)


def runtime_semantics_for(backend: Any) -> str:
    """Which ``elapsed_time`` definition a baseline uses, or ``"unknown"``.

    Never guesses, for the same reason ``self_reported_cache_semantics`` does
    not: a wrong definition is worse than a declared absence.
    """
    return RUNTIME_SEMANTICS_BY_BASELINE.get(backend_name(backend), RUNTIME_UNKNOWN)


def total_tokens(usage: dict[str, Any] | None, cache_semantics: str) -> int | None:
    """Billable total under the stated convention, or ``None`` if unstated."""
    if not usage:
        return None
    inp = _int(usage.get("input_tokens"))
    out = _int(usage.get("output_tokens"))
    cache = _int(usage.get("cache_read_tokens"))
    if cache_semantics == CACHE_SUBSET:
        return inp + out
    if cache_semantics == CACHE_DISJOINT:
        return inp + out + cache
    return None


def _int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Reaching the Gateway
# --------------------------------------------------------------------------- #


def _candidate_endpoints() -> list[tuple[str, str]]:
    """(provenance URL, token) pairs worth probing, best first.

    Two launch paths reach ``run_batch.py`` and they plumb the Gateway
    differently, so neither is assumed:

    * ``eval_framework/baseline_verifier/wildclawbench/run_*.sh`` exports
      ``GATEWAY_V1``/``GATEWAY_TOKEN`` from ``resolve_wildclaw_gateway``;
    * ``benchmarks/WildClawBench/script/run.sh`` (what the eval_framework
      adapter invokes) exports neither, and only ``OPENROUTER_BASE_URL`` /
      ``OPENROUTER_API_KEY`` point at whatever is serving the run.

    ``OPENROUTER_BASE_URL`` may equally be a third-party relay, which is why
    every candidate is *probed* rather than trusted: only a response carrying
    ``gateway_version`` is this repo's Gateway.
    """
    explicit_url = os.environ.get("WILDCLAW_GATEWAY_USAGE_URL", "").strip()
    explicit_token = os.environ.get("WILDCLAW_GATEWAY_TOKEN", "").strip()
    pairs = [
        (explicit_url, explicit_token or os.environ.get("GATEWAY_TOKEN", "")),
        (os.environ.get("GATEWAY_V1", ""), os.environ.get("GATEWAY_TOKEN", "")),
        (
            os.environ.get("OPENROUTER_BASE_URL", ""),
            os.environ.get("OPENROUTER_API_KEY", ""),
        ),
    ]
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for raw_url, token in pairs:
        url = _provenance_url(raw_url)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append((url, (token or "").strip()))
    return out


def _provenance_url(base: str) -> str:
    base = (base or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/v1/gateway/provenance"):
        return base
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/gateway/provenance"


def _get_json(
    url: str, token: str, timeout: float, *, task_id: str = ""
) -> dict[str, Any] | None:
    request = urllib.request.Request(url, method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("x-api-key", token)
    if task_id:
        request.add_header(TASK_ID_HEADER, task_id)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class _Endpoint:
    """Resolved Gateway provenance endpoint, probed once per process."""

    _lock = threading.Lock()
    _resolved = False
    _url = ""
    _token = ""
    _reason = "not probed"

    @classmethod
    def resolve(cls) -> tuple[str, str, str]:
        with cls._lock:
            if cls._resolved:
                return cls._url, cls._token, cls._reason
            cls._resolved = True
            candidates = _candidate_endpoints()
            if not candidates:
                cls._reason = (
                    "no candidate endpoint: none of WILDCLAW_GATEWAY_USAGE_URL, "
                    "GATEWAY_V1 or OPENROUTER_BASE_URL is set"
                )
                return cls._url, cls._token, cls._reason
            for url, token in candidates:
                payload = _get_json(url, token, _PROBE_TIMEOUT_S)
                if payload is not None and "gateway_version" in payload:
                    cls._url, cls._token = url, token
                    cls._reason = "ok"
                    logger.info("[gateway-usage] counting through %s", url)
                    return cls._url, cls._token, cls._reason
            cls._reason = (
                "no candidate answered GET /v1/gateway/provenance with a "
                f"gateway_version: tried {', '.join(u for u, _ in candidates)}"
            )
            logger.info("[gateway-usage] %s", cls._reason)
            return cls._url, cls._token, cls._reason

    @classmethod
    def reset(cls) -> None:
        """Forget the probe.  Tests only."""
        with cls._lock:
            cls._resolved = False
            cls._url = ""
            cls._token = ""
            cls._reason = "not probed"


def gateway_available() -> bool:
    url, _, _ = _Endpoint.resolve()
    return bool(url)


def fetch_gateway_usage(*, task_id: str = "") -> dict[str, Any] | None:
    """The Gateway's live *cumulative* counter, normalized.

    Same payload shape as ``eval_framework.sentinel_gateway.fetch_gateway_usage``
    so the two halves of the repo report the same thing under the same names.
    ``None`` when no Gateway is in the path, or when it could not be reached --
    a usage record must never invent numbers, and a failed read is reported as
    an absence rather than as zeros.

    Cumulative and batch-wide: this counter covers every request the singleton
    Gateway has served since it started, from every client.  One task's slice is
    a delta between two of these -- see ``GatewayUsageWindow``.

    ``task_id``, when given, asks the Gateway to scope its answer to just that
    correlation id's requests (see ``TASK_ID_HEADER``) instead of the whole
    process's counters.  Whether it actually understood the request rides back
    under the private ``_usage_scope`` key: ``"task"`` means the Gateway
    confirmed the numbers above are genuinely this id's alone; anything else
    (including a gateway that predates ``TASK_ID_HEADER`` and silently ignores
    it) means they are still the whole process's, and must not be read as
    task-exclusive just because a task id was sent.
    """
    url, token, _ = _Endpoint.resolve()
    if not url:
        return None
    payload = _get_json(url, token, _PROBE_TIMEOUT_S, task_id=task_id)
    if payload is None:
        return None
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return None
    snapshot = {
        "schema_version": 1,
        "status": "observed",
        "source": "inference_gateway_upstream",
        "endpoint": url,
        "_usage_scope": str(payload.get("usage_scope") or ""),
    }
    for gateway_key, our_key in _GATEWAY_KEY_MAP.items():
        snapshot[our_key] = _int(raw.get(gateway_key))
    # Timing and anomalies, for adjudicating a wall-clock timeout. Kept under
    # private keys because `_counters_only` -- what reaches usage.json's
    # cumulative_before/after -- filters to `_COUNTERS`, so these ride along
    # inside the process without widening the record's published shape.
    state = payload.get("requests")
    snapshot["_requests"] = dict(state) if isinstance(state, dict) else None
    anomalies = payload.get("anomalies")
    snapshot["_anomaly_count"] = (
        _int(anomalies.get("count")) if isinstance(anomalies, dict) else None
    )
    return snapshot


# --------------------------------------------------------------------------- #
# One task's slice of the shared counter
# --------------------------------------------------------------------------- #


class GatewayUsageWindow:
    """The Gateway counter's movement across one task's agent run.

    Open it immediately before the agent runs and close it immediately after --
    *before* grading.  WildClaw's LLM judge talks to the same Gateway with the
    same token, so a window left open across ``grade_the_task`` would bill the
    judge's tokens to the agent.

    ``concurrent_tasks_max`` is the number of task windows that were open at any
    point during this one.  With ``--parallel 1`` (and with the eval_framework
    adapter, which invokes ``script/run.sh`` once per task) it is 1 and the delta
    is exactly this task's.  With ``--parallel 4`` it is not, and the record says
    so instead of quietly attributing four tasks' tokens to one.
    """

    _registry_lock = threading.Lock()
    _live: set["GatewayUsageWindow"] = set()

    def __init__(self, *, task_id: str = "") -> None:
        # The per-task correlation id this window's caller stamps on its own
        # outbound requests (see ``TASK_ID_HEADER``).  Optional and backward
        # compatible: a caller that does not pass one falls back to the old
        # concurrent-window-counting ``exclusive`` reasoning exactly as
        # before -- run_batch.py's own ``task_id`` (already a stable,
        # per-task-unique identity: it is the container name) is what is
        # passed in practice.
        self.task_id = task_id
        self.before: dict[str, Any] | None = None
        self.after: dict[str, Any] | None = None
        self.concurrent_tasks_max = 1
        self._closed = False

    @classmethod
    def open(cls, *, task_id: str = "") -> "GatewayUsageWindow":
        window = cls(task_id=task_id)
        window.before = fetch_gateway_usage(task_id=task_id)
        with cls._registry_lock:
            cls._live.add(window)
            live = len(cls._live)
            for other in cls._live:
                other.concurrent_tasks_max = max(other.concurrent_tasks_max, live)
        return window

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Snapshot before deregistering: a sibling task opening in between must
        # still count as an overlap of this window.
        if self.before is not None:
            self.after = fetch_gateway_usage(task_id=self.task_id)
        with self._registry_lock:
            self._live.discard(self)

    def __enter__(self) -> "GatewayUsageWindow":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @property
    def task_scoped(self) -> bool:
        """Whether this window's counter is genuinely this task's alone.

        True only when a correlation id was sent *and* the Gateway's answer,
        both before and after, confirmed it understood the request as
        task-scoped (``_usage_scope == "task"``).  A Gateway that predates
        ``TASK_ID_HEADER`` silently ignores the header and answers with the
        whole process's counter under the same key names -- checking the
        confirmation, not just "did we send an id", is what keeps that case
        from being misread as attributable.
        """
        if not self.task_id:
            return False
        before, after = self.before, self.after
        return (
            isinstance(before, dict)
            and before.get("_usage_scope") == "task"
            and isinstance(after, dict)
            and after.get("_usage_scope") == "task"
        )

    @property
    def exclusive(self) -> bool:
        return self.task_scoped or self.concurrent_tasks_max <= 1

    def timeout_evidence(self) -> dict[str, Any]:
        """What the Gateway saw across this task, for a timeout adjudication.

        Consumed by ``src/utils/transient_errors.py::timeout_was_inference_anomaly``
        and written verbatim into ``usage.json`` so the decision can be
        re-derived from the artifacts. Building it here rather than there keeps
        the rule (which patterns and thresholds mean what) in the module every
        layer shares, and the retrieval (which is Gateway-shaped and
        WildClaw-specific) in the module that already owns retrieval.

        Every age is computed against the **Gateway's own clock** (``now`` in
        its ``requests`` block), never against ours: the WildClaw singleton
        happens to run on the same host as the batch, but Sentinel's runs
        inside the trial container and the two clocks need not agree.
        """

        before, after = self.before, self.after
        base: dict[str, Any] = {
            "schema_version": 1,
            "attributable": False,
            "unattributable_reason": None,
            "request_count": None,
            "anomaly_count": None,
            "in_flight_at_close": None,
            "in_flight_age_s": None,
            "idle_tail_s": None,
            "stalled_request_age_s": _stalled_request_age_s(),
        }
        if not before or not after:
            _, _, reason = _Endpoint.resolve()
            base["unattributable_reason"] = (
                f"no gateway counter for this task ({reason})"
            )
            return base
        if not self.exclusive:
            base["unattributable_reason"] = (
                f"{self.concurrent_tasks_max} task windows overlapped, so the "
                "gateway's movement cannot be attributed to this task"
            )
            return base

        state_before = before.get("_requests")
        state_after = after.get("_requests")
        if not isinstance(state_after, dict) or not isinstance(state_before, dict):
            base["unattributable_reason"] = (
                "the gateway does not report per-request timing (it predates "
                "the provenance 'requests' block); only counts are available"
            )
            return base

        delta = self.delta()
        base["request_count"] = None if delta is None else delta["request_count"]

        anomalies_before = before.get("_anomaly_count")
        anomalies_after = after.get("_anomaly_count")
        if isinstance(anomalies_before, int) and isinstance(anomalies_after, int):
            base["anomaly_count"] = max(0, anomalies_after - anomalies_before)

        now = state_after.get("now")
        opened_at = state_before.get("now")
        base["in_flight_at_close"] = _int(state_after.get("in_flight"))

        oldest = state_after.get("oldest_in_flight_started_at")
        if (
            isinstance(now, (int, float))
            and isinstance(oldest, (int, float))
            and isinstance(opened_at, (int, float))
            # Only a request that started inside this task's window belongs to
            # it. A shared singleton gateway can still be holding a stalled
            # request from an *earlier* task whose handler never returned;
            # blaming this task's timeout on that would retry a healthy run.
            and oldest >= opened_at
        ):
            base["in_flight_age_s"] = round(float(now) - float(oldest), 1)

        last = state_after.get("last_completed_at")
        if isinstance(now, (int, float)) and isinstance(last, (int, float)):
            base["idle_tail_s"] = round(float(now) - float(last), 1)

        if base["request_count"] is None:
            base["unattributable_reason"] = (
                "the gateway counter moved backwards (it was restarted "
                "mid-task), so the difference measures nothing"
            )
            return base

        base["attributable"] = True
        return base

    def delta(self) -> dict[str, Any] | None:
        """This window's movement of the Gateway counter, or ``None``.

        ``None`` means no Gateway was in the path (or it became unreachable
        mid-task).  A counter that went *backwards* also returns ``None``: that
        means the Gateway was restarted mid-task, so the difference is not a
        measurement of anything.
        """
        if not self.before or not self.after:
            return None
        out: dict[str, Any] = {}
        for key in _COUNTERS:
            value = _int(self.after.get(key)) - _int(self.before.get(key))
            if value < 0:
                return None
            out[key] = value
        return out


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


def annotate_usage(
    self_reported: dict[str, Any],
    *,
    backend: Any,
    window: GatewayUsageWindow | None,
) -> dict[str, Any]:
    """Return the ``usage.json`` record: labelled, with both counts kept.

    Contract:

    * The top-level counters stay where every existing consumer already looks
      (``print_summary``, ``run_layout.summarize_wildclaw_run``, the
      eval_framework adapter), and carry whichever count is *authoritative*.
    * ``usage_source`` and ``cache_semantics`` at the top level say who counted
      those numbers and how to total them.  They are never absent.
    * ``self_reported`` always holds the backend's own numbers verbatim, even
      when the Gateway's supersede them.  The disagreement between the two has
      been the signal twice now, so it is not thrown away.
    * ``gateway_usage`` always holds the Gateway's side, including the reason it
      is absent when it is.

    ``cost_usd`` is left self-reported in every case: the Gateway serves a
    subscription and prices nothing, so it has no opinion to contribute.

    ``elapsed_time`` likewise stays exactly as the baseline measured it -- the
    gateway sees request latency, not the agent's wall clock -- but
    ``runtime_source``/``runtime_semantics`` record which of the five
    definitions produced it.

    ``gateway_usage.timeout_adjudication`` carries what the Gateway saw about
    *timing* across the same window: enough for
    ``transient_errors.timeout_was_inference_anomaly`` to say whether a
    wall-clock timeout was the agent running long or the upstream stalling.
    It is written on every task so the threshold stays re-calibratable from a
    normal batch's artifacts, not only from its failures.
    """
    self_semantics = self_reported_cache_semantics(backend)
    self_block = dict(self_reported)
    self_block["usage_source"] = USAGE_SOURCE_BACKEND
    self_block["cache_semantics"] = self_semantics

    record = dict(self_reported)
    record["self_reported"] = self_block

    # Timing is deliberately NOT consolidated: the gateway cannot measure it,
    # so it stays the baseline's own number and is always source "backend".
    # Recording which definition produced it is the whole point.
    runtime_semantics = runtime_semantics_for(backend)
    record["runtime_source"] = USAGE_SOURCE_BACKEND
    record["runtime_semantics"] = runtime_semantics
    record["runtime_definition"] = {
        key: (list(value) if isinstance(value, list) else value)
        for key, value in RUNTIME_DEFINITIONS[runtime_semantics].items()
    }
    self_block["runtime_source"] = USAGE_SOURCE_BACKEND
    self_block["runtime_semantics"] = runtime_semantics

    # Stamped on every task, not only on the ones that time out: run_batch.py
    # reads it off the result to decide whether a timeout earns a retry, and
    # keeping it on the healthy runs too is what makes the threshold
    # re-calibratable from the artifacts of a normal batch.
    timeout_evidence = (
        window.timeout_evidence()
        if window is not None
        else {
            "schema_version": 1,
            "attributable": False,
            "unattributable_reason": "no usage window opened",
        }
    )

    delta = window.delta() if window is not None else None
    if delta is None:
        _, _, reason = _Endpoint.resolve()
        record["gateway_usage"] = {
            "schema_version": 1,
            "status": "unavailable",
            "reason": reason if window is not None else "no usage window opened",
            "source": "inference_gateway_upstream",
            "timeout_adjudication": timeout_evidence,
        }
        record["usage_source"] = (
            USAGE_SOURCE_NONE if _reported_nothing(self_reported) else USAGE_SOURCE_BACKEND
        )
        record["cache_semantics"] = self_semantics
        return record

    gateway_block: dict[str, Any] = {
        "schema_version": 1,
        "status": "observed",
        "source": "inference_gateway_upstream",
        "endpoint": (window.before or {}).get("endpoint", ""),
        "usage_source": USAGE_SOURCE_GATEWAY,
        "cache_semantics": GATEWAY_CACHE_SEMANTICS,
        "concurrent_tasks_max": window.concurrent_tasks_max,
        "attribution": (
            "task_scoped" if window.task_scoped
            else ("exclusive" if window.exclusive else "overlapped")
        ),
        **delta,
        "total_tokens": total_tokens(delta, GATEWAY_CACHE_SEMANTICS),
        "cumulative_before": _counters_only(window.before),
        "cumulative_after": _counters_only(window.after),
        "timeout_adjudication": timeout_evidence,
    }

    if not window.exclusive:
        # A shared window is a real measurement of the Gateway, but not of this
        # task.  Keep it visible and keep the self-reported numbers on top.
        gateway_block["authoritative"] = False
        gateway_block["note"] = (
            f"{window.concurrent_tasks_max} task windows overlapped; this delta "
            "covers all of them, so it is not attributable to one task"
        )
    elif delta["request_count"] == 0 and not _reported_nothing(self_reported):
        # The Gateway was reachable but served nothing while this task ran, and
        # the backend says it did work: the agent was not on the Gateway (a
        # direct-relay run with a stale GATEWAY_V1 still in the environment, or
        # a baseline reaching its provider some other way).  Zeroing a real
        # measurement out would be the worst of both worlds.
        gateway_block["authoritative"] = False
        gateway_block["note"] = (
            "gateway served 0 requests during this task while the backend "
            "reported usage; the agent did not run through this gateway"
        )
    else:
        gateway_block["authoritative"] = True

    record["gateway_usage"] = gateway_block

    # The baseline's own collector is the task-level source of truth. The
    # Gateway delta is retained for audit and only fills in when the baseline
    # exported no usage at all.
    if gateway_block["authoritative"] and not _reported_nothing(self_reported):
        gateway_block["native_usage_authoritative"] = True
        gateway_block["authoritative"] = False
        gateway_block["note"] = (
            "native task usage is authoritative; Gateway delta is audit-only"
        )

    if gateway_block["authoritative"]:
        for key in _COUNTERS:
            record[key] = delta[key]
        record["total_tokens"] = gateway_block["total_tokens"]
        record["usage_source"] = USAGE_SOURCE_GATEWAY
        record["cache_semantics"] = GATEWAY_CACHE_SEMANTICS
        record["usage_disagreement"] = {
            key: delta[key] - _int(self_reported.get(key)) for key in _COUNTERS
        }
    else:
        record["usage_source"] = (
            USAGE_SOURCE_NONE if _reported_nothing(self_reported) else USAGE_SOURCE_BACKEND
        )
        record["cache_semantics"] = self_semantics
    return record


def _stalled_request_age_s() -> float | None:
    """The adjudication threshold, recorded alongside the evidence it judges.

    Imported lazily and defensively: this module is also loaded by tests and
    tools that have no ``src`` package on the path, and a missing threshold
    should degrade the record's completeness, never break usage accounting.
    """
    try:
        from src.utils.transient_errors import STALLED_REQUEST_AGE_S
    except Exception:  # noqa: BLE001
        return None
    return float(STALLED_REQUEST_AGE_S)


def _reported_nothing(usage: dict[str, Any]) -> bool:
    return not any(_int(usage.get(key)) for key in _COUNTERS)


def _counters_only(snapshot: dict[str, Any] | None) -> dict[str, int]:
    snapshot = snapshot or {}
    return {key: _int(snapshot.get(key)) for key in _COUNTERS}
