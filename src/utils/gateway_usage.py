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
authoritative because the delta cannot be attributed to one task).

Elapsed time is NOT consolidated the same way
---------------------------------------------
The gateway cannot supply it.  It sees request latency; ``elapsed_time`` is the
agent's wall clock, which also covers container startup, tool execution and
whatever else the harness put inside its own timer.  So each baseline keeps its
own number -- and the record says which definition produced it, because the
five definitions are not the same measurement.  Read off the runners:

===========  =============================================================
baseline     the clock
===========  =============================================================
claudecode   opened at ``run_task`` entry, so container start, workspace
             prep, skills and warmup are inside it; the wrapper's own
             retry attempts and their backoff are subtracted
codex        same
hermesagent  opened *after* container start, prep, skills, warmup and the
             hermes config write -- agent only; wrapper retry subtracted
openclaw     opened after all setup and the gateway's 2s readiness sleep --
             agent only; nothing subtracted
pylm         the whole ``docker exec`` of the container entrypoint, so it
             excludes container start but includes the entrypoint's own
             provider setup and trajectory export; nothing subtracted
===========  =============================================================

Two axes, and they do not line up:

*Scope.*  claudecode and codex time the whole task including container
startup; hermesagent, openclaw and pylm time the agent alone.

*Retries.*  This is the one that bites.  Three baselines subtract the time
their wrapper spent on retry attempts and backoff; **two do not, and their
retries are real.**  openclaw's runner has no retry code at all (verified: no
``retry``/``attempt``/``resume``/``transient``/loop construct in the file) --
but the openclaw CLI reconnects inside the container, ``MAX_RETRIES = 5`` with
1s/2s/4s/8s/16s backoff (``baselines/openclaw/src/agents/openai-ws-connection.ts``),
and the Python wrapper cannot subtract what it never saw.  pylm is the same
story: ``_run_container_cli`` is a bare ``perf_counter`` span around one
``docker exec`` and perdura retries inside it (it reports ``retry_count`` and
subtracts nothing).

    claudecode   excludes wrapper retry time
    codex        excludes
    hermesagent  excludes
    openclaw     INCLUDES -- retries happen in-container, invisible to the wrapper
    pylm         INCLUDES -- perdura's internal retries, never subtracted

So when the upstream is flaky, openclaw's and pylm's seconds are inflated by
roughly the amount the other three deduct.  Note also that *no* baseline can
subtract retries that happen inside the agent process -- a wrapper only times
the process -- so ``includes_in_container_retry_time`` is true for all five and
the discriminating field is ``excludes_wrapper_retry_time``.

``runtime_semantics`` names which definition produced the number, so nobody
plots five baselines' runtimes against each other without knowing that two of
the bars include something the other three deduct.  Unifying the formulas is
explicitly out of scope here; recording them is not.

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
RUNTIME_TASK_MINUS_RETRIES = "task_wall_clock_minus_wrapper_retries"
RUNTIME_AGENT_MINUS_RETRIES = "agent_wall_clock_minus_wrapper_retries"
RUNTIME_AGENT_WITH_RETRIES = "agent_wall_clock_including_retries"
RUNTIME_CONTAINER_CLI_WITH_RETRIES = "container_cli_wall_clock_including_retries"
RUNTIME_UNKNOWN = "unknown"
RUNTIME_SEMANTICS_VALUES = (
    RUNTIME_TASK_MINUS_RETRIES,
    RUNTIME_AGENT_MINUS_RETRIES,
    RUNTIME_AGENT_WITH_RETRIES,
    RUNTIME_CONTAINER_CLI_WITH_RETRIES,
    RUNTIME_UNKNOWN,
)

#: ``includes_container_setup``
#:     whether container start, workspace prep, skills and warmup sit inside
#:     the clock.
#: ``excludes_wrapper_retry_time``
#:     whether the Python wrapper subtracted the time its own retry attempts
#:     and backoff cost.  This is the discriminating field: three baselines do,
#:     two do not, and the two that do not still retry.
#: ``includes_in_container_retry_time``
#:     true for every baseline.  A wrapper times a process; retries inside that
#:     process are inside the number and cannot be removed after the fact.
#: ``retry_sites``
#:     where the retries this number does or does not count actually happen.
RUNTIME_DEFINITIONS = {
    RUNTIME_TASK_MINUS_RETRIES: {
        "includes_container_setup": True,
        "excludes_wrapper_retry_time": True,
        "includes_in_container_retry_time": True,
        "retry_sites": ["wrapper", "in-agent"],
        "note": "clock opens at run_task entry, before the container starts; "
                "the wrapper's own retry attempts and backoff are subtracted, "
                "the agent's internal ones are not",
    },
    RUNTIME_AGENT_MINUS_RETRIES: {
        "includes_container_setup": False,
        "excludes_wrapper_retry_time": True,
        "includes_in_container_retry_time": True,
        "retry_sites": ["wrapper", "in-agent"],
        "note": "clock opens after container start, prep, skills and warmup; "
                "the wrapper's own retry attempts and backoff are subtracted",
    },
    RUNTIME_AGENT_WITH_RETRIES: {
        "includes_container_setup": False,
        "excludes_wrapper_retry_time": False,
        "includes_in_container_retry_time": True,
        "retry_sites": ["in-agent"],
        "note": "clock opens just before the agent process and nothing is "
                "subtracted: the runner has no retry code, but the openclaw "
                "CLI reconnects inside the container (MAX_RETRIES = 5, "
                "1s/2s/4s/8s/16s backoff) and the wrapper never sees it",
    },
    RUNTIME_CONTAINER_CLI_WITH_RETRIES: {
        "includes_container_setup": False,
        "excludes_wrapper_retry_time": False,
        "includes_in_container_retry_time": True,
        "retry_sites": ["in-agent"],
        "note": "the whole docker exec of the container entrypoint: excludes "
                "container start but includes the entrypoint's provider setup "
                "and trajectory export, and includes perdura's internal "
                "retries (reported as retry_count, never subtracted)",
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
    "claudecode": RUNTIME_TASK_MINUS_RETRIES,
    "codex": RUNTIME_TASK_MINUS_RETRIES,
    "hermesagent": RUNTIME_AGENT_MINUS_RETRIES,
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


def _get_json(url: str, token: str, timeout: float) -> dict[str, Any] | None:
    request = urllib.request.Request(url, method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("x-api-key", token)
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


def fetch_gateway_usage() -> dict[str, Any] | None:
    """The Gateway's live *cumulative* counter, normalized.

    Same payload shape as ``eval_framework.sentinel_gateway.fetch_gateway_usage``
    so the two halves of the repo report the same thing under the same names.
    ``None`` when no Gateway is in the path, or when it could not be reached --
    a usage record must never invent numbers, and a failed read is reported as
    an absence rather than as zeros.

    Cumulative and batch-wide: this counter covers every request the singleton
    Gateway has served since it started, from every client.  One task's slice is
    a delta between two of these -- see ``GatewayUsageWindow``.
    """
    url, token, _ = _Endpoint.resolve()
    if not url:
        return None
    payload = _get_json(url, token, _PROBE_TIMEOUT_S)
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
    }
    for gateway_key, our_key in _GATEWAY_KEY_MAP.items():
        snapshot[our_key] = _int(raw.get(gateway_key))
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

    def __init__(self) -> None:
        self.before: dict[str, Any] | None = None
        self.after: dict[str, Any] | None = None
        self.concurrent_tasks_max = 1
        self._closed = False

    @classmethod
    def open(cls) -> "GatewayUsageWindow":
        window = cls()
        window.before = fetch_gateway_usage()
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
            self.after = fetch_gateway_usage()
        with self._registry_lock:
            self._live.discard(self)

    def __enter__(self) -> "GatewayUsageWindow":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @property
    def exclusive(self) -> bool:
        return self.concurrent_tasks_max <= 1

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

    delta = window.delta() if window is not None else None
    if delta is None:
        _, _, reason = _Endpoint.resolve()
        record["gateway_usage"] = {
            "schema_version": 1,
            "status": "unavailable",
            "reason": reason if window is not None else "no usage window opened",
            "source": "inference_gateway_upstream",
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
        "attribution": "exclusive" if window.exclusive else "overlapped",
        **delta,
        "total_tokens": total_tokens(delta, GATEWAY_CACHE_SEMANTICS),
        "cumulative_before": _counters_only(window.before),
        "cumulative_after": _counters_only(window.after),
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


def _reported_nothing(usage: dict[str, Any]) -> bool:
    return not any(_int(usage.get(key)) for key in _COUNTERS)


def _counters_only(snapshot: dict[str, Any] | None) -> dict[str, int]:
    snapshot = snapshot or {}
    return {key: _int(snapshot.get(key)) for key in _COUNTERS}
