"""Uniform execution-time spans for every benchmark x harness task record.

One policy, one record shape, every harness.  The headline number
(``execution_seconds``) counts the harness process boot, the agent's own
execution and the harness teardown/close -- and nothing else:

* counted:   ``boot``, ``agent``, ``teardown``
* recorded, never counted: ``env_setup`` (container/workspace preparation,
  task-environment prewarm, memory seeding), ``provider_setup`` (writing the
  model provider configuration), ``export`` (trajectory/artifact export) and
  ``judge`` (grading).

Retries inside the task attempt stay inside the counted spans (a resumed or
re-spawned attempt is still the harness running the task).  A timeout is
recorded with its real elapsed wall time and ``timed_out: true``; the timeout
itself is never substituted for the measurement.

Record contract (``schema_version`` 1)::

    {"schema_version": 1,
     "policy": "harness-boot-to-teardown/v1",
     "execution_seconds": <float>,            # sum of the counted spans present
     "counted_spans": ["boot", "agent", "teardown"],
     "spans": {"<name>": {"seconds": float,
                          "started_at": "<UTC ISO8601>",
                          "ended_at": "<UTC ISO8601>"}, ...},
     "timed_out": <bool>, "timeout_s": <float|null>}

Absent spans are omitted, never written as zero.  A harness whose boot cannot
be separated from its agent run records one ``agent`` span covering both and
sets ``"boot_included_in": "agent"``.

Durations come from ``time.monotonic``; wall-clock time is used only for the
ISO stamps.  Spans are *exclusive*: when a span stops while other spans are
still open (a ``boot`` nested inside an ``agent`` span, for example), its
duration is removed from every enclosing open span, so the spans of one
recorder never overlap and their sum is meaningful.

This file is mirrored verbatim at
``benchmarks/WildClawBench/src/utils/timing_spans.py`` (WildClawBench cannot
import ``eval_framework``); ``test_timing_spans.py`` keeps the two identical.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Any, Callable, Iterator, Mapping

SCHEMA_VERSION = 1
POLICY = "harness-boot-to-teardown/v1"
SPAN_NAMES = (
    "env_setup",
    "provider_setup",
    "boot",
    "agent",
    "teardown",
    "export",
    "judge",
)
COUNTED_SPANS = ("boot", "agent", "teardown")
UNCOUNTED_SPANS = tuple(name for name in SPAN_NAMES if name not in COUNTED_SPANS)
#: Attribute name a backend/agent session uses to expose its recorder.
SESSION_ATTRIBUTE = "timing_spans"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _check_name(name: str) -> None:
    if name not in SPAN_NAMES:
        raise ValueError(f"unknown timing span {name!r}; expected one of {SPAN_NAMES}")


class SpanRecorder:
    """Start/stop named spans and render them as a ``timing`` record.

    Instrumentation must never take a run down, so the lifecycle calls are
    forgiving: starting an already-open span or stopping one that is not open
    is a no-op.  Only an unknown span name raises -- that is a programming
    error, not a runtime condition.  A span started again after it stopped
    accumulates (several turns, several retry attempts): ``seconds`` is the
    total, ``started_at`` the first start and ``ended_at`` the last end.
    """

    def __init__(
        self,
        *,
        timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        # name -> {"seconds": float, "started_at": str, "ended_at": str}
        self._spans: dict[str, dict[str, Any]] = {}
        # name -> [monotonic start, wall start, excluded seconds]
        self._open: dict[str, list[Any]] = {}
        self.timed_out = False
        self.timeout_s: float | None = float(timeout_s) if timeout_s is not None else None
        self.boot_included_in: str | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self, name: str) -> None:
        _check_name(name)
        with self._lock:
            if name in self._open:
                return
            self._open[name] = [self._clock(), self._wall_clock(), 0.0]

    def stop(self, name: str) -> float:
        """Close ``name``; return the seconds it added (0.0 if it was not open)."""

        _check_name(name)
        with self._lock:
            entry = self._open.pop(name, None)
            if entry is None:
                return 0.0
            started_mono, started_wall, excluded = entry
            now = self._clock()
            elapsed = max(0.0, now - started_mono)
            seconds = max(0.0, elapsed - excluded)
            # Exclusive spans: whatever is still open around this one does
            # not also get to count the time this one just accounted for.
            # Only this span's own (exclusive) seconds are handed on -- its
            # children already excluded theirs when they stopped -- and never
            # more than the enclosing span has itself been open.
            for other in self._open.values():
                other[2] += min(seconds, max(0.0, now - other[0]))
            self._accumulate(
                name,
                seconds,
                started_at=iso(started_wall),
                ended_at=iso(self._wall_clock()),
            )
            return seconds

    def stop_all(self) -> None:
        """Close every open span, innermost (most recently started) first."""

        with self._lock:
            for name in reversed(list(self._open)):
                self.stop(name)

    def is_open(self, name: str) -> bool:
        with self._lock:
            return name in self._open

    @contextmanager
    def span(self, name: str) -> Iterator["SpanRecorder"]:
        self.start(name)
        try:
            yield self
        finally:
            self.stop(name)

    # -- externally measured spans --------------------------------------------

    def add(
        self,
        name: str,
        seconds: float,
        *,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> None:
        """Record a span measured elsewhere (another process, a container).

        With only a duration the span is stamped as ending now.  Unlike
        ``stop`` this does not shrink enclosing open spans: the caller knows
        whether the external measurement overlaps them (see ``carve``).
        """

        _check_name(name)
        try:
            value = max(0.0, float(seconds))
        except (TypeError, ValueError):
            return
        end = _parse_iso(ended_at) or self._wall_clock()
        start = _parse_iso(started_at) or (end - timedelta(seconds=value))
        with self._lock:
            self._accumulate(name, value, started_at=iso(start), ended_at=iso(end))

    def carve(self, outer: str, inner: Mapping[str, Any]) -> None:
        """Attribute parts of an already-recorded ``outer`` span to other spans.

        ``inner`` maps span names to span dicts (or plain seconds) that were
        measured *within* ``outer`` by someone else -- for example the
        provider setup a backend did inside the ``start_agent_session`` call
        an adapter timed as ``boot``.  Each one is subtracted from ``outer``
        (never below zero) and recorded under its own name.  An inner span
        named like ``outer`` is already part of it and is skipped.
        """

        _check_name(outer)
        with self._lock:
            for name, value in inner.items():
                if name == outer or name not in SPAN_NAMES:
                    continue
                if isinstance(value, Mapping):
                    seconds = value.get("seconds")
                    started_at = value.get("started_at")
                    ended_at = value.get("ended_at")
                else:
                    seconds, started_at, ended_at = value, None, None
                try:
                    amount = max(0.0, float(seconds))
                except (TypeError, ValueError):
                    continue
                current = self._spans.get(outer)
                if current is not None:
                    taken = min(amount, current["seconds"])
                    current["seconds"] -= taken
                self.add(name, amount, started_at=started_at, ended_at=ended_at)

    def absorb(self, record: Mapping[str, Any] | "SpanRecorder" | None) -> None:
        """Merge another recorder's (or record's) spans and flags into this one."""

        if record is None:
            return
        if isinstance(record, SpanRecorder):
            record = record.to_record()
        if not isinstance(record, Mapping):
            return
        spans = record.get("spans")
        if isinstance(spans, Mapping):
            for name, value in spans.items():
                if name in SPAN_NAMES and isinstance(value, Mapping):
                    self.add(
                        name,
                        value.get("seconds") or 0.0,
                        started_at=value.get("started_at"),
                        ended_at=value.get("ended_at"),
                    )
        with self._lock:
            if record.get("timed_out"):
                self.timed_out = True
            if self.timeout_s is None and record.get("timeout_s") is not None:
                try:
                    self.timeout_s = float(record["timeout_s"])
                except (TypeError, ValueError):
                    pass
            if record.get("boot_included_in") and self.boot_included_in is None:
                self.boot_included_in = str(record["boot_included_in"])

    # -- queries -------------------------------------------------------------

    def seconds(self, name: str) -> float | None:
        with self._lock:
            span = self._spans.get(name)
            return None if span is None else span["seconds"]

    def snapshot(self) -> dict[str, float]:
        """Closed-span seconds by name, for ``spans_since``."""

        with self._lock:
            return {name: span["seconds"] for name, span in self._spans.items()}

    def spans_since(self, snapshot: Mapping[str, float]) -> dict[str, dict[str, Any]]:
        """What each span gained after ``snapshot`` (only spans that grew)."""

        with self._lock:
            grown: dict[str, dict[str, Any]] = {}
            for name, span in self._spans.items():
                gained = span["seconds"] - float(snapshot.get(name, 0.0))
                if gained > 0 or name not in snapshot:
                    grown[name] = {
                        "seconds": max(0.0, gained),
                        "ended_at": span["ended_at"],
                    }
            return grown

    def mark_timeout(self, timed_out: bool = True, *, timeout_s: float | None = None) -> None:
        with self._lock:
            self.timed_out = bool(timed_out) or self.timed_out
            if timeout_s is not None:
                try:
                    self.timeout_s = float(timeout_s)
                except (TypeError, ValueError):
                    pass

    def to_record(self) -> dict[str, Any]:
        """The ``timing`` block.  Open spans are not included until stopped."""

        with self._lock:
            spans = {
                name: {
                    "seconds": round(self._spans[name]["seconds"], 3),
                    "started_at": self._spans[name]["started_at"],
                    "ended_at": self._spans[name]["ended_at"],
                }
                for name in SPAN_NAMES
                if name in self._spans
            }
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "policy": POLICY,
                "execution_seconds": round(
                    sum(spans[name]["seconds"] for name in COUNTED_SPANS if name in spans),
                    3,
                ),
                "counted_spans": list(COUNTED_SPANS),
                "spans": spans,
                "timed_out": bool(self.timed_out),
                "timeout_s": self.timeout_s,
            }
            if self.boot_included_in:
                record["boot_included_in"] = self.boot_included_in
            return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any] | None, **kwargs: Any) -> "SpanRecorder":
        recorder = cls(**kwargs)
        recorder.absorb(record)
        return recorder

    # -- internals -------------------------------------------------------------

    def _accumulate(self, name: str, seconds: float, *, started_at: str, ended_at: str) -> None:
        span = self._spans.get(name)
        if span is None:
            self._spans[name] = {
                "seconds": seconds,
                "started_at": started_at,
                "ended_at": ended_at,
            }
            return
        span["seconds"] += seconds
        if started_at < span["started_at"]:
            span["started_at"] = started_at
        if ended_at > span["ended_at"]:
            span["ended_at"] = ended_at


def session_recorder(session: Any) -> SpanRecorder | None:
    """The recorder a backend attached to an agent session, if any.

    Agent-session wrappers keep the backend session as ``_session``; either
    level may carry it.
    """

    for candidate in (session, getattr(session, "_session", None)):
        recorder = getattr(candidate, SESSION_ATTRIBUTE, None)
        if isinstance(recorder, SpanRecorder):
            return recorder
    return None


def validate_timing_record(record: Any) -> list[str]:
    """Contract violations in a ``timing`` block (empty list when valid)."""

    problems: list[str] = []
    if not isinstance(record, Mapping):
        return ["timing record is not an object"]
    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version is {record.get('schema_version')!r}")
    if record.get("policy") != POLICY:
        problems.append(f"policy is {record.get('policy')!r}")
    if list(record.get("counted_spans") or ()) != list(COUNTED_SPANS):
        problems.append(f"counted_spans is {record.get('counted_spans')!r}")
    spans = record.get("spans")
    if not isinstance(spans, Mapping):
        return problems + ["spans is not an object"]
    for name, span in spans.items():
        if name not in SPAN_NAMES:
            problems.append(f"unknown span {name!r}")
            continue
        if not isinstance(span, Mapping):
            problems.append(f"span {name!r} is not an object")
            continue
        seconds = span.get("seconds")
        if not isinstance(seconds, (int, float)) or seconds < 0:
            problems.append(f"span {name!r} seconds is {seconds!r}")
        for key in ("started_at", "ended_at"):
            if _parse_iso(span.get(key)) is None:
                problems.append(f"span {name!r} {key} is {span.get(key)!r}")
    expected = round(
        sum(
            float(spans[name].get("seconds") or 0.0)
            for name in COUNTED_SPANS
            if isinstance(spans.get(name), Mapping)
        ),
        3,
    )
    actual = record.get("execution_seconds")
    if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 0.0015:
        problems.append(f"execution_seconds {actual!r} != sum of counted spans {expected!r}")
    if not isinstance(record.get("timed_out"), bool):
        problems.append("timed_out is not a bool")
    timeout_s = record.get("timeout_s")
    if timeout_s is not None and not isinstance(timeout_s, (int, float)):
        problems.append(f"timeout_s is {timeout_s!r}")
    included = record.get("boot_included_in")
    if included is not None and included != "agent":
        problems.append(f"boot_included_in is {included!r}")
    return problems
