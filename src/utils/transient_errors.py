"""One retry policy for every benchmark in this repo.

The rule, decided by the owner of the numbers: **a task attempt is retried only
on demonstrated evidence of an inference anomaly.**  An agent that simply did
not finish, or that failed on its own merits, is a measurement and must stand.

Why one module
--------------
The five benchmarks reached this repo with five policies -- WildClaw retried
once on a string match, Sentinel retried a turn five times and a delivery five
more, ToolMaze and BEAM's probe path retried nothing at all -- so the same
upstream blip cost a WildClaw task a fresh container, a Sentinel turn a resend,
and a ToolMaze task its whole score.  Numbers produced under three policies are
not comparable, and no amount of care inside any one harness fixes that.  So the
policy lives here, once, and every harness asks the same question of the same
function.

Three layers, and only two of them are this module's business
-------------------------------------------------------------
*Task-level retry* reruns the whole task from scratch: a fresh container, a
fresh clock, a fresh conversation.  It discards a measurement, so it is the
expensive one and the one the owner's rule is about.  ``should_retry_attempt``
below is that decision, and ``MAX_TASK_ATTEMPTS`` bounds it.

*In-agent resume* continues the agent's own session after a provider error:
same conversation, same workspace, and -- critically -- the elapsed time of the
failed attempt is refunded but nothing else is, so a resume cannot buy the
agent extra budget.  That is not a second measurement, it is the same one
continuing, and removing it would convert every upstream blip into a task
failure and bias against whichever baseline happened to hit more blips.  It is
kept, and keyed off ``resumable_provider_error`` so all five baselines resume on
exactly the same signatures.

*Infrastructure retry* -- a grading subprocess, a container start, an apt/npm
warmup, an OAuth refresh, a judge call -- measures nothing about the agent and
is deliberately out of scope.  ``is_transient_error`` has a second caller in
WildClaw's ``src/utils/grading.py`` that retries a grading subprocess; nothing
here may narrow it.

Strings, and the one thing that is not a string
----------------------------------------------
Most predicates here are *pure functions of a string*, because the runners
classify at a point where they have nothing else -- no Gateway handle, no usage
record.  But a string is whatever some client library decided to print, so a
table of them only ever knows the spellings it has already been shown: perdura
dying on ``MidStreamFallbackError`` reached ``should_retry_attempt`` as the
token ``transport_error``, matched nothing, and was banked as a 0.0 on
``20260908T083632Z-smoke9-pylmcli``.

The evidence that is not a string is the Gateway's own record of the attempt.
Every baseline's inference crosses ``inference_gateway.py``, which classifies
an upstream refusal itself and -- decisively -- forgets it again the moment the
upstream serves anything else.  What survives to the end of an attempt's window
is a refusal that was never taken back.  ``standing_upstream_anomaly`` and
``timeout_was_inference_anomaly`` read that record, and both live here rather
than at their call sites: the rule belongs next to the patterns it qualifies,
or the two drift apart, which is the reason this module exists.  The caller
brings the evidence; this module owns the judgement.

Where the copies live
---------------------
ToolMaze, BEAM, Sentinel and vendor_onboarding all reach their tasks through
``eval_framework/runner.py``, so they import this file directly.  WildClaw does
not: ``benchmarks/WildClawBench`` is its own git repository and its
``eval/run_batch.py`` puts only that directory on ``sys.path``, so nothing
under it can import ``eval_framework``.  This file is therefore authoritative
and is copied verbatim to::

    benchmarks/WildClawBench/src/utils/transient_errors.py

by ``script/sync_inference_gateway.py``, with
``eval_framework/test_inference_gateway_sync.py`` failing the moment the two
differ.  Edit this one; never the mirror.  The mirror keeps its old name
because five files inside that repository import it as
``src.utils.transient_errors``.
"""

# Failures a running session can never talk its way out of: the upstream has
# repudiated the conversation state the session is built on, or the account
# has no instant-inference quota left. A runner that is watching a *still
# running* agent may stop it on these, because continuing cannot recover.
UNRECOVERABLE_SESSION_PATTERNS = (
    # The upstream repudiating the encrypted conversation state a session is
    # built on -- surfaced as a 4xx carrying this marker. No amount of
    # continuing the same session recovers it, and the gateway now records it
    # as an anomaly for Sentinel's side of the same fix
    # (inference_gateway.py::note_anomaly).
    "invalid_encrypted_content",
    # The same repudiation in the prose phrasings hermes-agent's runner
    # observed, where the machine-readable marker above never reached the log.
    "encrypted content could not be verified",
    "could not be decrypted or parsed",
    # Instant-inference quota exhaustion, in both phrasings the relay emits:
    # codex logs "Insufficient quota available for instant inference",
    # hermes-agent logs "insufficient quota for instant inference". The quota
    # refills on its own, so the next attempt is worth making.
    "quota available for instant inference",
    "quota for instant inference",
)

# Failures upstream of the agent that a running agent may still recover from
# on its own: the provider, the relay in front of it, or the socket between
# them. They justify a resume once the run has actually exited non-zero, but
# not killing a live run that is retrying them internally.
UPSTREAM_FAILURE_PATTERNS = (
    "Upstream service temporarily unavailable",
    "Upstream error",  # covers e.g. "HTTP 400: Upstream error: 400" from the relay
    "ECONNRESET",
    # The same reset spelled the way Python's socket layer reports it
    # ("[Errno 104] Connection reset by peer"); the Node-style token above
    # never appears in the claudecode/hermes/pylm stacks.
    "Connection reset by peer",
    "network aborted",
    "ETIMEDOUT",
    "EAI_AGAIN",
    # Hermes reports a provider-unavailable turn as a successful HTTP
    # envelope whose payload contains the upstream's concrete failure.  BEAM
    # promotes that payload into ``error_detail``; these markers make the
    # existing shared task-level policy recognize it as a transient inference
    # anomaly.
    "provider_unavailable",
    "ConnectError",
    "UNEXPECTED_EOF_WHILE_READING",
    "HTTP 500",
    "502 Bad Gateway",
    "503 Service Unavailable",
    # The relay refusing to route because every upstream channel in the group
    # is busy or cooling down. Same class as "Upstream service temporarily
    # unavailable" -- observed in claudecode runs, which carried this marker
    # in its own now-removed list.
    "current group has no available channels",
    # perdura's own retry-exhausted summary (see
    # perdura/infra/runtime/execution/step_decision_provider.py). A single
    # slow/hanging upstream call can consume the whole step deadline on its
    # first attempt, leaving perdura's internal RetryConfig (up to 10
    # attempts) no budget for a second try within that step -- so this
    # surfaces as a hard task failure on what is otherwise a transient
    # upstream stall. Retrying the whole task (fresh container, fresh
    # deadline) gives it another shot instead of an unlucky slow response
    # ending the task outright.
    "Model inference failed after provider reliability processing",
    # The provider SDKs' own transport-failure class names.  A client that
    # cannot open a connection to the inference endpoint has measured the
    # inference path, not the agent -- the same statement ECONNRESET above
    # makes, in the spelling the openai/anthropic Python SDKs use.  Matched on
    # the class name rather than its message ("Connection error.") because the
    # message is generic enough to appear in a task's own transcript.
    #
    # Measured live: WildClaw smoke run 20260907T191009Z-smoke-hermesagent,
    # 04_Search_Retrieval/task_7.  hermes-agent exhausted its own three
    # in-process retries against http://172.17.0.1:53021/v1 -- the gateway was
    # not listening -- and exited 1 after 31s having made zero model calls.
    # None of the signatures above matched, so the attempt was kept as a
    # measurement of an agent that never got to run.
    "APIConnectionError",
    "APITimeoutError",
)

PROVIDER_ERROR_PATTERNS = UNRECOVERABLE_SESSION_PATTERNS + UPSTREAM_FAILURE_PATTERNS

# Narrower than UNRECOVERABLE_SESSION_PATTERNS above on purpose: quota
# exhaustion refills on its own ("the next attempt is worth making", per that
# tuple's own comment), so it stays eligible for a fresh Task retry. Only the
# encrypted-session repudiation is truly session-scoped -- resuming the same
# session cannot recover it, and neither can a fresh container reaching the
# same upstream state, so it is excluded from Task-level retry eligibility
# entirely and left to the bounded in-session Resume loop alone. Ported from
# benchmarks/WildClawBench/src/utils/transient_errors.py, where this
# distinction was built and proven before eval_framework had it; keep the two
# in sync (see script/sync_inference_gateway.py's docstring on why they must
# not drift apart again).
SESSION_ONLY_MARKERS = (
    "invalid_encrypted_content",
    "encrypted content could not be verified",
    "could not be decrypted or parsed",
)

# Deterministic request-shape failures: retrying the same prompt/session
# cannot change the request shape, so they must not enter either in-session
# Resume or Task-level retry. A context-length overflow or an invalid
# parameter reaching this file is proof that a fresh attempt will fail
# identically -- retrying spends a container/session for a result that is
# already known.
DETERMINISTIC_REQUEST_MARKERS = (
    "context_length_exceeded",
    "context length exceeded",
    "maximum context length",
    "maximum context window",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "invalid_request_error",
    "unsupported parameter",
    "invalid parameter",
    "extra inputs are not permitted",
)

# A task that ran out of wall-clock. Deliberately matched on the generic
# phrasing rather than any one harness's wording ("pylm run timed out
# after 900 seconds", "Command '[...]' timed out after 1319 seconds", ...)
# so every baseline is judged by the same rule -- a baseline-specific
# prefix here would recreate exactly the per-baseline divergence this
# table exists to prevent.
#
# Matching this is NOT on its own a reason to retry. A wall-clock timeout is
# two different events wearing one error string, and only one of them is worth
# a fresh container:
#
#   * the agent was still working when the clock ran out. That is a
#     measurement -- a slow agent is a real result -- and re-running it just
#     buys a second roll of the dice on a task the agent already failed to
#     finish inside its budget.
#   * one upstream request hung long enough to eat the budget before it could
#     surface as a provider error. That attempt says nothing about the agent,
#     and a fresh container with a fresh clock is the only way to turn it into
#     a measurement.
#
# The two are indistinguishable from the error string, which is why this table
# used to treat both as transient. They are distinguishable from the Gateway,
# which sees every request on the wire -- so the string match below only says
# "this is a wall-clock timeout", and timeout_was_inference_anomaly() decides
# whether it was the second kind. See that function for the measured evidence.
#
# Kept out of PROVIDER_ERROR_PATTERNS: only the task-level retry may act on
# this. See resumable_provider_error() for why an in-runner resume must not.
WALL_CLOCK_PATTERNS = (
    "timed out after",
)

TRANSIENT_ERROR_PATTERNS = PROVIDER_ERROR_PATTERNS + WALL_CLOCK_PATTERNS


# Failures that match a pattern above but must NOT cost a whole-task retry.
#
# The trajectory export runs *after* the task has already finished and been
# graded, so its 30s timeout says nothing about the measurement -- retrying on
# it rebuilds the container and re-runs a task whose score was already valid.
# Observed live: task_1_arxiv_digest was graded at 19:20:53 and then re-run at
# 19:20:55 because `perdura ... export trajectory ... timed out after 30
# seconds` matched the generic "timed out after" rule above. Losing the
# trajectory zip costs one diagnostic artifact; discarding a graded result
# costs the measurement itself, which is the more expensive of the two.
NON_RETRYABLE_MARKERS = (
    "export', 'trajectory'",
    "export trajectory",
)


def non_retryable_phase(error: str | None) -> str | None:
    """The NON_RETRYABLE_MARKERS signature in ``error``, or None.

    Its own predicate rather than ``_matched_pattern(error,
    NON_RETRYABLE_MARKERS)``, which cannot work: ``_matched_pattern`` *applies*
    this veto before it matches anything, so asking it for the veto's own
    markers always answers None.  The veto used to be reachable only from
    inside that function, which meant it guarded the string legs and nothing
    else -- fine while every leg was a string leg, wrong the moment
    ``should_retry_attempt`` grew one that reads the Gateway instead.
    """

    if not error:
        return None
    lowered = error.lower()
    for marker in NON_RETRYABLE_MARKERS:
        if marker.lower() in lowered:
            return marker
    return None


def deterministic_request_error(error: str | None) -> str | None:
    """Return a request-shape failure that must never be retried, or None."""

    if not error:
        return None
    lowered = error.lower()
    # The encrypted-session signature has its own bounded same-session policy
    # (SESSION_ONLY_MARKERS / task_retryable_provider_error), not this one.
    if any(marker in lowered for marker in SESSION_ONLY_MARKERS):
        return None
    for marker in DETERMINISTIC_REQUEST_MARKERS:
        if marker in lowered:
            return marker
    if any(
        marker in lowered
        for marker in ("http 400", "status 400", "status=400", "code 400", "returned 400")
    ):
        return "HTTP 400 request error"
    return None


def _matched_pattern(error: str | None, patterns: tuple[str, ...]) -> str | None:
    if not error:
        return None
    # Case-folded because the same signature reaches this table in whichever
    # case the layer that printed it used: the relay logs "ECONNRESET" while
    # the Python stacks log "econnreset" after lowercasing their own output.
    # The runners each used to lowercase before matching their private lists;
    # doing it here keeps that working for all of them.
    if non_retryable_phase(error) is not None or deterministic_request_error(error) is not None:
        return None
    lowered = error.lower()
    for pattern in patterns:
        if pattern.lower() in lowered:
            return pattern
    return None


def is_transient_error(error: str | None) -> bool:
    return _matched_pattern(error, TRANSIENT_ERROR_PATTERNS) is not None


def wall_clock_timeout(error: str | None) -> str | None:
    """Return the wall-clock signature in ``error``, or None.

    Pure string matching, like every other predicate here, so the in-container
    runners -- which have no Gateway handle at the point they classify -- can
    still ask "is this a timeout?" and get the same answer run_batch.py gets.
    Whether that timeout deserves a retry is a separate question with a
    separate function, because answering it needs evidence a string does not
    carry.
    """
    return _matched_pattern(error, WALL_CLOCK_PATTERNS)


#: How long an inference request must have been open, at the moment the task's
#: clock ran out, before the timeout is blamed on the upstream rather than on
#: the agent.
#:
#: Read off the request-duration distribution in this repo's own Gateway logs
#: (834 completed upstream requests across 96 ``*gateway*.log`` files under
#: ``eval_results/`` and ``benchmarks/WildClawBench/output/``):
#:
#:     p50 7.2s   p90 16.5s   p99 38.2s   p99.9 61.0s   max 69.9s
#:
#: Not one completed request in that corpus reached 90s. The stall this has to
#: catch runs to the Gateway's own upstream timeout (``--timeout 600``, whose
#: internal stall gate trips at 0.9x = 540s), so the band between 70s and 540s
#: is empty of observations in both directions. 120s sits inside it -- about
#: 1.7x the longest healthy request ever recorded, and well under half the
#: smallest possible stall -- and is placed near the healthy edge so a stall is
#: caught early rather than only once the Gateway gives up on it.
STALLED_REQUEST_AGE_S = 120.0


def standing_upstream_anomaly(evidence: dict[str, object] | None) -> str | None:
    """The Gateway's own verdict that the upstream failed this attempt, or None.

    This is the *structural* inference-anomaly signal, and it is the one this
    module should reach for first, because it is the only one that is
    agent-independent.  Every predicate above it matches a string that some
    client library chose, so each new client -- and each new version of an old
    one -- is a new spelling this table does not know.  ``anomaly_count``
    instead comes from the process every baseline's inference crosses:
    ``inference_gateway.py`` classifies the refusal itself
    (``TRANSIENT_UPSTREAM_FAILURE`` and friends, keyed on the upstream's own
    ``type``/``code``/status rather than on prose) and
    ``gateway_usage.py::GatewayUsageWindow.timeout_evidence`` reports the count
    still *standing* at the end of this attempt's window.

    "Standing" is the whole discriminator and it is why this is not just a
    louder version of the string tables.  A provisional refusal is cleared by
    the next completed request (``Gateway.note_upstream_recovered``), so a
    blip the client's own retry absorbed counts zero here.  What is left is a
    refusal the upstream never took back -- the last thing the gateway did for
    this attempt was decline to serve it -- which is an inference anomaly by
    construction and nothing to do with the agent.

    Measured on the artifacts, in both directions:

    * ToolMaze C1_task_001_P0, ``20260908T083632Z-smoke9-pylmcli``.  perdura's
      litellm client died on ``MidStreamFallbackError`` wrapping the upstream's
      ``{"type": "service_unavailable_error", "code": "server_is_overloaded"}``;
      the failure text that reached the harness was the token
      ``transport_error`` and nothing else, matching no pattern in this file.
      The gateway had already recorded the refusal and it was never recovered:
      ``anomaly_count: 1`` sits in that attempt's own usage record.  The
      attempt was banked as a 0.0 anyway; a manual re-run finished with no
      error.
    * Sentinel ``20260908T065004Z-prov-openclaw``.  One in-band 503, then 23
      completed requests and a normal finish.  Recovered, so ``anomaly_count``
      is 0 and the score stands.

    Attributability is required, exactly as it is for a timeout: when task
    windows overlapped, the gateway's movement belongs to no one task and an
    unprovable case is a result, not a re-run.
    """

    if not isinstance(evidence, dict):
        return None
    if not evidence.get("attributable"):
        return None
    count = _as_int(evidence.get("anomaly_count"))
    if not count:
        return None
    return (
        f"the gateway recorded {count} standing upstream anomaly/anomalies "
        "during this attempt -- a refusal no later completed request took "
        "back, so the attempt measured the inference path, not the agent"
    )


def timeout_was_inference_anomaly(
    evidence: dict[str, object] | None,
) -> tuple[bool, str]:
    """Whether a wall-clock timeout was the upstream's fault, with the reason.

    ``evidence`` is what the Gateway saw across the task's own agent run --
    built by ``src/utils/gateway_usage.py::GatewayUsageWindow.timeout_evidence``
    and written into ``usage.json`` under
    ``gateway_usage.timeout_adjudication``, so every decision this function
    makes can be re-derived from the artifacts afterwards.

    The discriminator is **whether a request was open when the clock ran out**,
    not how long the Gateway had been quiet. That distinction is the whole
    point, and it is the opposite of the obvious guess:

    * A quiet Gateway means the agent is running tools and not asking. In
      healthy Sentinel pylm runs that silence reaches 482.2s
      (``20260908T033244Z-10-pylm``, micromail-attachment-name: a 482.2s hole
      between one completion and the next request, both sides healthy). Any
      "the last request was long ago, so the upstream stalled" rule with a
      threshold under ~500s therefore retries healthy slow agents -- which is
      exactly the behaviour being removed here.
    * A request open for minutes is the stall, because no completed request has
      ever taken longer than 70s (see STALLED_REQUEST_AGE_S).

    A further leg is also "the upstream, not the agent": the Gateway recorded
    an anomaly of its own during the run (it classifies upstream stalls and
    repudiated sessions itself -- see ``inference_gateway.py::note_anomaly``).
    Merely serving zero requests is deliberately *not* evidence of an upstream
    failure: the agent, container, or Gateway itself may have failed before a
    request was sent, and task-level retry is reserved for upstream inference
    anomalies.

    Absence of evidence is not an anomaly. When the Gateway could not be
    reached, is too old to report request timing, or the window overlapped
    another task so its counters are not attributable to this one, the answer
    is False: the rule is that only a demonstrated inference anomaly earns a
    retry, so an unprovable case is a result, not a re-run.
    """

    if not evidence:
        return False, "no gateway evidence for this task; not retried"
    if not evidence.get("attributable"):
        return False, str(
            evidence.get("unattributable_reason")
            or "gateway evidence is not attributable to this task"
        )

    standing = standing_upstream_anomaly(evidence)
    if standing:
        return True, standing

    requests = _as_int(evidence.get("request_count"))

    age = evidence.get("in_flight_age_s")
    if isinstance(age, (int, float)) and age >= STALLED_REQUEST_AGE_S:
        return True, (
            f"an inference request was still open {float(age):.0f}s after it "
            f"started when the clock ran out (>= {STALLED_REQUEST_AGE_S:.0f}s)"
        )

    in_flight = _as_int(evidence.get("in_flight_at_close"))
    if not in_flight:
        open_text = "nothing was left open at the deadline"
    elif isinstance(age, (int, float)):
        open_text = (
            f"the one request open at the deadline was only {float(age):.0f}s "
            f"old (< {STALLED_REQUEST_AGE_S:.0f}s)"
        )
    else:
        open_text = (
            f"{in_flight} request(s) were open at the deadline but none of them "
            "started inside this task's window"
        )
    tail = evidence.get("idle_tail_s")
    tail_text = (
        f", last completion {float(tail):.0f}s before it"
        if isinstance(tail, (int, float))
        else ""
    )
    return False, (
        f"the gateway served {requests} requests for this task and {open_text}"
        f"{tail_text}: the agent was still working, so the timeout is the "
        "measurement"
    )


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def resumable_provider_error(output: str | None) -> str | None:
    """Return the provider-failure signature in ``output``, or None.

    The in-runner resume loops key on this instead of on
    TRANSIENT_ERROR_PATTERNS because they refund the failed attempt's elapsed
    time from the task budget. Refunding a wall-clock exhaustion would hand
    the next attempt the full budget again and loop forever, so only the
    task-level retry -- one fresh container, bounded by MAX_TRANSIENT_RETRIES
    -- may act on WALL_CLOCK_PATTERNS.

    Callers must first establish that the attempt actually failed. A marker
    appearing in the output of a *successful* run is the task's own transcript
    quoting it (a task whose mock API returns HTTP 400, say), not a failure.
    """
    return _matched_pattern(output, PROVIDER_ERROR_PATTERNS)


def task_retryable_provider_error(output: str | None) -> str | None:
    """Return an upstream signature eligible for a fresh Task retry, or None.

    Deliberately a different predicate from ``resumable_provider_error``: a
    session-only failure (SESSION_ONLY_MARKERS) may be resumed same-session a
    bounded number of times, but exhausting that loop is not on its own a
    reason to throw away the container and start a new Task -- resuming
    already established that the *upstream* session state was the problem,
    not the inference path in general, and a fresh Task pays the same cost
    (fresh container, fresh clock) for a signature the bounded Resume loop is
    the correct place to keep trying. Keeping the veto here, next to the
    other predicate, makes the separation structural instead of depending on
    every caller to remember it.
    """

    if not output:
        return None
    lowered = output.lower()
    if any(marker in lowered for marker in SESSION_ONLY_MARKERS):
        return None
    return _matched_pattern(output, PROVIDER_ERROR_PATTERNS)


def unrecoverable_session_error(output: str | None) -> str | None:
    """Return the session-fatal signature in ``output``, or None.

    Narrower than resumable_provider_error() and used only where a runner is
    deciding whether to stop an agent that is still running (hermes-agent's
    log poll). A transport failure there is one the agent's own retry loop
    usually absorbs, so killing the run on it would throw away a run that was
    about to recover; the repudiated-session and quota signatures cannot be
    absorbed by anyone.
    """
    return _matched_pattern(output, UNRECOVERABLE_SESSION_PATTERNS)


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #

#: How many times one task may be *run*, counting the first attempt.  Two: the
#: measurement, and -- only on demonstrated evidence that the first attempt
#: measured the inference path rather than the agent -- one replacement for it.
#:
#: The number matters far less than the gate above it, and that is the point.
#: Under a string-only rule a bigger number is a bigger thumb on the scale (an
#: agent that runs long gets rolled again until it scores; measured live,
#: 01_Productivity_Flow/task_1 scored 0.0 on an attempt that used its whole
#: 1200s budget and 0.3705 on the free re-run it was handed).  Under an
#: evidence gate a retry replaces an attempt that measured nothing, so one is
#: enough: a stall that survives a fresh container and a fresh clock is not
#: transient, and a second re-run would only spend more of a shared OAuth quota
#: on a task that is telling us something real.
MAX_TASK_ATTEMPTS = 2

#: A narrower exception to the number above, not a replacement for it.
#: ``resumable_provider_error`` is upstream's own signature -- a repudiated
#: session, exhausted quota, a dropped connection -- and by construction says
#: nothing about the agent; two attempts hitting the same upstream signature
#: back to back is itself evidence the anomaly is still live, not that a third
#: attempt is a bigger thumb on the scale.  Every other retry reason (a plain
#: stall, an agent's own failure) still stops at MAX_TASK_ATTEMPTS -- only a
#: task whose *every* prior attempt failed on a resumable provider signature
#: gets the extra try.
MAX_TASK_ATTEMPTS_RESUMABLE_PROVIDER_ERROR = 3

#: How long a runner waits before resuming an agent whose attempt died on a
#: provider error.  Zero, for all five baselines.
#:
#: Backoff is real work, but it is not this layer's work: the Gateway every
#: baseline sits behind already owns it.  Pacer.note_status
#: (inference_gateway.py) reads the upstream's own Retry-After, falls
#: back to min(2^consecutive, max_backoff), and gates *every* client
#: through one coherent schedule -- which a per-runner sleep cannot do, because
#: five runners sleeping privately cannot agree on when the upstream is ready.
#: A sleep here is therefore a second, uncoordinated backoff stacked on a
#: working one.
#:
#: What the divergence cost: codex waited **600s** after three consecutive
#: retryable failures (CODEX_ENCRYPTED_CONTENT_RETRY_DELAY_SECONDS) while
#: the other four waited 2s, and codex alone had no per-resume wait at all.
#: The 600s branch has never fired in any run recorded on this host -- no
#: runner.resume_delay event exists in any agent.log under
#: benchmarks/WildClawBench/output* or eval_results/ -- so there is no
#: measurement supporting any positive value, and the largest one was
#: measuring nothing at all.  Zero is the value the data supports.
RESUME_BACKOFF_S = 0.0

#: How many times one agent turn may be resumed, same-session, after a
#: resumable_provider_error -- matching WildClaw's own
#: HERMES_RESUME_ATTEMPTS/OPENCLAW_RESUME_ATTEMPTS (both 3), so a task
#: resumed under eval_framework's shared runner and one resumed under
#: WildClaw's eval/run_batch.py get the same number of chances before either
#: escalates to a fresh Task/container.  This bounds the *resume* layer only:
#: it is spent before MAX_TASK_ATTEMPTS_RESUMABLE_PROVIDER_ERROR is ever
#: reached, not instead of it -- exhausting a resume budget is itself the
#: "still failing" signal the task-level retry then acts on.
RESUME_ATTEMPTS = 3


def should_resume_in_session(error: str | None) -> bool:
    """Should this attempt continue the same session instead of ending the turn?

    One call, so the five backends that each drive their own resume loop
    (claude_code, codex, hermes, openclaw, pylm_cli) cannot drift on what
    counts as resumable -- the same trap ``should_retry_attempt`` exists to
    close at the task level. Currently identical to
    ``bool(resumable_provider_error(error))``; kept as its own name so a
    backend's resume loop reads as answering "should I resume?" rather than
    re-deriving that from the lower-level pattern matcher.
    """

    return bool(resumable_provider_error(error))


def should_retry_attempt(
    error: str | None,
    evidence: dict[str, object] | None = None,
) -> tuple[bool, str]:
    """The whole policy, in one call: rerun this task, or keep it?  And why.

    ``error`` is whatever the harness recorded for the failed attempt (an
    exception's text, a runner's error string, a non-zero-exit summary).
    ``evidence`` is what the Gateway saw across the attempt --
    ``GatewayUsageWindow.timeout_evidence()``, written into the attempt's own
    usage record so every decision here can be re-derived from the artifacts.

    Both callers -- WildClaw's ``eval/run_batch.py`` and the shared
    ``eval_framework/runner.py`` that drives Sentinel, ToolMaze, BEAM and
    vendor_onboarding -- go through this function, so the five benchmarks
    cannot answer the same question differently.

    Returns ``(retry, reason)``.  The reason is always populated, including for
    "no", because a *refusal* to retry is the interesting half: it is the line
    that says a failure was kept as a measurement, and an operator reading a
    zero months later needs to see why.
    """

    if not error:
        return False, "the attempt did not fail"

    # The one veto that outranks every piece of evidence below.  It used to be
    # applied only inside ``_matched_pattern``, which meant it guarded the
    # string legs and nothing else; the gateway leg added below would otherwise
    # re-run a task whose score was already valid because its *post-grading*
    # trajectory export timed out.  Stated here so it covers all three legs.
    phase = non_retryable_phase(error)
    if phase is not None:
        return False, (
            f"the failure is in a phase that runs after the task was graded "
            f"({phase!r}), so it says nothing about the measurement; kept"
        )

    # Retrying the same input cannot change its shape -- a context-length
    # overflow or an invalid parameter is proof a fresh attempt fails
    # identically, so it is vetoed before either kind of provider evidence
    # below gets a say.
    deterministic = deterministic_request_error(error)
    if deterministic is not None:
        return False, (
            f"the attempt failed with a deterministic request error ({deterministic!r}); "
            "the same input cannot succeed by retrying"
        )

    # A session-only signature (the upstream repudiating the encrypted
    # conversation state a session is built on) is handled only by the
    # bounded in-session Resume loop; exhausting that loop is not on its own
    # authorization for a fresh Task/container, so it is excluded here even
    # though resumable_provider_error() (the Resume loop's own gate) matches
    # it. See task_retryable_provider_error().
    session_only = _matched_pattern(error, SESSION_ONLY_MARKERS)
    if session_only is not None:
        return False, (
            f"the attempt failed with the session-only signature ({session_only!r}); "
            "this signature is handled only by the bounded in-session Resume "
            "loop and is not a Task/container retry candidate"
        )

    # A string can carry both signatures -- perdura's retry-exhausted summary
    # ends a run that also ran out of clock -- and the provider half is
    # decisive: an upstream that repudiated the session, ran out of quota or
    # dropped the connection is an inference anomaly by construction, and
    # needs no second opinion from the Gateway.
    provider = task_retryable_provider_error(error)
    if provider:
        return True, (
            f"the attempt failed on an upstream signature ({provider!r}), which "
            "is an inference anomaly by construction"
        )

    # The structural leg, and the reason this function no longer opens with an
    # ``is_transient_error`` gate.  That gate made the *string* the entry
    # condition for consulting any evidence at all, so an attempt the Gateway
    # had already recorded as an upstream failure was banked as a measurement
    # whenever the client happened to spell its death in words this file does
    # not know -- which is exactly what happened to ToolMaze C1_task_001_P0 on
    # ``20260908T083632Z-smoke9-pylmcli`` (failure text: ``transport_error``;
    # that attempt's own record: ``anomaly_count: 1``).  Widening the string
    # table would have fixed that one client's spelling and left the next one
    # to be discovered the same way; asking the Gateway fixes all five,
    # because every baseline's inference crosses it.
    #
    # It is deliberately not a *looser* rule than the string legs -- it is a
    # different kind of evidence, and a strictly harder one to produce.  A
    # standing anomaly means the upstream refused and never served this
    # attempt again; a refusal any client absorbed leaves nothing behind.  See
    # ``standing_upstream_anomaly``.
    standing = standing_upstream_anomaly(evidence)
    if standing:
        return True, standing

    if wall_clock_timeout(error):
        return timeout_was_inference_anomaly(evidence)

    return False, (
        "the failure carries no inference-anomaly signature and the gateway "
        "recorded no standing upstream anomaly for this attempt, so the agent "
        "failed on its own merits and the attempt is the measurement"
    )


#: Where a harness leaves ``timeout_evidence()`` in the record it hands back.
#: Three shapes, because three harnesses wrote their result dicts before there
#: was a policy to read them: WildClaw nests the usage record under ``usage``,
#: the shared adapters put the gateway block at the top level, and BEAM's
#: one-shot record keeps it beside the trajectory.  Listed rather than guessed
#: so a fourth shape fails loudly at review instead of silently never retrying.
_EVIDENCE_PATHS: tuple[tuple[str, ...], ...] = (
    ("usage", "gateway_usage", "timeout_adjudication"),
    ("gateway_usage", "timeout_adjudication"),
    ("usage", "timeout_adjudication"),
)


def attempt_evidence(result: object) -> dict[str, object] | None:
    """The Gateway's record of one attempt, dug out of that attempt's result.

    Read off the result rather than passed down, because the window that
    produced it is opened and closed deep inside the run and its findings are
    already written to the usage record -- so the retry decision is made from
    exactly the artifact an operator can re-read afterwards.
    """

    for path in _EVIDENCE_PATHS:
        node: object = result
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, dict):
            return node
    return None


#: Where a harness leaves the failure text.  ``error`` is the shared runner's
#: and WildClaw's; ``error_detail`` is ToolMaze's (its ``status: "error"``
#: records carry the detail there and leave ``error`` unset).
_ERROR_KEYS = ("error", "error_detail")


def attempt_error(result: object) -> str | None:
    """The failure text of one attempt, whichever key its harness used."""

    if not isinstance(result, dict):
        return None
    parts = [str(result[key]) for key in _ERROR_KEYS if result.get(key)]
    return "\n".join(parts) if parts else None
