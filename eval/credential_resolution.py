"""Credential resolution for WildClawBench's ``eval/run_batch.py``.

Background: every ``run_*.sh`` launcher in the OUTER repo (a sibling
checkout, see ``_OUTER_REPO_ROOT`` below) --
``eval_framework/baseline_verifier/wildclawbench/run_{pylm,codex,
claudecode,hermesagent,openclaw}.sh`` -- resolves this run's per-experiment
inference gateway (``resolve_wildclaw_gateway``) and an independent LLM-judge
credential (``resolve_judge_env``, both in that directory's
``config_lib.sh``) *before* ever invoking ``python3 eval/run_batch.py``.
Someone bypassed that and called ``eval/run_batch.py`` directly, which
skipped all of it: ``JUDGE_MODEL``/``OPENROUTER_API_KEY``/
``OPENROUTER_BASE_URL`` were empty, so every LLM-judge grading call and every
gateway usage probe ran with empty credentials, and a whole benchmark
category silently collapsed to a score of 0.0 instead of failing loudly.

This module exists so that cannot happen again regardless of how
``run_batch.py`` is invoked: it is imported unconditionally at the top of
``run_batch.py`` and ``ensure_wildclaw_judge_env()`` is called before any
task dispatch (and before this module's own ``OPENROUTER_*``/``GLOBAL_*``
module-level constants are read from the environment).

SOURCE OF TRUTH / DEFENSE IN DEPTH
-----------------------------------
This module is the *hard requirement*. The five ``run_*.sh`` launchers keep
calling ``resolve_wildclaw_gateway`` + ``resolve_judge_env`` too -- that
stays in place as cheap defense-in-depth, not because this module trusts it:
if a launcher's own resolution ever regresses or is skipped, this module is
what actually stops the run rather than letting it limp along with empty
judge/usage credentials. The launchers also do something this module
deliberately does *not* attempt to reproduce: ``resolve_wildclaw_gateway``
brings up a whole per-experiment gateway subprocess (docker, a descriptor
file, a port) to produce ``GATEWAY_V1``/``GATEWAY_TOKEN`` in the first
place. This module only ever *reads* those two vars if something else
(a ``run_*.sh`` launcher, normally) already set them; it cannot synthesize
them from ``.env`` the way it can synthesize ``OPENROUTER_*`` from
``OPENAI_API_KEY``.

This module's ``ensure_wildclaw_judge_env`` implements the *exact same*
precedence as ``config_lib.sh``'s bash ``resolve_judge_env`` function (read
that function's own large comment block for the full rationale):

  1. ``JUDGE_MODEL`` defaults to ``gpt-5.5`` (see
     ``_default_judge_model_from_config_lib`` below for how that default is
     kept in sync with bash rather than copy-pasted) unless already set.
  2. If the resolved ``JUDGE_MODEL`` differs from ``RUNNER_MODEL`` (the
     model under test -- exported by ``config_lib.sh``'s
     ``load_runner_config``, or empty if unset) AND a real
     ``OPENAI_API_KEY`` is available (environment, or read directly out of
     the outer repo's ``.env``): route ``OPENROUTER_API_KEY``/
     ``OPENROUTER_BASE_URL`` at the real OpenAI API
     (``OPENAI_BASE_URL``, else ``OPENAI_API_BASE``, else
     ``https://api.openai.com/v1``) -- an independent judge.
  3. Otherwise (an explicit self-grade, or no ``OPENAI_API_KEY`` available):
     fall back to this run's own agent gateway credentials
     (``GATEWAY_TOKEN``/``GATEWAY_V1``), matching bash's behavior exactly.

One divergence from the bash function is deliberate: ``resolve_judge_env``
always "degrades" to the agent's own gateway credentials rather than ever
hard-failing, on the assumption that ``resolve_wildclaw_gateway`` already
succeeded (or the launcher already exited 1) by the time it runs, so
``GATEWAY_TOKEN``/``GATEWAY_V1`` are always available as a fallback in
practice. This module cannot assume that -- it must also cover direct
invocation, where *nothing* has been resolved -- so when neither branch above
has anything to fall back on, it raises ``WildClawCredentialError`` instead
of silently proceeding with empty credentials. That all-empty case is
precisely how the prior run's judge/usage credentials were silently
corrupted; this module exists to fail loudly instead.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, MutableMapping


class WildClawCredentialError(RuntimeError):
    """Raised when run_batch.py cannot resolve WildClawBench judge/gateway
    credentials through any path config_lib.sh's resolve_judge_env would
    also accept. See this module's docstring for the full precedence and
    the bash/Python source-of-truth split.
    """


# Last-resort fallback only -- see _default_judge_model_from_config_lib.
# Keep this equal to config_lib.sh's _WILDCLAW_DEFAULT_JUDGE_MODEL; it is
# used only if that file cannot be found/parsed (e.g. this repo checked out
# standalone, without the outer PyLM_Eval_why repo alongside it).
_FALLBACK_DEFAULT_JUDGE_MODEL = "gpt-5.5"

_THIS_FILE = Path(__file__).resolve()
# WildClawBench root (mirrors run_batch.py's own ROOT_DIR).
_WCB_ROOT = _THIS_FILE.parent.parent


def _find_outer_repo_root(start: Path) -> Path:
    """Walk upward from `start` (the WildClawBench root) looking for the
    outer PyLM_Eval_why checkout -- the sibling repo (benchmarks/WildClawBench
    is that repo's own git submodule, not an ancestor within this git repo)
    containing both `.env` and
    `eval_framework/baseline_verifier/wildclawbench/config_lib.sh`.

    Normally this is exactly two levels up (`start.parent.parent`), the same
    traversal every run_*.sh does for its own REPO_ROOT
    ("$SCRIPT_DIR/../../.."). But a git worktree checkout of WildClawBench
    itself (`.worktrees/<branch>/...`, used throughout this repo for
    isolated development -- see CLAUDE.md) adds extra nesting that a fixed
    two-level traversal misses. Falls back to the fixed two-level guess if
    nothing is found within a few levels, so this never raises.
    """
    candidate = start
    for _ in range(6):
        candidate = candidate.parent
        marker = (
            candidate
            / "eval_framework"
            / "baseline_verifier"
            / "wildclawbench"
            / "config_lib.sh"
        )
        if marker.is_file():
            return candidate
    return start.parent.parent


# Outer PyLM_Eval_why repo root, and the same directory whose .env every
# run_*.sh sources before calling into config_lib.sh.
_OUTER_REPO_ROOT = _find_outer_repo_root(_WCB_ROOT)
_ENV_FILE = _OUTER_REPO_ROOT / ".env"
_CONFIG_LIB_SH = (
    _OUTER_REPO_ROOT
    / "eval_framework"
    / "baseline_verifier"
    / "wildclawbench"
    / "config_lib.sh"
)

_JUDGE_MODEL_DEFAULT_RE = re.compile(r'_WILDCLAW_DEFAULT_JUDGE_MODEL="([^"]+)"')


def _default_judge_model_from_config_lib(config_lib_path: Path = _CONFIG_LIB_SH) -> str:
    """The default JUDGE_MODEL, read straight out of config_lib.sh rather
    than duplicated as a second literal that could silently drift from it.
    Falls back to _FALLBACK_DEFAULT_JUDGE_MODEL if the file is missing or
    its assignment no longer matches the expected shape -- this must never
    raise, since it runs unconditionally at run_batch.py startup.
    """
    try:
        text = config_lib_path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_DEFAULT_JUDGE_MODEL
    match = _JUDGE_MODEL_DEFAULT_RE.search(text)
    return match.group(1) if match else _FALLBACK_DEFAULT_JUDGE_MODEL


_DOTENV_LINE_RE = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$')


def _read_dotenv_values(keys: Iterable[str], path: Path = _ENV_FILE) -> dict[str, str]:
    """Minimal, defensive direct read of a fixed set of keys out of the
    outer repo's ``.env``.

    ``run_batch.py`` already calls python-dotenv's ``load_dotenv()`` at
    import time, which walks up from this file's directory looking for a
    ``.env`` and, in the normal case, finds this same outer-repo one and
    loads it into ``os.environ`` (without overwriting anything already
    set) -- so in the normal case this function never has anything to do:
    ``os.environ`` already has whatever ``.env`` had. This is a direct,
    best-effort fallback for the case that ordering changes,
    ``python-dotenv``'s frame-based search does not find the file (e.g. this
    module is exercised directly, outside of ``run_batch.py``, such as in a
    unit test), or the package is unavailable. It intentionally only
    understands simple ``KEY=value`` / ``export KEY=value`` lines
    (optionally quoted), not full bash syntax -- ``run_*.sh`` itself sources
    this same file with real bash (``set -a; source "$REPO_ROOT/.env"; set
    +a``), and that remains the authoritative parse; this is only a safety
    net for the Python side.
    """
    keys = set(keys)
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _DOTENV_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if key not in keys:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def ensure_wildclaw_judge_env(env: MutableMapping[str, str] | None = None) -> None:
    """Resolve ``JUDGE_MODEL``/``OPENROUTER_API_KEY``/``OPENROUTER_BASE_URL``
    in ``env`` (defaults to ``os.environ``), in place, using the exact same
    precedence as ``config_lib.sh``'s ``resolve_judge_env``. See this
    module's docstring for the full rationale and the one deliberate
    divergence (a hard failure here instead of resolve_judge_env's
    unconditional degrade).

    Never overwrites a value already present in ``env``: if one of the
    ``run_*.sh`` launchers already resolved these (the normal case), this
    is a no-op beyond possibly backfilling a still-missing ``JUDGE_MODEL``.

    Raises ``WildClawCredentialError`` if no judge credential -- real OpenAI
    or this run's own agent gateway -- can be resolved at all.
    """
    if env is None:
        env = os.environ

    # Pass the module-level path globals explicitly (rather than relying on
    # each helper's own default parameter, which is bound once at import
    # time) so tests can monkeypatch _ENV_FILE/_CONFIG_LIB_SH on this module
    # and have that actually take effect here.
    default_judge_model = _default_judge_model_from_config_lib(_CONFIG_LIB_SH)
    judge_model = env.get("JUDGE_MODEL") or default_judge_model
    runner_model = env.get("RUNNER_MODEL", "")

    already_resolved = bool(env.get("OPENROUTER_API_KEY")) and bool(
        env.get("OPENROUTER_BASE_URL")
    )
    if already_resolved:
        env.setdefault("JUDGE_MODEL", judge_model)
        return

    openai_key = env.get("OPENAI_API_KEY", "")
    openai_base = env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE") or ""
    if not openai_key:
        dotenv_values = _read_dotenv_values(
            {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"}, _ENV_FILE
        )
        openai_key = dotenv_values.get("OPENAI_API_KEY", "")
        openai_base = (
            openai_base
            or dotenv_values.get("OPENAI_BASE_URL")
            or dotenv_values.get("OPENAI_API_BASE")
            or ""
        )
    openai_base = openai_base or "https://api.openai.com/v1"

    gateway_token = env.get("GATEWAY_TOKEN", "")
    gateway_v1 = env.get("GATEWAY_V1", "")

    if judge_model != runner_model and openai_key:
        resolved_key, resolved_base = openai_key, openai_base
    elif gateway_token and gateway_v1:
        resolved_key, resolved_base = gateway_token, gateway_v1
    else:
        raise WildClawCredentialError(
            "WildClawBench judge/gateway credentials could not be resolved: "
            "no OPENROUTER_API_KEY/OPENROUTER_BASE_URL already set, no "
            "usable OPENAI_API_KEY (checked the environment and "
            f"{_ENV_FILE}), and no agent gateway credential "
            "(GATEWAY_TOKEN/GATEWAY_V1) either. run_batch.py must be "
            "invoked through one of eval_framework/baseline_verifier/"
            "wildclawbench/run_{pylm,codex,claudecode,hermesagent,"
            "openclaw}.sh (or script/run.sh <backend>, which delegates to "
            "one of those) -- those resolve this run's per-experiment "
            "inference gateway and an independent LLM judge before ever "
            "reaching run_batch.py. Calling `python eval/run_batch.py` "
            "directly, without going through one of those launchers, is "
            "exactly what silently corrupted a prior benchmark run's "
            "judge/usage credentials; this check exists to fail loudly "
            "instead of repeating that."
        )

    env.setdefault("JUDGE_MODEL", judge_model)
    env.setdefault("OPENROUTER_API_KEY", resolved_key)
    env.setdefault("OPENROUTER_BASE_URL", resolved_base)
