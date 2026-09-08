"""One place that says what approval/sandbox posture each baseline runs under.

WildClaw skips approvals for all five baselines, and it did so correctly, but
until this module existed the *declaration* lived in three unrelated kinds of
place and only two of them were in the repository:

    codex      harness argv    ``--dangerously-bypass-approvals-and-sandbox``
    perdura    harness argv    ``--sandbox-mode dangerous_skip``
    claude     Docker layer    ``/claude_code/start.sh`` ran
                               ``claude --dangerously-skip-permissions``; the
                               harness only set ``IS_SANDBOX=1``
    openclaw   Docker layer    ``/root/.openclaw/exec-approvals.json`` and
                               ``/root/.openclaw/openclaw.json``
    hermes     nowhere         it fell through to hermes-agent's own default
                               ``approvals.mode: "manual"`` and was saved only
                               by ``tools/approval.py``'s container
                               short-circuit

Two costs, both paid. Answering "what was the approval posture of this run?"
meant opening two Docker images -- a grep of this whole tree for
``dangerously-skip-permissions|bypassPermissions|permission-mode`` or for
openclaw's ``security``/``ask`` returned nothing. And a posture nothing in the
repository states is a posture nothing can notice drifting: hermes had no
declaration at all, and no test, no log line and no run artifact objected.

So this module is the single point of declaration, and every wired runner both
reads its command line / config out of here and writes what it applied into the
run's own output directory (``approval_posture.json``, via :func:`record`), so
the posture of a finished run is recoverable from its artifacts without a
container, an image, or this file.

Two deliberate properties:

* ``argv`` and ``config`` are the real thing, not a description of it. A runner
  interpolates ``posture.argv`` into the command it execs and writes
  ``posture.config`` through the CLI's own config surface. A comment here can
  go stale silently; a value the run is built from cannot.
* The image-side settings are, at the time of writing, still present. Claude's
  ``start.sh`` puts ``--dangerously-skip-permissions`` before its ``"$@"``, so
  the harness flag arrives as a harmless repeat, and openclaw's baked JSON
  agrees key for key with what the harness now writes. That overlap is
  intentional: an interval in which neither side declared the posture would be
  worse than the state this replaces. Removing the baked files is a Dockerfile
  change and belongs to whoever owns the image build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ApprovalPosture:
    """How one baseline is told to skip approvals, and how to say so afterwards."""

    baseline: str
    #: ``"argv"`` when the flags below are the declaration, ``"config"`` when
    #: the config mapping is, ``"argv+config"`` when a harness needs both.
    mechanism: str
    #: Flags the harness interpolates into the command it runs.
    argv: tuple[str, ...] = ()
    #: Settings the harness writes through the agent's own config surface.
    #: Keys are dotted paths in the agent's config namespace.
    config: dict[str, Any] = field(default_factory=dict)
    #: Files the harness materializes verbatim inside the container.
    files: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


CODEX = ApprovalPosture(
    baseline="codex",
    mechanism="argv",
    argv=("--dangerously-bypass-approvals-and-sandbox",),
    config={"approval_policy": "never"},
    rationale=(
        "Codex's documented default asks before every command and there is no "
        "UI in a headless container to answer with. The flag is codex's own "
        "name for the bypass; approval_policy is stated too so a config read "
        "and an argv read agree."
    ),
)

CLAUDE = ApprovalPosture(
    baseline="claude",
    mechanism="argv",
    argv=("--dangerously-skip-permissions",),
    rationale=(
        "Claude Code's equivalent of the codex flag. It used to reach the CLI "
        "only from inside /claude_code/start.sh, i.e. from a Docker layer, so "
        "the harness could not be read to find out what policy a run had; the "
        "harness now passes it, and start.sh's own copy is a repeat rather "
        "than the source."
    ),
)

OPENCLAW = ApprovalPosture(
    baseline="openclaw",
    mechanism="argv+config",
    config={
        "tools.profile": "full",
        "tools.exec.security": "full",
        "tools.exec.ask": "off",
    },
    files={
        "/root/.openclaw/exec-approvals.json": {
            "version": 1,
            "defaults": {"security": "full", "ask": "off"},
            "agents": {"*": {"allowlist": [{"pattern": "*"}]}},
        }
    },
    rationale=(
        "OpenClaw has no bypass flag; its exec defaults are security \"deny\" "
        "with ask \"on-miss\" and an askFallback of deny, so a headless run "
        "either blocks on an approval nobody will give or has every exec "
        "denied. The same three keys the outer "
        "eval_framework/backends/openclaw_backend.py writes are written here, "
        "plus the exec-approvals allowlist the CLI reads separately from "
        "openclaw.json."
    ),
)

HERMES = ApprovalPosture(
    baseline="hermes",
    mechanism="config",
    config={"approvals.mode": "off"},
    rationale=(
        "hermes-agent's DEFAULT_CONFIG is approvals.mode: \"manual\", deep-"
        "merged under any user config, so omitting the block selected manual "
        "approvals. Nothing failed only because "
        "tools/approval.py:check_all_command_guards short-circuits to approved "
        "for env_type in (\"docker\", ...) before it ever reads the mode -- a "
        "policy that holds while an implementation detail stays put is not a "
        "declared policy."
    ),
)

PERDURA = ApprovalPosture(
    baseline="perdura",
    mechanism="argv",
    argv=("--sandbox-mode", "dangerous_skip"),
    rationale=(
        "Perdura denies a tool call under its default policy and falls back to "
        "request_access(), which suspends a headless run waiting for an "
        "operator approval that never arrives. Naming dangerous_skip is itself "
        "the complete statement of intent -- see "
        "apps/perdura-cli/src/perdura/cli/sandbox_confirmation.py, which says "
        "outright that no separate confirmation flag exists -- and it forces "
        "access_level=full plus the File/Network/Tool ANY wildcard."
    ),
)

POSTURES: dict[str, ApprovalPosture] = {
    posture.baseline: posture
    for posture in (CODEX, CLAUDE, OPENCLAW, HERMES, PERDURA)
}

#: Written into each task's output directory.
ARTIFACT_NAME = "approval_posture.json"


def record(
    output_dir: Path,
    posture: ApprovalPosture,
    *,
    applied: dict[str, Any] | None = None,
) -> Path:
    """Write what this task actually applied, so the run can be audited later.

    ``applied`` is the runner's own account of how it delivered the posture --
    the command fragment it interpolated, the config commands it ran and their
    exit codes. It is separate from ``posture`` on purpose: ``posture`` is what
    was intended and ``applied`` is what happened, and a run where those differ
    is exactly the run somebody will need to reconstruct.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ARTIFACT_NAME
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline": posture.baseline,
        "mechanism": posture.mechanism,
        "argv": list(posture.argv),
        "config": dict(posture.config),
        "files": dict(posture.files),
        "rationale": posture.rationale,
        "applied": applied or {},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


__all__ = (
    "ARTIFACT_NAME",
    "ApprovalPosture",
    "CLAUDE",
    "CODEX",
    "HERMES",
    "OPENCLAW",
    "PERDURA",
    "POSTURES",
    "record",
)
