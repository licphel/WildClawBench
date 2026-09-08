#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash script/run.sh <openclaw|claudecode|codex|hermesagent|pylm> [run_batch args...]

The benchmark model is fixed by the repository runner config (gpt-5.6-terra).
Each invocation gets its own output directory and inference gateway.

Examples:
  bash script/run.sh openclaw --category 01_Productivity_Flow --parallel 1
  bash script/run.sh pylm --task tasks/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling.md
EOF
  exit 1
fi

backend="$1"
shift

case "$backend" in
  openclaw|claudecode|codex|hermesagent|pylm)
    ;;
  *)
    echo "Unknown backend: $backend" >&2
    echo "Expected one of: openclaw, claudecode, codex, hermesagent, pylm" >&2
    exit 1
    ;;
esac

WCB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WCB_ROOT/../.." && pwd)"
# The outer eval_framework adapter deliberately supplies an absolute output
# root and sets this to 0; all standalone unified-entrypoint invocations use
# the manifest-backed layout.
if [[ "${WILDCLAW_RUN_LAYOUT:-}" != "0" ]]; then
  export WILDCLAW_RUN_LAYOUT=1
fi
export WILDCLAW_RUN_LABEL="${WILDCLAW_RUN_LABEL:-$backend}"

exec bash "$REPO_ROOT/eval_framework/baseline_verifier/wildclawbench/run_${backend}.sh" "$@"
