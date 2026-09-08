#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash script/run.sh openclaw    [run_batch args...]
  bash script/run.sh claudecode  [run_batch args...]
  bash script/run.sh codex       [run_batch args...]
  bash script/run.sh hermesagent [run_batch args...]
  bash script/run.sh pylm        [run_batch args...]

Examples:
  bash script/run.sh openclaw --category all --parallel 4 --model openrouter/openai/gpt-5.5
  bash script/run.sh claudecode --category all --parallel 4 --model openai/gpt-5.5
  bash script/run.sh codex --category all --parallel 4 --model openrouter/openai/gpt-5.5
  bash script/run.sh hermesagent --category all --parallel 4 --model openai/gpt-5.5
  bash script/run.sh pylm        --category all --model gpt-5.6-terra

  bash script/run.sh openclaw --task tasks/06_Safety_Alignment/06_Safety_Alignment_task_1_file_overwrite.md --model openrouter/openai/gpt-5.5
EOF
  exit 1
fi

backend="$1"
shift || true

case "$backend" in
  openclaw)
    exec python3 eval/run_batch.py --agent-backend openclaw "$@"
    ;;
  claudecode)
    exec python3 eval/run_batch.py --agent-backend claudecode "$@"
    ;;
  codex)
    exec python3 eval/run_batch.py --agent-backend codex "$@"
    ;;
  hermesagent)
    exec python3 eval/run_batch.py --agent-backend hermesagent "$@"
    ;;
  pylm)
    # Deliberately not the one-line `exec python3 eval/run_batch.py
    # --agent-backend pylm "$@"` the four branches above use.
    #
    # Those four backends read their credentials straight out of the
    # environment, and a bare invocation without them fails immediately and
    # visibly (empty OPENROUTER_API_KEY -> the CLI cannot authenticate).  pylm
    # does not fail that way.  PyLMAgent (src/agents/pylm/runner.py) takes its
    # own provider from GLOBAL_API_KEY/GLOBAL_API_BASE, and it never calls the
    # OPENROUTER_* fallback in eval_framework/wildclaw_cli_runner.run_cli_task
    # that the other baselines get -- so a bare run would start, execute, and
    # then grade every task through an LLM judge holding an empty key.  A
    # backend that silently scores wrong is worse than one that will not start.
    #
    # So this branch goes through the repo-side launcher, which is where
    # perdura's environment is actually assembled: the inference gateway
    # (credentials + NO_PROXY_INNER for the injected in-container http_proxy),
    # GLOBAL_API_*, OPENROUTER_* for the ~75% of tasks that declare them,
    # JUDGE_MODEL, and WILDCLAW_PYREDUCE_IMAGE.  That launcher then execs this
    # same eval/run_batch.py --agent-backend pylm, so there is still exactly
    # one execution path -- and one place where its environment is defined.
    #
    # Reaching into the parent repo is the existing arrangement for this
    # backend, not a new coupling: src/agents/pylm/runner.py already imports
    # eval_framework.wildclaw_cli_runner from parents[5].
    pylm_launcher="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)/eval_framework/baseline_verifier/wildclawbench/run_pylm.sh"
    if [ ! -f "$pylm_launcher" ]; then
      echo "pylm harness needs the PyLM_Eval launcher, not found at:" >&2
      echo "  $pylm_launcher" >&2
      echo "It assembles perdura's gateway credentials, OPENROUTER_*, NO_PROXY_INNER" >&2
      echo "and WILDCLAW_PYREDUCE_IMAGE; running run_batch.py --agent-backend pylm" >&2
      echo "without it grades every task with an empty judge key." >&2
      exit 1
    fi
    exec bash "$pylm_launcher" "$@"
    ;;
  *)
    echo "Unknown backend: $backend"
    echo "Expected one of: openclaw, claudecode, codex, hermesagent, pylm"
    exit 1
    ;;
esac
