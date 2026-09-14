"""run_batch.py must import cleanly in both ways it is actually loaded:

* as a script (every run_*.sh launcher: `python3 eval/run_batch.py ...`);
* as `eval.run_batch`, a submodule of the `eval` namespace package
  (eval_framework/wildclaw_cli_runner.py's pylm transcript-grading reuse
  path: `from eval.run_batch import grade_the_task, save_usage`).

Only the WildClawBench root is put on sys.path for that second case (see
wildclaw_cli_runner.py's own sys.path setup) -- eval/ itself is not. A bare
`import credential_resolution` inside run_batch.py therefore only works
because run_batch.py explicitly adds its own directory to sys.path first;
this guards that against regressing back to a plain
`sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))` +
`import credential_resolution`, which works when run_batch.py is executed
directly (its own directory is sys.path[0] automatically) but raises
ModuleNotFoundError under the `eval.run_batch` import path above.

Runs a real subprocess rather than importing in-process: importing
run_batch.py directly in-process would need real judge/gateway credentials
(it calls ensure_wildclaw_judge_env() at import time) and would drag in
docker_utils/agent backends into this test process for no reason. The
subprocess here supplies a real OPENAI_API_KEY so the credential check
itself succeeds and any failure is specifically about the import mechanism.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_WCB_ROOT = Path(__file__).resolve().parents[2]


def test_eval_run_batch_importable_as_package_submodule():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "from eval.run_batch import grade_the_task, save_usage",
        ],
        cwd=str(_WCB_ROOT),
        env={"PATH": "/usr/bin:/bin", "OPENAI_API_KEY": "sk-test-dummy"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
