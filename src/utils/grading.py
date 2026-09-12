from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from src.utils.transient_errors import is_transient_error

logger = logging.getLogger(__name__)

load_dotenv()
TMP_WORKSPACE = "/tmp_workspace"
MAX_GRADING_RETRIES = 2

GRADER_PYTHON = "/opt/wildclaw-grader/bin/python"
"""The one interpreter every baseline is graded with.

Grading runs inside the agent's own container, and the five images put five
different interpreters on ``python3`` -- openclaw/claudecode ``/usr/bin/python3``
3.10.12, codex a conda env 3.11.15, hermes ``/opt/hermes/.venv`` 3.12.13, pylm a
PyReduce venv 3.13.15 -- with different packages behind them. That made the same
``automated_checks`` score differently per baseline: the official claudecode
image has neither pymupdf nor PyPDF2, so ``06_Safety_Alignment_task_1``'s
``looks_like_mae_pdf()`` took its ``return True`` fallback and unlocked 0.5 of
that task for one baseline only, and ``05_..._task_8`` imports ``bs4`` at
grading time where two images have none. The images now all carry this venv at
this fixed path.

The agent-visible ``python3`` was separately aligned afterwards: codex,
claudecode, hermes and pylm now all resolve it to ``/root/miniconda3/envs/eval``
(CPython 3.12.13), the environment the tasks name by absolute path. openclaw is
the deliberate exception and still resolves ``/usr/bin/python3`` 3.10.12 -- see
the note in eval_framework/baseline_verifier/wildclawbench/Dockerfile.openclaw.
That alignment is independent of this constant: grading uses GRADER_PYTHON
either way, so scores stay comparable regardless of what the agent runs under.
"""


def _grader_interpreter(task_id: str) -> str:
    """``GRADER_PYTHON`` if the container has it, else the agent's ``python3``.

    A hard switch would break every container built before the grader landed.
    The fallback is deliberately loud rather than silent: grading on the agent's
    own interpreter is exactly the defect this replaces, so a run that does it
    has to say so in its log instead of quietly producing numbers that are not
    comparable across baselines.
    """

    probe = subprocess.run(
        ["docker", "exec", task_id, "test", "-x", GRADER_PYTHON],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode == 0:
        return GRADER_PYTHON
    logger.warning(
        "[%s] %s missing; grading falls back to the agent image's own python3. "
        "Scores from this run are NOT comparable across baselines -- rebuild the "
        "image with the shared grader layer.",
        task_id,
        GRADER_PYTHON,
    )
    return "python3"


def _write_score(output_dir: Path, task_id: str, scores: dict) -> None:
    score_path = output_dir / "score.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(
        json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[%s] Grading results written to → %s", task_id, score_path)


def _error_score(output_dir: Path, task_id: str, message: str) -> dict:
    scores = {"overall_score": 0.0, "error": message}
    _write_score(output_dir, task_id, scores)
    return scores


def _grading_error(
    output_dir: Path,
    task_id: str,
    message: str,
    write_error_score: bool,
) -> dict:
    if write_error_score:
        return _error_score(output_dir, task_id, message)
    return {"error": message}


def write_error_score(output_dir: Path, task_id: str, message: str) -> dict:
    return _error_score(output_dir, task_id, message)


def run_grading(
    task_id: str,
    automated_checks: str,
    output_dir: Path,
    extra_env: str = "",
    lobster_env: list[str] | None = None,
    transcript_container_path: str = "",
    write_error_score: bool = False,
) -> dict:
    logger.info("[%s] Starting in-container grading...", task_id)

    loader_src = Path(__file__).with_name("transcript_loader.py")
    if not loader_src.exists():
        logger.error("[%s] transcript loader module not found: %s", task_id, loader_src)
        return _grading_error(
            output_dir,
            task_id,
            f"transcript loader module not found: {loader_src}",
            write_error_score,
        )

    runner_code = "\n".join([
        "import json",
        "from _transcript_loader import load_transcript",
        f"_transcript = load_transcript({json.dumps(transcript_container_path)})",
        "",
        automated_checks,
        "",
        f'result = grade(transcript=_transcript, workspace_path="{TMP_WORKSPACE}")',
        "print(json.dumps(result))",
    ]) + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(runner_code)
        runner_host = f.name

    try:
        r_loader = subprocess.run(
            ["docker", "cp", str(loader_src), f"{task_id}:/tmp/_transcript_loader.py"],
            capture_output=True, text=True,
        )
        if r_loader.returncode != 0:
            logger.error("[%s] docker cp transcript loader failed: %s", task_id, r_loader.stderr)
            return _grading_error(
                output_dir,
                task_id,
                f"docker cp transcript loader failed: {r_loader.stderr}",
                write_error_score,
            )

        r = subprocess.run(
            ["docker", "cp", runner_host, f"{task_id}:/tmp/_grade_runner.py"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.error("[%s] docker cp failed: %s", task_id, r.stderr)
            return _grading_error(
                output_dir,
                task_id,
                f"docker cp failed: {r.stderr}",
                write_error_score,
            )

        env_args: list[str] = []
        for line in extra_env.splitlines():
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            value = os.environ.get(key, "")
            env_args += ["-e", f"{key}={value}"]
            masked = (value[:4] + "***") if value else "(empty)"
            logger.info("[%s] Injecting grading env: %s=%s", task_id, key, masked)

        for key in (lobster_env or []):
            value = os.environ.get(key, "")
            if not value:
                logger.warning("[%s] Grading lobster env key %s not found, skipping", task_id, key)
                continue
            env_args += ["-e", f"{key}={value}"]
            masked = value[:4] + "***"
            logger.info("[%s] Injecting grading lobster env: %s=%s", task_id, key, masked)

        # Some tasks' grade() functions call an LLM judge over the same relay
        # (OPENROUTER_API_KEY/BASE_URL, injected above) that the agent itself
        # uses — subject to the same transient upstream hiccups. Grading is
        # read-only over an already-collected transcript/workspace, so
        # re-running it is safe and idempotent; retry it in place rather than
        # letting one relay blip silently zero out an otherwise-good task.
        r = None
        interpreter = _grader_interpreter(task_id)
        for attempt in range(1, MAX_GRADING_RETRIES + 2):
            r = subprocess.run(
                ["docker", "exec", *env_args, task_id, interpreter, "/tmp/_grade_runner.py"],
                capture_output=True,
                text=True,
                timeout=3200,
            )
            if r.returncode == 0:
                break
            if not is_transient_error(r.stderr) or attempt == MAX_GRADING_RETRIES + 1:
                logger.error("[%s] Grading script execution failed: %s", task_id, r.stderr)
                return _grading_error(
                    output_dir,
                    task_id,
                    f"grade script failed: {r.stderr}",
                    write_error_score,
                )
            logger.warning(
                "[%s] Grading script hit a transient error (attempt %d/%d), retrying: %s",
                task_id, attempt, MAX_GRADING_RETRIES + 1, r.stderr[:300],
            )

        try:
            scores = json.loads(r.stdout.strip())
        except json.JSONDecodeError:
            scores = None
            for line in reversed(r.stdout.strip().splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        scores = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            if scores is None:
                logger.error("[%s] Failed to parse grading result, no valid JSON found in stdout\nstdout: %s", task_id, r.stdout[:500])
                return _grading_error(
                    output_dir,
                    task_id,
                    "json parse failed: no valid JSON in stdout",
                    write_error_score,
                )

    finally:
        Path(runner_host).unlink(missing_ok=True)

    _write_score(output_dir, task_id, scores)
    return scores


def format_scores(task_id: str, scores: dict) -> str:
    if "error" in scores and not any(
        isinstance(v, (int, float)) for v in scores.values()
    ):
        return f"[{task_id}] Grading error: {scores['error']}"
    lines = [f"\n{'='*60}", f"  {task_id}", f"{'='*60}"]

    for k, v in scores.items():
        if isinstance(v, (int, float)):
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"  {bar} {v:.2f}  {k}")

    lines.append("=" * 60)
    return "\n".join(lines)

def print_summary(results: list[dict], category: str, output_dir: Path, model_name: str) -> None:
    print(f"\n{'#'*60}")
    print(f"  Summary Report — {category}")
    print(f"{'#'*60}")

    all_scores: dict[str, float] = {}
    for r in results:
        task_id = r["task_id"]
        scores = r['scores']
        if not scores:
            if r.get("error"):
                print(f"  ✗ {task_id}: {r['error']}")
            else:
                print(f"  - {task_id}: No scores")
            continue
        numeric_dict = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
        
        if not numeric_dict:
            if "error" in scores:
                print(f"  ✗ {task_id}: Grading error {scores['error']}")
            else:
                print(f"  - {task_id}: No valid numeric scores")
            continue

        avg = sum(numeric_dict.values()) / len(numeric_dict)
        status = "!" if r.get("error") or scores.get("error") else "✓"
        note = ""
        if r.get("error"):
            note = f" agent_error={r['error']}"
        elif scores.get("error"):
            note = f" grading_error={scores['error']}"
        print(f"  {status} {task_id}: avg {avg:.2f}  ({len(numeric_dict)} items){note}")

        final_score_val = numeric_dict.get('overall_score', avg)
        all_scores[task_id] = final_score_val

    if all_scores:
        print(f"\n  Final scores per task:")
        for k, score in sorted(all_scores.items()):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"    {bar} {score:.2f}  {k}")

    print(f"\n  Token usage and cost per task:")
    print(f"    {'Task ID':<55} {'Output Tokens':>12} {'Cost(USD)':>12}")
    print(f"    {'-'*55} {'-'*12} {'-'*12}")
    total_output_tokens = 0
    total_cost_usd = 0.0
    unknown_cost_tasks = []
    for r in sorted(results, key=lambda x: x["task_id"]):
        usage = r.get("usage", {})
        out_tok = usage.get("output_tokens")
        if not isinstance(out_tok, (int, float)):
            out_tok = 0
        raw_cost = usage.get("cost_usd")
        cost = raw_cost if isinstance(raw_cost, (int, float)) else 0.0
        if not isinstance(raw_cost, (int, float)):
            unknown_cost_tasks.append(r["task_id"])
        total_output_tokens += out_tok
        total_cost_usd += cost
        cost_display = f"{cost:>11.4f}$" if isinstance(raw_cost, (int, float)) else "        N/A"
        print(f"    {r['task_id']:<55} {out_tok:>12} {cost_display}")
    print(f"    {'Total':<55} {total_output_tokens:>12} {total_cost_usd:>11.4f}$")
    if unknown_cost_tasks:
        print(f"    Cost unavailable for {len(unknown_cost_tasks)} task(s); total is known costs only")

    summary_path = output_dir / category / f"summary_{model_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n  Summary written to → {summary_path}")
    print("#" * 60)

def _finite_number(value: object) -> int | float | None:
    """Return numeric JSON values only; match OpenClaw's finite-number rule."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _token_count(value: object) -> int | None:
    number = _finite_number(value)
    if number is None:
        return None
    return max(0, int(number))


def _pick_token(mapping: dict, *keys: str) -> tuple[int | None, str | None]:
    for key in keys:
        if key in mapping:
            value = _token_count(mapping[key])
            if value is not None:
                return value, key
    return None, None


def _pick_cost(mapping: dict, *keys: str) -> float | None:
    for key in keys:
        if key in mapping:
            value = _finite_number(mapping[key])
            if value is not None:
                return max(0.0, float(value))
    return None


_USAGE_KEYS = frozenset(
    {
        "input",
        "output",
        "cacheRead",
        "cacheWrite",
        "inputTokens",
        "outputTokens",
        "promptTokens",
        "completionTokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cache_read",
        "cache_write",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "cached_input_tokens",
        "cached_tokens",
        "cached",
        "cache_creation_input_tokens",
        "total",
        "totalTokens",
        "total_tokens",
        "cost",
        "cost_usd",
        "total_cost_usd",
    }
)


def _looks_like_usage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if _USAGE_KEYS.intersection(value):
        return True
    return any(
        isinstance(value.get(key), dict)
        and bool(
            {"cached_tokens", "cache_write_tokens", "cache_creation_input_tokens"}
            .intersection(value[key])
        )
        for key in ("input_tokens_details", "prompt_tokens_details")
    )


def _find_usage_block(entry: dict, message: dict) -> dict:
    """Find one native usage snapshot without counting the same event twice.

    OpenClaw normally persists ``message.usage``.  CLI/provider adapters can
    instead put the snapshot in ``stats`` or on the event itself, and some
    versions use snake_case OpenAI names.  Prefer a populated message block,
    while retaining an empty block as a faithful zero-usage snapshot.
    """
    fallback: dict | None = None
    containers = (message, entry)
    for container in containers:
        for key in (
            "usage",
            "token_usage",
            "tokenUsage",
            "stats",
            "last_token_usage",
            "lastTokenUsage",
        ):
            candidate = container.get(key)
            if not isinstance(candidate, dict):
                continue
            if fallback is None:
                fallback = candidate
            if _looks_like_usage(candidate):
                return candidate
    return fallback or {}


def _normalize_transcript_usage(usage: dict, entry: dict | None = None) -> dict:
    """Normalize the usage variants accepted by OpenClaw's ``normalizeUsage``.

    OpenClaw's persisted assistant messages are usually disjoint
    ``input/output/cacheRead/cacheWrite`` buckets, but provider-shaped records
    can use OpenAI Responses/Chat or CLI names.  In those formats the prompt
    total includes cache buckets, so subtract them from input just as
    OpenClaw does before writing its normalized assistant message.
    """
    cache_read, _ = _pick_token(
        usage,
        "cacheRead",
        "cache_read",
        "cacheReadTokens",
        "cache_read_tokens",
        "cache_read_input_tokens",
        "cached_input_tokens",
        "cached",
        "cached_tokens",
    )
    cache_write, _ = _pick_token(
        usage,
        "cacheWrite",
        "cache_write",
        "cacheWriteTokens",
        "cache_write_tokens",
        "cache_creation_input_tokens",
        "cache_write_input_tokens",
    )
    for detail_key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(detail_key)
        if not isinstance(details, dict):
            continue
        if cache_read is None:
            cache_read, _ = _pick_token(details, "cached_tokens")
        if cache_write is None:
            cache_write, _ = _pick_token(
                details, "cache_write_tokens", "cache_creation_input_tokens"
            )

    raw_input, input_key = _pick_token(
        usage,
        "input",
        "inputTokens",
        "input_tokens",
        "promptTokens",
        "prompt_tokens",
        "prompt_n",
    )
    raw_output, _ = _pick_token(
        usage,
        "output",
        "outputTokens",
        "output_tokens",
        "completionTokens",
        "completion_tokens",
        "predicted_n",
    )

    # ``input`` is OpenClaw's already-disjoint bucket.  The other provider
    # aliases are prompt totals when a cache detail is present, so normalize
    # them to the same disjoint convention.
    has_cached_alias = any(
        key in usage
        for key in (
            "cached_input_tokens",
            "cached",
            "cached_tokens",
            "cache_read_input_tokens",
        )
    )
    has_cache_detail = any(
        isinstance(usage.get(key), dict)
        and any(
            detail in usage[key]
            for detail in (
                "cached_tokens",
                "cache_write_tokens",
                "cache_creation_input_tokens",
            )
        )
        for key in ("input_tokens_details", "prompt_tokens_details")
    )
    if raw_input is not None and input_key != "input":
        if has_cached_alias or has_cache_detail:
            raw_input -= cache_read or 0
        if has_cache_detail or "cache_write_input_tokens" in usage:
            raw_input -= cache_write or 0
    input_tokens = _token_count(raw_input)

    total_tokens, _ = _pick_token(usage, "total", "totalTokens", "total_tokens")
    if total_tokens is None:
        total_tokens = sum(
            value or 0 for value in (input_tokens, raw_output, cache_read, cache_write)
        )

    cost: float | None = None
    cost_value = usage.get("cost")
    if isinstance(cost_value, dict):
        cost = _pick_cost(cost_value, "total", "total_usd", "totalUSD")
    else:
        cost = _pick_cost(usage, "cost", "cost_usd", "costUSD", "total_cost_usd")
    if cost is None and isinstance(entry, dict):
        entry_cost = entry.get("cost")
        if isinstance(entry_cost, dict):
            cost = _pick_cost(entry_cost, "total", "total_usd", "totalUSD")
        else:
            cost = _pick_cost(entry, "cost_usd", "costUSD", "total_cost_usd")

    return {
        "input_tokens": input_tokens or 0,
        "output_tokens": raw_output or 0,
        "cache_read_tokens": cache_read or 0,
        "cache_write_tokens": cache_write or 0,
        "total_tokens": total_tokens or 0,
        "cost_usd": cost or 0.0,
    }


def extract_usage_from_jsonl(jsonl_path: Path) -> dict:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }
    if not jsonl_path.exists():
        return totals
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        msg = entry.get("message")
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            assistant_message = msg
        elif entry.get("role") == "assistant":
            assistant_message = entry
        elif entry.get("type") in {"assistant", "assistant_message"}:
            assistant_message = msg if isinstance(msg, dict) else entry
        else:
            continue
        totals["request_count"] += 1
        usage = _normalize_transcript_usage(
            _find_usage_block(entry, assistant_message), entry
        )
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
            "cost_usd",
        ):
            totals[key] += usage[key]
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    return totals

def print_global_summary(results: list[dict], output_dir: Path, model_name: str) -> None:
    print(f"\n{'#'*60}")
    print(f"  Global Summary Report — ALL CATEGORIES")
    print(f"{'#'*60}")

    total_tasks = len(results)
    scored_tasks = 0
    missing_score_tasks = 0
    total_score = 0.0
    for r in results:
        scores = r.get("scores", {})
        numeric = {
            k: v
            for k, v in scores.items()
            if isinstance(v, (int, float))
        } if scores else {}
        if not numeric:
            missing_score_tasks += 1
            continue
        final = numeric.get("overall_score", sum(numeric.values()) / len(numeric))
        total_score += final
        scored_tasks += 1

    global_avg = 0.0
    if total_tasks > 0:
        global_avg = total_score / total_tasks
        bar = "█" * int(global_avg * 10) + "░" * (10 - int(global_avg * 10))
        print(f"\n  Completed tasks: {scored_tasks} / {total_tasks}")
        print(f"  Tasks without a valid score.json: {missing_score_tasks}")
        if missing_score_tasks > 0:
            print("  Possible causes: task execution failed, such as OOM, or grading failed.")
        print(f"  Global average: {bar} {global_avg:.4f}")
    else:
        print("  No tasks found")

    total_out_tok = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
    total_cost    = sum(r.get("usage", {}).get("cost_usd",      0.0) for r in results)
    print(f"  Total output tokens: {total_out_tok}   Total cost: ${total_cost:.4f}")

    summary_path = output_dir / f"summary_all_{model_name}.json"
    summary_path.write_text(
        json.dumps(
            {"global_avg": global_avg if total_tasks else None,
             "task_count": total_tasks,
             "scored_task_count": scored_tasks,
             "missing_score_task_count": missing_score_tasks,
             "results": results},
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n  Global summary written to → {summary_path}")
    print("#" * 60)
