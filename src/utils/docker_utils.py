from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path, PurePosixPath
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# -grader: carries the shared grading interpreter (src/utils/grading.py::
# GRADER_PYTHON). The agent-visible environment is unchanged from v1.3-node.
# openclaw's task image.  Renamed from wildclawbench-ubuntu:v1.3-node-grader
# when the OpenClaw CLI stopped being whatever the base image was built with
# and became a build argument (Dockerfile.openclaw ARG OPENCLAW_VERSION);
# the tag now names the CLI release, as the codex and claudecode tags do.
DOCKER_IMAGE  = os.environ.get("DOCKER_IMAGE", "").strip()
TMP_WORKSPACE = "/tmp_workspace"
WORKSPACE_BASELINE_PATH = "/tmp/wildclaw_workspace_baseline.json"

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip()


_DEFAULT_CONTAINER_NO_PROXY = "localhost,127.0.0.1,::1,host.docker.internal"


def _rewrite_proxy_host(value: str) -> str:
    """Make a host-local proxy reachable from a bridged task container."""
    if not value:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return value
    replacement = os.environ.get("BENCHMARK_CONTAINER_PROXY_HOST", "host.docker.internal")
    netloc = replacement
    if parsed.port is not None:
        netloc = f"{replacement}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def container_proxy_env(source: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve the proxy environment from the host to the container viewpoint.

    ``HTTP_PROXY_INNER`` is already a container-reachable value and therefore
    wins.  The old code only read that pair, while the normal benchmark proxy
    launcher exports ``BENCHMARK_CONTAINER_*`` and many invocations only have
    host-side ``HTTP_PROXY``.  The latter is rewritten only when proxy use was
    explicitly enabled, so an ambient developer proxy is not silently adopted.
    """
    source = source if source is not None else os.environ
    http = source.get("HTTP_PROXY_INNER") or source.get("BENCHMARK_CONTAINER_HTTP_PROXY")
    https = source.get("HTTPS_PROXY_INNER") or source.get("BENCHMARK_CONTAINER_HTTPS_PROXY")
    all_proxy = source.get("ALL_PROXY_INNER") or source.get("BENCHMARK_CONTAINER_ALL_PROXY")
    no_proxy = (
        source.get("NO_PROXY_INNER")
        or source.get("BENCHMARK_CONTAINER_NO_PROXY")
        or source.get("NO_PROXY")
        or source.get("no_proxy")
    )

    if not http and source.get("BENCHMARK_USE_PROXY") == "1":
        http = _rewrite_proxy_host(source.get("HTTP_PROXY", ""))
    if not https and source.get("BENCHMARK_USE_PROXY") == "1":
        https = _rewrite_proxy_host(source.get("HTTPS_PROXY") or http or "")
    if not all_proxy and source.get("BENCHMARK_USE_PROXY") == "1":
        all_proxy = _rewrite_proxy_host(source.get("ALL_PROXY", "") or http or "")

    if http and not https:
        https = http
    if http or https or all_proxy:
        no_proxy = no_proxy or _DEFAULT_CONTAINER_NO_PROXY

    resolved: dict[str, str] = {}
    for upper, lower, value in (
        ("HTTP_PROXY", "http_proxy", http),
        ("HTTPS_PROXY", "https_proxy", https),
        ("ALL_PROXY", "all_proxy", all_proxy),
    ):
        if value:
            resolved[upper] = value
            resolved[lower] = value
    if no_proxy:
        resolved["NO_PROXY"] = no_proxy
        resolved["no_proxy"] = no_proxy
    return resolved

def remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)

def start_container(task_id: str, workspace_path: str, extra_env: str = "",
                    tmp_path: str = "", lobster_env: list[str] | None = None) -> None:
    if not DOCKER_IMAGE:
        raise RuntimeError("DOCKER_IMAGE must be set before starting a WildClaw container")
    workspace = Path(workspace_path).expanduser()
    if not workspace.is_dir():
        raise RuntimeError(f"Workspace path does not exist or is not a directory: {workspace}")

    proxy_env = container_proxy_env()
    env_args = []
    # Explicitly clear image-baked proxy values when no runtime proxy was
    # selected. Some legacy WildClaw images carried a dead proxy in ENV.
    for key in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "NO_PROXY", "no_proxy",
    ):
        env_args += ["-e", f"{key}={proxy_env.get(key, '')}"]
    if BRAVE_API_KEY:
        env_args += ["-e", f"BRAVE_API_KEY={BRAVE_API_KEY}"]
    for line in extra_env.splitlines():
        key = line.strip()
        if not key or key.startswith("#"):
            continue
        value = os.environ.get(key, "")
        env_args += ["-e", f"{key}={value}"]
        masked = (value[:4] + "***") if value else "(empty)"
        logger.info("[%s] Injecting env var: %s=%s", task_id, key, masked)

    for key in (lobster_env or []):
        value = os.environ.get(key, "")
        if not value:
            logger.warning("[%s] Lobster env key %s not found in environment, skipping", task_id, key)
            continue
        env_args += ["-e", f"{key}={value}"]
        masked = value[:4] + "***"
        logger.info("[%s] Injecting lobster env: %s=%s", task_id, key, masked)
 
    cmd = [
        "docker", "run", "-d",
        "--name", task_id,
        *env_args,
        "-v", f"{workspace}:/app:ro",
        DOCKER_IMAGE,
        "/bin/bash", "-c", "tail -f /dev/null",
    ]
    logger.info("[%s] Starting container, mounting %s → /app (ro)", task_id, workspace)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Container startup failed:\n{r.stderr}")
    logger.info("[%s] Container ID: %s", task_id, r.stdout.strip()[:12])

    if tmp_path and os.path.exists(tmp_path):
        mkdir_cmd = ["docker", "exec", task_id, "mkdir", "-p", "/tmp_workspace/tmp"]
        subprocess.run(mkdir_cmd, capture_output=True)

        cp_cmd = ["docker", "cp", f"{tmp_path}/.", f"{task_id}:/tmp_workspace/tmp/"]
        
        logger.info("[%s] Copying temp files: %s → /tmp_workspace/tmp", task_id, tmp_path)
        cp_r = subprocess.run(cp_cmd, capture_output=True, text=True)
        
        if cp_r.returncode != 0:
            logger.error("[%s] File copy failed: %s", task_id, cp_r.stderr)
        else:
            logger.info("[%s] Temp file copy complete", task_id)

def setup_workspace(task_id: str, thinking: str | None = None) -> None:
    logger.info("[%s] Copying /app → %s", task_id, TMP_WORKSPACE)
    r = subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c",
         f"cp -r /app/. {TMP_WORKSPACE} && chmod -R u+w {TMP_WORKSPACE}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Workspace copy failed:\n{r.stderr}")

    if thinking is not None:
        logger.info("[%s] Setting thinkingDefault to %s", task_id, thinking)
        thinking_result = subprocess.run(
            ["docker", "exec", task_id,
             "openclaw", "config", "set", "agents.defaults.thinkingDefault", thinking],
            capture_output=True, text=True,
        )
        if thinking_result.returncode != 0:
            raise RuntimeError(
                f"Failed to set thinkingDefault to {thinking}:\n{thinking_result.stderr}"
            )

    # Symlink OpenClaw workspace → TMP_WORKSPACE so the image tool's
    # media-local-roots check allows reading files under /tmp_workspace.
    subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c",
         f"rm -rf /root/.openclaw/workspace && ln -s {TMP_WORKSPACE} /root/.openclaw/workspace"],
        capture_output=True, text=True,
    )

def setup_skills(
    task_id: str,
    skills: str,
    skills_path: str,
    container_skills_root: str = "/root/skills",
) -> None:
    container_skills_root = container_skills_root.rstrip("/")
    subprocess.run(
        ["docker", "exec", task_id, "mkdir", "-p", container_skills_root],
        capture_output=True,
        text=True,
    )
    seen_dest_names: set[str] = set()
    for line in skills.splitlines():
        line = line.strip()
        if not line:
            continue
        src_rel = line.replace("\\", "/").strip("/")
        dest_name = PurePosixPath(src_rel).name
        if not dest_name:
            logger.warning("[%s] Invalid skill path %r, skipping", task_id, line)
            continue
        if dest_name in seen_dest_names:
            logger.warning(
                "[%s] Duplicate flattened skill target %s from %s, skipping",
                task_id,
                dest_name,
                line,
            )
            continue
        seen_dest_names.add(dest_name)
        subprocess.run(
            ["docker", "exec", task_id,
             "mkdir", "-p", f"{container_skills_root}/{dest_name}"],
            capture_output=True, text=True,
        )
        r = subprocess.run(
            ["docker", "cp",
             f"{skills_path}/{src_rel}/.", f"{task_id}:{container_skills_root}/{dest_name}/"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning(
                "[%s] Failed to copy skill %s to %s/%s: %s",
                task_id,
                line,
                container_skills_root,
                dest_name,
                r.stderr.strip(),
            )


def inject_openclaw_models(task_id: str, models_config: dict) -> None:
    """Inject custom models into ~/.openclaw/openclaw.json."""
    container_tmp_path = "/tmp/openclaw_models.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp_file:
        json.dump(models_config, tmp_file, indent=2)
        tmp_file_path = tmp_file.name

    try:
        cp_r = subprocess.run(
            ["docker", "cp", tmp_file_path, f"{task_id}:{container_tmp_path}"],
            capture_output=True, text=True,
        )
        if cp_r.returncode != 0:
            raise RuntimeError(f"Failed to copy models config into container:\n{cp_r.stderr}")

        inject_cmd = f"""python3 - <<'PY'
import json
import pathlib

config_path = pathlib.Path('/root/.openclaw/openclaw.json')
models_path = pathlib.Path('{container_tmp_path}')

config = json.loads(config_path.read_text()) if config_path.exists() else {{}}
models = json.loads(models_path.read_text())
config['models'] = models

config_path.write_text(json.dumps(config, indent=2))
PY"""
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", inject_cmd],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Failed to inject models config:\n{r.stderr}")
    finally:
        Path(tmp_file_path).unlink(missing_ok=True)

    logger.info("[%s] Injected custom models config", task_id)


def run_warmup(
    task_id: str,
    warmup: str,
    *,
    detach_background: bool = False,
) -> None:
    """Execute warmup bash commands line by line inside the container (skip blank lines and comments)."""
    if not warmup.strip():
        return
    commands = [
        line.strip()
        for line in warmup.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not commands:
        return

    retry_delay = 10.0
    max_retries = 0
    retry_desc = "unlimited" if max_retries <= 0 else str(max_retries)

    logger.info(
        "[%s] Running warmup (%d commands, retries=%s, retry_delay=%.1fs)",
        task_id,
        len(commands),
        retry_desc,
        retry_delay,
    )
    for idx, cmd in enumerate(commands, start=1):
        logger.info("[%s] warmup: %s", task_id, cmd)
        stripped_cmd = cmd.rstrip()

        attempts = 0
        while True:
            attempts += 1
            if detach_background and stripped_cmd.endswith("&"):
                background_cmd = stripped_cmd[:-1].strip()
                log_path = f"/tmp/wildclaw_warmup_{idx}.log"
                # `-c`, not `-lc`, in both shells here and everywhere else in
                # the harness.  These two lines were the only place a warmup
                # command's PATH depended on whether it ended in `&`: the
                # backgrounded branch used a login shell and the branch below
                # does not, so the same warmup line saw two different PATHs
                # depending on a trailing character.  In the hermes image that
                # was a real difference -- a login shell there sourced uv's
                # env script and inserted /root/.local/bin, which carries uv's
                # own package-less python3.12.  Every image's ENV PATH is now
                # authoritative and complete, so a login shell adds nothing.
                wrapped = (
                    f"cd {TMP_WORKSPACE} && "
                    f"nohup /bin/bash -c {shlex.quote(background_cmd)} "
                    f"> {shlex.quote(log_path)} 2>&1 < /dev/null &"
                )
                r = subprocess.run(
                    ["docker", "exec", task_id, "/bin/bash", "-c", wrapped],
                    capture_output=True,
                    text=True,
                )
            else:
                r = subprocess.run(
                    ["docker", "exec", task_id, "/bin/bash", "-c", cmd],
                    capture_output=True, text=True,
                )

            if r.returncode != 0:
                stderr = (r.stderr or "").strip()
                stdout = (r.stdout or "").strip()
                if max_retries > 0 and attempts > max_retries:
                    raise RuntimeError(
                        f"Warmup command failed after {attempts} attempts: {cmd!r}\n"
                        f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
                    )
                logger.warning(
                    "[%s] Warmup command failed (command %d/%d, attempt %d, rc=%d); retrying in %.1fs: %s\nstdout: %s\nstderr: %s",
                    task_id,
                    idx,
                    len(commands),
                    attempts,
                    r.returncode,
                    retry_delay,
                    cmd,
                    stdout[-1000:],
                    stderr[-1000:],
                )
                time.sleep(max(0.0, retry_delay))
                continue

            if attempts > 1:
                logger.info(
                    "[%s] Warmup command succeeded after %d attempts: %s",
                    task_id,
                    attempts,
                    cmd,
                )
            break


def run_background(
    task_id: str, bash_cmd: str, log_path: Path, append: bool = False
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a" if append else "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["docker", "exec", task_id, "/bin/bash", "-c",
         f"cd {TMP_WORKSPACE} && {bash_cmd}"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    proc._log_file = log_file
    logger.info("[%s] Started process PID=%s → %s", task_id, proc.pid, log_path)
    return proc


def close_proc_log(proc: subprocess.Popen) -> None:
    """Close the log file handle created by run_background."""
    log_file = getattr(proc, "_log_file", None)
    if log_file and not log_file.closed:
        log_file.close()


def collect_output_from_container(
    task_id: str,
    output_dir: Path,
    *,
    include_workspace_changes: bool = False,
) -> None:
    """Collect task output files from the container to output_dir/task_output/.

    Collection strategy:
      1. All files under /tmp/openclaw/ (agent session logs, etc.)
      2. Task output files under /tmp_workspace/results/
      3. Optionally, the full /tmp_workspace/ tree for runners that may write
         deliverables outside results/
    """
    task_output_dir = output_dir / "task_output"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    _copy_dir_from_container(task_id, "/tmp/openclaw/.", str(task_output_dir))

    workspace_out = task_output_dir / "workspace"
    workspace_out.mkdir(parents=True, exist_ok=True)

    if include_workspace_changes:
        ok = _copy_dir_from_container(task_id, f"{TMP_WORKSPACE}/.", str(workspace_out))
        if not ok:
            logger.warning("[%s] workspace directory does not exist or is empty", task_id)
        return

    results_out = workspace_out / "results"
    results_out.mkdir(parents=True, exist_ok=True)
    ok = _copy_dir_from_container(
        task_id, f"{TMP_WORKSPACE}/results/.", str(results_out),
    )
    if not ok:
        logger.warning("[%s] results/ directory does not exist or is empty", task_id)


def snapshot_workspace_state(task_id: str) -> None:
    snapshot_cmd = "python3 - <<'PY'\n" + "\n".join([
        "import json",
        "from pathlib import Path",
        f"root = Path({TMP_WORKSPACE!r})",
        f"snapshot_path = Path({WORKSPACE_BASELINE_PATH!r})",
        "files = {}",
        "if root.exists():",
        "    for path in root.rglob('*'):",
        "        if not (path.is_file() or path.is_symlink()):",
        "            continue",
        "        try:",
        "            stat = path.lstat()",
        "        except OSError:",
        "            continue",
        "        rel = path.relative_to(root).as_posix()",
        "        files[rel] = {",
        "            'size': stat.st_size,",
        "            'mtime_ns': stat.st_mtime_ns,",
        "            'is_symlink': path.is_symlink(),",
        "        }",
        "snapshot_path.write_text(json.dumps(files, sort_keys=True), encoding='utf-8')",
        "PY",
    ])
    r = subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c", snapshot_cmd],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Workspace baseline snapshot failed:\n{r.stderr}")
    logger.info("[%s] Workspace baseline saved at %s", task_id, WORKSPACE_BASELINE_PATH)


def _copy_changed_workspace_outputs_from_container(task_id: str, dest: Path) -> None:
    list_cmd = "python3 - <<'PY'\n" + "\n".join([
        "import json",
        "from pathlib import Path",
        f"root = Path({TMP_WORKSPACE!r})",
        f"snapshot_path = Path({WORKSPACE_BASELINE_PATH!r})",
        "if not snapshot_path.exists():",
        "    print(json.dumps([]))",
        "    raise SystemExit(0)",
        "before = json.loads(snapshot_path.read_text(encoding='utf-8'))",
        "excluded_prefixes = (",
        "    'results/', 'gt/', 'tmp/', '.git/',",
        "    'node_modules/', '.venv/', 'venv/', '__pycache__/', '.cache/',",
        ")",
        "excluded_names = {'results', 'gt', 'tmp', '.git', 'node_modules', '.venv', 'venv', '__pycache__', '.cache'}",
        "changed = []",
        "if root.exists():",
        "    for path in root.rglob('*'):",
        "        if not (path.is_file() or path.is_symlink()):",
        "            continue",
        "        rel = path.relative_to(root).as_posix()",
        "        if rel in excluded_names or rel.startswith(excluded_prefixes):",
        "            continue",
        "        try:",
        "            stat = path.lstat()",
        "        except OSError:",
        "            continue",
        "        current = {",
        "            'size': stat.st_size,",
        "            'mtime_ns': stat.st_mtime_ns,",
        "            'is_symlink': path.is_symlink(),",
        "        }",
        "        if before.get(rel) != current:",
        "            changed.append(rel)",
        "print(json.dumps(sorted(changed)))",
        "PY",
    ])
    r = subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c", list_cmd],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.warning("[%s] Failed to list changed workspace outputs: %s", task_id, r.stderr.strip())
        return

    try:
        changed_paths = json.loads(r.stdout.strip() or "[]")
    except json.JSONDecodeError:
        logger.warning("[%s] Failed to parse changed workspace output list: %s", task_id, r.stdout[:200])
        return

    for rel_path in changed_paths:
        if not isinstance(rel_path, str) or rel_path.startswith("/") or ".." in Path(rel_path).parts:
            continue
        dest_path = dest / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_file_from_container(task_id, f"{TMP_WORKSPACE}/{rel_path}", dest_path)


def inject_lobster_workspace(task_id: str, workspace_path: str) -> None:
    """Copy the entire lobster workspace into /root/ (the OpenClaw workspace in the image).

    This brings in everything: SOUL.md, USER.md, MEMORY.md, memory/, skills/, etc.
    """
    r = subprocess.run(
        ["docker", "cp", f"{workspace_path}/.", f"{task_id}:/root/"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.error("[%s] Lobster workspace copy failed: %s", task_id, r.stderr)
    else:
        logger.info("[%s] Lobster workspace copied: %s → /root/", task_id, workspace_path)


def _copy_dir_from_container(task_id: str, src: str, dest: str) -> bool:
    r = subprocess.run(
        ["docker", "cp", f"{task_id}:{src}", dest],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        logger.info("[%s] Collected container directory %s → %s", task_id, src, dest)
        return True
    return False


def _copy_file_from_container(task_id: str, src: str, dest: Path) -> bool:
    r = subprocess.run(
        ["docker", "cp", f"{task_id}:{src}", str(dest)],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        logger.info("[%s] Collected container file %s → %s", task_id, src, dest)
        return True
    logger.warning("[%s] Container file copy failed (%s): %s", task_id, src, r.stderr.strip())
    return False
