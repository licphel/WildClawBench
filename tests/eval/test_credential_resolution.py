"""Tests for eval/credential_resolution.py.

Covers the bug this module fixes: calling ``python eval/run_batch.py``
directly (bypassing every ``run_*.sh`` launcher, which is what normally
calls ``config_lib.sh``'s ``resolve_wildclaw_gateway``/``resolve_judge_env``)
used to leave ``JUDGE_MODEL``/``OPENROUTER_API_KEY``/``OPENROUTER_BASE_URL``
empty, silently corrupting every LLM-judge grade and usage probe in the run.

Run from the WildClawBench root:
    python3 -m pytest tests/eval/test_credential_resolution.py -q
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_WCB_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _WCB_ROOT / "eval"
sys.path.insert(0, str(_EVAL_DIR))

import credential_resolution as cr  # noqa: E402


def test_default_judge_model_parsed_from_config_lib(tmp_path, monkeypatch):
    config_lib = tmp_path / "config_lib.sh"
    config_lib.write_text(
        'blah blah\n_WILDCLAW_DEFAULT_JUDGE_MODEL="gpt-9.9-test"\nmore blah\n'
    )
    assert cr._default_judge_model_from_config_lib(config_lib) == "gpt-9.9-test"


def test_default_judge_model_falls_back_when_config_lib_missing(tmp_path):
    missing = tmp_path / "does-not-exist.sh"
    assert (
        cr._default_judge_model_from_config_lib(missing)
        == cr._FALLBACK_DEFAULT_JUDGE_MODEL
    )


def _no_dotenv_env(tmp_path, monkeypatch, config_lib_text='_WILDCLAW_DEFAULT_JUDGE_MODEL="gpt-5.6-terra"\n'):
    """Point the module at a tmp_path-scoped, deterministic .env/config_lib.sh
    pair instead of the real outer-repo files, so tests don't depend on --
    or risk touching -- the real shared .env.
    """
    env_file = tmp_path / ".env"
    config_lib = tmp_path / "config_lib.sh"
    config_lib.write_text(config_lib_text)
    monkeypatch.setattr(cr, "_ENV_FILE", env_file)
    monkeypatch.setattr(cr, "_CONFIG_LIB_SH", config_lib)
    return env_file


def test_independent_judge_when_model_differs_and_openai_key_present(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "JUDGE_MODEL": "gpt-5.5",
        "OPENAI_API_KEY": "sk-openai",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["JUDGE_MODEL"] == "gpt-5.5"
    assert env["OPENROUTER_API_KEY"] == "sk-openai"
    assert env["OPENROUTER_BASE_URL"] == "https://api.openai.com/v1"


def test_default_judge_matches_runner_and_self_grades(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "OPENAI_API_KEY": "sk-openai",
        "GATEWAY_TOKEN": "agent-key",
        "GATEWAY_V1": "http://agent-gateway/v1",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["JUDGE_MODEL"] == "gpt-5.6-terra"
    assert env["OPENROUTER_API_KEY"] == "agent-key"
    assert env["OPENROUTER_BASE_URL"] == "http://agent-gateway/v1"


def test_openai_base_url_takes_precedence_over_openai_api_base(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "JUDGE_MODEL": "gpt-5.5",
        "OPENAI_API_KEY": "sk-openai",
        "OPENAI_BASE_URL": "https://openai.example/v1",
        "OPENAI_API_BASE": "https://ignored.example/v1",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["OPENROUTER_BASE_URL"] == "https://openai.example/v1"


def test_openai_api_base_used_when_openai_base_url_unset(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "JUDGE_MODEL": "gpt-5.5",
        "OPENAI_API_KEY": "sk-openai",
        "OPENAI_API_BASE": "https://openai-base.example/v1",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["OPENROUTER_BASE_URL"] == "https://openai-base.example/v1"


def test_explicit_self_grade_uses_agent_gateway(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "JUDGE_MODEL": "gpt-5.6-terra",
        "OPENAI_API_KEY": "sk-openai",
        "GATEWAY_TOKEN": "agent-key",
        "GATEWAY_V1": "http://agent-gateway/v1",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["JUDGE_MODEL"] == "gpt-5.6-terra"
    assert env["OPENROUTER_API_KEY"] == "agent-key"
    assert env["OPENROUTER_BASE_URL"] == "http://agent-gateway/v1"


def test_missing_openai_key_falls_back_to_agent_gateway(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "GATEWAY_TOKEN": "agent-key",
        "GATEWAY_V1": "http://agent-gateway/v1",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["JUDGE_MODEL"] == "gpt-5.6-terra"
    assert env["OPENROUTER_API_KEY"] == "agent-key"
    assert env["OPENROUTER_BASE_URL"] == "http://agent-gateway/v1"


# --- (a) raise loudly when nothing at all can be resolved ------------------


def test_raises_when_nothing_resolvable(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)  # .env file does not exist
    env: dict[str, str] = {}  # no OPENROUTER_*, no OPENAI_*, no GATEWAY_*
    with pytest.raises(cr.WildClawCredentialError) as exc_info:
        cr.ensure_wildclaw_judge_env(env)
    message = str(exc_info.value)
    assert "run_batch.py must be invoked through one of" in message
    assert "run_pylm" in message or "run_{pylm" in message


def test_raises_when_env_file_exists_but_has_no_openai_key(tmp_path, monkeypatch):
    env_file = _no_dotenv_env(tmp_path, monkeypatch)
    env_file.write_text("SOME_OTHER_VAR=1\n")
    env: dict[str, str] = {"RUNNER_MODEL": "gpt-5.6-terra"}
    with pytest.raises(cr.WildClawCredentialError):
        cr.ensure_wildclaw_judge_env(env)


# --- (b) resolve via OPENAI_API_KEY/OPENAI_API_BASE read straight out of .env


def test_reads_openai_credentials_directly_from_dotenv_file(tmp_path, monkeypatch):
    env_file = _no_dotenv_env(tmp_path, monkeypatch)
    env_file.write_text(
        "# a comment\n"
        "OPENAI_API_KEY=sk-from-dotenv\n"
        "OPENAI_API_BASE=https://openai-dotenv.example/v1\n"
        "SOME_UNRELATED_VAR=ignored\n"
    )
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "JUDGE_MODEL": "gpt-5.5",
    }  # OPENAI_API_KEY NOT in env itself
    cr.ensure_wildclaw_judge_env(env)
    assert env["JUDGE_MODEL"] == "gpt-5.5"
    assert env["OPENROUTER_API_KEY"] == "sk-from-dotenv"
    assert env["OPENROUTER_BASE_URL"] == "https://openai-dotenv.example/v1"


def test_dotenv_export_prefix_and_quotes_are_handled(tmp_path, monkeypatch):
    env_file = _no_dotenv_env(tmp_path, monkeypatch)
    env_file.write_text(
        'export OPENAI_API_KEY="sk-quoted"\n'
        "export OPENAI_BASE_URL='https://quoted.example/v1'\n"
    )
    env = {"RUNNER_MODEL": "gpt-5.6-terra", "JUDGE_MODEL": "gpt-5.5"}
    cr.ensure_wildclaw_judge_env(env)
    assert env["OPENROUTER_API_KEY"] == "sk-quoted"
    assert env["OPENROUTER_BASE_URL"] == "https://quoted.example/v1"


# --- (c) never overwrite already-present values -----------------------------


def test_does_not_overwrite_already_present_openrouter_vars(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "OPENAI_API_KEY": "sk-should-be-ignored",
        "OPENROUTER_API_KEY": "already-set-key",
        "OPENROUTER_BASE_URL": "http://already-set/v1",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["OPENROUTER_API_KEY"] == "already-set-key"
    assert env["OPENROUTER_BASE_URL"] == "http://already-set/v1"
    # JUDGE_MODEL was missing, so it gets backfilled with the default.
    assert env["JUDGE_MODEL"] == "gpt-5.6-terra"


def test_does_not_overwrite_already_present_judge_model(tmp_path, monkeypatch):
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "JUDGE_MODEL": "gpt-4.1-custom",
        "OPENAI_API_KEY": "sk-openai",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["JUDGE_MODEL"] == "gpt-4.1-custom"
    # gpt-4.1-custom != RUNNER_MODEL, and OPENAI_API_KEY is present, so this
    # still routes to the real OpenAI API, using the preserved JUDGE_MODEL.
    assert env["OPENROUTER_API_KEY"] == "sk-openai"


def test_does_not_overwrite_partial_openrouter_key_but_fills_gap(tmp_path, monkeypatch):
    """Only OPENROUTER_API_KEY was pre-set (no OPENROUTER_BASE_URL): the key
    stays untouched, but the resolver still needs to fill the base url, so
    this is NOT treated as "already resolved".
    """
    _no_dotenv_env(tmp_path, monkeypatch)
    env = {
        "RUNNER_MODEL": "gpt-5.6-terra",
        "JUDGE_MODEL": "gpt-5.5",
        "OPENAI_API_KEY": "sk-openai",
        "OPENROUTER_API_KEY": "pre-set-key",
    }
    cr.ensure_wildclaw_judge_env(env)
    assert env["OPENROUTER_API_KEY"] == "pre-set-key"
    assert env["OPENROUTER_BASE_URL"] == "https://api.openai.com/v1"


# --- process-level: actually invoked directly, with no way to resolve ------


def test_subprocess_raises_when_invoked_directly_with_no_credentials(tmp_path):
    """A real end-to-end check that running credential_resolution (as
    run_batch.py does at import time) fails loudly -- not just that the
    function raises when called in-process. Runs against an isolated copy
    of the module (its own tmp_path, with no .env alongside it) rather than
    the real checkout, so this cannot touch -- or depend on -- the real
    shared outer-repo .env other sessions may be using concurrently.
    """
    fake_outer_root = tmp_path / "outer"
    fake_wcb_eval = fake_outer_root / "benchmarks" / "WildClawBench" / "eval"
    fake_wcb_eval.mkdir(parents=True)
    shutil.copy(_EVAL_DIR / "credential_resolution.py", fake_wcb_eval / "credential_resolution.py")
    # Deliberately no .env and no config_lib.sh anywhere under fake_outer_root.

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import credential_resolution as cr; cr.ensure_wildclaw_judge_env({})",
        ],
        cwd=str(fake_wcb_eval),
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "WildClawCredentialError" in proc.stderr
    assert "run_batch.py must be invoked through one of" in proc.stderr


@pytest.mark.skipif(
    not cr._ENV_FILE.exists(),
    reason="no outer-repo .env present in this checkout to integration-test against",
)
def test_run_batch_help_succeeds_via_real_outer_env():
    """Integration smoke test against the real run_batch.py and the real
    outer-repo .env on this host: with every judge/gateway var cleared,
    ``python eval/run_batch.py --help`` must still succeed (exit 0) because
    ensure_wildclaw_judge_env() (called at module import time, before
    argparse ever runs) can resolve a judge credential from the real
    OPENAI_API_KEY in .env -- mirroring config_lib.sh's resolve_judge_env
    when only OPENAI_API_KEY/OPENAI_API_BASE are available. This is the same
    check --help exercises for any invocation, including a real
    ``--category all`` run: the credential check runs unconditionally before
    any argument is even parsed.
    """
    stripped_env = {"PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, str(_EVAL_DIR / "run_batch.py"), "--help"],
        cwd=str(_WCB_ROOT),
        env=stripped_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
