from __future__ import annotations

import io
from pathlib import Path

from src.utils import docker_utils


def test_copy_file_uses_docker_exec_cat_not_docker_cp(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "out" / "chat.jsonl"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        stdout = kwargs.get("stdout")
        if stdout is not None and hasattr(stdout, "write"):
            stdout.write(b"hello\n")
        return type("R", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr(docker_utils.subprocess, "run", fake_run)
    assert docker_utils._copy_file_from_container("ctr", "/root/chat.jsonl", dest)
    assert dest.read_bytes() == b"hello\n"
    assert calls[0][:3] == ["docker", "exec", "ctr"]
    assert "cp" not in calls[0]


def test_copy_dir_pipes_tar_instead_of_docker_cp(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "task_output"
    popen_cmds: list[list[str]] = []

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None):
            popen_cmds.append(list(cmd))
            self.stdout = io.BytesIO(b"")
            self.returncode = 0

        def communicate(self):
            return b"", b""

    def fake_run(cmd, **kwargs):
        assert cmd[:1] == ["tar"]
        return type("R", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr(docker_utils.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(docker_utils.subprocess, "run", fake_run)
    assert docker_utils._copy_dir_from_container("ctr", "/tmp/openclaw/.", str(dest))
    assert popen_cmds[0][:4] == ["docker", "exec", "ctr", "tar"]
    assert dest.is_dir()
