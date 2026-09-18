from src.utils.docker_utils import warmup_presence_check


def test_skips_global_npm_when_binary_exists() -> None:
    check = warmup_presence_check("npm install -g agent-browser")
    assert check is not None
    assert "command -v agent-browser" in check


def test_skips_ffmpeg_apt_line() -> None:
    check = warmup_presence_check("apt-get update && apt-get install -y ffmpeg")
    assert check is not None
    assert "command -v ffmpeg" in check


def test_skips_pip_openai_pillow() -> None:
    check = warmup_presence_check("pip install openai Pillow")
    assert check is not None
    assert "import openai, PIL" in check


def test_skips_playwright_chromium_if_cache_present() -> None:
    check = warmup_presence_check("python3 -m playwright install chromium")
    assert check is not None
    assert "ms-playwright" in check


def test_does_not_skip_mock_servers_or_cleanup() -> None:
    assert warmup_presence_check(
        "export SLACK_FIXTURES=/tmp_workspace/tmp/messages.json && python3 /tmp_workspace/mock_services/slack/server.py &"
    ) is None
    assert warmup_presence_check("rm -f -r /tmp_workspace/tmp") is None
    assert warmup_presence_check("sleep 2") is None
    assert warmup_presence_check('~/miniconda3/envs/eval/bin/python -c "import numpy, cv2"') is None
