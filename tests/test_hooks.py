"""Tests for the project's Claude Code hook scripts in .claude/hooks/."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".claude" / "hooks"


def run_guard(command: str) -> int:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return subprocess.run([sys.executable, str(HOOKS / "guard_pkill.py")], input=payload,
                          capture_output=True, text=True).returncode


@pytest.mark.parametrize("command", [
    'pkill -f "streamlit run"',
    "sleep 1; pkill -f streamlit && echo hi",
    "kill $(pgrep -f '.venv/bin/streamlit')",
    "pkill --full 'streamlit run'",
    "sudo pkill -9 -f uvicorn",
    "/usr/bin/pkill -f 'python app.py'",
])
def test_guard_blocks_self_matching_patterns(command):
    assert run_guard(command) == 2


@pytest.mark.parametrize("command", [
    "pkill -f '[s]treamlit run'",
    "kill $(pgrep -f '[.]venv/bin/streamlit')",
    "pkill streamlit",
    "pgrep -af '[b]in/streamlit'",
    "echo pkill -f is dangerous",
    "cat > notes.sh <<'EOF'\npkill -f streamlit\nEOF\nls",
    "ls -la",
])
def test_guard_allows_safe_commands(command):
    assert run_guard(command) == 0


def test_guard_ignores_malformed_input():
    r = subprocess.run([sys.executable, str(HOOKS / "guard_pkill.py")], input="not json",
                       capture_output=True, text=True)
    assert r.returncode == 0


def run_lint(file_path: str, project_dir: Path) -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_name": "Write", "tool_input": {"file_path": file_path}})
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir),
           "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    return subprocess.run(["bash", str(HOOKS / "lint_python.sh")], input=payload, capture_output=True,
                          text=True, env=env)


def test_lint_hook_reports_issues_and_passes_clean_files(tmp_path):
    if shutil.which("ruff", path=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}") is None:
        pytest.skip("ruff not installed")
    bad = tmp_path / "bad.py"
    bad.write_text("import os\n")
    good = tmp_path / "good.py"
    good.write_text("x = 1\n")
    r = run_lint(str(bad), tmp_path)
    assert r.returncode == 2 and "F401" in r.stderr
    assert run_lint(str(good), tmp_path).returncode == 0
    assert run_lint(str(tmp_path / "notes.md"), tmp_path).returncode == 0
    # files outside the project are ignored
    assert run_lint(str(bad), tmp_path / "elsewhere").returncode == 0
