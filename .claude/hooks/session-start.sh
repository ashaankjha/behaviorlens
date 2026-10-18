#!/bin/bash
# SessionStart hook: make tests, lint and the app runnable in Claude Code cloud sessions.
# Idempotent and non-interactive; synchronous so deps exist before the first command.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# MediaPipe loads libEGL even for CPU inference (docs/LEARNINGS.md §2).
if ! ldconfig -p | grep -q 'libEGL.so.1' || ! ldconfig -p | grep -q 'libGLESv2.so.2'; then
  if command -v apt-get >/dev/null; then
    (apt-get install -y -q libegl1 libgles2 >/dev/null 2>&1 \
      || (apt-get update -q >/dev/null 2>&1 && apt-get install -y -q libegl1 libgles2 >/dev/null 2>&1)) \
      || echo "warning: could not install libegl1/libgles2; real pose inference will fail" >&2
  fi
fi

PY=python3.11
command -v "$PY" >/dev/null || PY=python3
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
.venv/bin/pip install -q --disable-pip-version-check -r requirements.txt ruff playwright

# Pose model: best effort (tests that need it skip when it is missing).
.venv/bin/python scripts/download_model.py >/dev/null 2>&1 \
  || echo "warning: pose model download failed; run scripts/download_model.py later" >&2

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$PWD/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
echo "BehaviorLens environment ready (.venv, requirements, ruff, playwright, pose model)"
