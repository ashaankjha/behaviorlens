#!/bin/bash
# Stop the app started by run_app.sh. Uses the pid file; never `pkill -f "streamlit run"`
# (the pattern matches the calling shell itself; see docs/LEARNINGS.md M1).
set -uo pipefail
DATA_DIR="${1:?data dir required}"
if [ -f "$DATA_DIR/streamlit.pid" ]; then
  kill "$(cat "$DATA_DIR/streamlit.pid")" 2>/dev/null || true
  rm -f "$DATA_DIR/streamlit.pid"
fi
echo stopped
