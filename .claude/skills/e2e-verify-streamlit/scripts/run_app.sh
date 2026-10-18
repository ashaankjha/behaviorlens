#!/bin/bash
# Start the BehaviorLens Streamlit app in the background on a throwaway data dir.
# Usage: run_app.sh <data-dir> [port]   -> prints the URL once /_stcore/health is ok.
set -euo pipefail
DATA_DIR="${1:?data dir required}"
PORT="${2:-8501}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
mkdir -p "$DATA_DIR"
cd "$ROOT"
BEHAVIORLENS_DATA_DIR="$DATA_DIR" nohup .venv/bin/streamlit run app.py --server.port "$PORT" \
  --server.headless true > "$DATA_DIR/streamlit.log" 2>&1 &
echo $! > "$DATA_DIR/streamlit.pid"
for _ in $(seq 1 60); do
  if curl -fs "http://localhost:$PORT/_stcore/health" >/dev/null; then
    echo "http://localhost:$PORT"; exit 0
  fi
  sleep 1
done
echo "app did not become healthy; see $DATA_DIR/streamlit.log" >&2
exit 1
