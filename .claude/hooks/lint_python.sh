#!/bin/bash
# PostToolUse hook (Edit|Write|MultiEdit): ruff-check the edited Python file and feed
# problems back to Claude (exit 2 -> stderr is shown to the model). Silent when clean.
set -uo pipefail
ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
FILE=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("tool_input") or {}).get("file_path",""))' 2>/dev/null)
case "$FILE" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$FILE" ] || exit 0
# Only lint files inside this project (scratch files elsewhere are not our concern).
case "$(realpath "$FILE")" in
  "$(realpath "$ROOT")"/*) ;;
  *) exit 0 ;;
esac
RUFF="$ROOT/.venv/bin/ruff"
[ -x "$RUFF" ] || RUFF=$(command -v ruff || true)
[ -n "$RUFF" ] || exit 0
OUT=$(cd "$ROOT" && "$RUFF" check --quiet "$FILE" 2>&1)
if [ -n "$OUT" ]; then
  echo "ruff found issues in $FILE (fix before moving on):" >&2
  echo "$OUT" | head -40 >&2
  exit 2
fi
exit 0
