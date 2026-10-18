#!/usr/bin/env python3
"""PreToolUse hook (Bash): block `pkill -f PATTERN` / `pgrep -f PATTERN` whose pattern
matches the calling shell's own command line (docs/LEARNINGS.md M1).

The tool runs commands via `bash -c '<whole command>'`, so a plain pattern like
"streamlit run" also matches that bash process and kills the tool's own shell (exit 144).
A bracket class such as '[s]treamlit run' matches the target but not itself.

Only real invocations are checked: heredoc bodies and words that are not in command
position (e.g. `echo pkill -f x`) are ignored (docs/LEARNINGS.md M13).
"""

import json
import re
import shlex
import sys

HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\s*(?:\n|$)", re.S)
PREFIXES = {"sudo", "exec", "command", "nohup", "time", "xargs", "env"}


def strip_heredocs(command: str) -> str:
    return HEREDOC.sub("\n", command)


def unsafe_patterns(command: str) -> list[str]:
    bad = []
    for segment in re.split(r"[;&|\n]+|\$\(|`", strip_heredocs(command)):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        # Skip leading prefixes such as `sudo` to find the command word.
        i = 0
        while i < len(tokens) and tokens[i] in PREFIXES:
            i += 1
        if i >= len(tokens) or tokens[i].rsplit("/", 1)[-1] not in ("pkill", "pgrep"):
            continue
        args = tokens[i + 1:]
        full = any(a == "--full" or (a.startswith("-") and not a.startswith("--") and "f" in a[1:])
                   for a in args)
        positional = [a for a in args if not a.startswith("-")]
        if full and positional and "[" not in positional[-1]:
            bad.append(positional[-1].rstrip(")"))
    return bad


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    command = (data.get("tool_input") or {}).get("command", "")
    bad = unsafe_patterns(command)
    if not bad:
        return 0
    pat = bad[0]
    safe = f"[{pat[0]}]{pat[1:]}" if pat else pat
    print(
        f"Blocked: `-f '{pat}'` also matches this tool's own shell command line and would kill it "
        f"(docs/LEARNINGS.md M1). Use a self-excluding pattern such as -f '{safe}', or kill by PID "
        "(e.g. .claude/skills/e2e-verify-streamlit/scripts/stop_app.sh).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
