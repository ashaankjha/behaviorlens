---
name: update-learnings
description: Record a new lesson in docs/LEARNINGS.md, the project's living learnings log. Use whenever you hit a surprising library or environment behavior, make and fix a mistake, change a design decision, or finish a task that taught something non-obvious. Also use at the end of any substantial work session in this repo.
---

# Maintain docs/LEARNINGS.md

`docs/LEARNINGS.md` is the project's memory across agents and sessions. Keep it accurate, specific and short.

## When to write

- Something failed in a way that wasn't obvious from the error (environment, library version, browser, OS libs).
- You made a mistake and fixed it, including bugs found by tests, review or real data.
- A design decision was made or reversed.
- A verification recipe worked, or turned out to be misleading.

## How

1. Read the current file. Don't duplicate; **edit** an existing line if it covers the topic.
2. Put the entry in the right section:
   - §2 *Environment and tooling*: a table row with the topic and a concrete fact (versions, exact error strings, exact fix commands).
   - §3 *Mistakes and their fixes*: the next number `M<n>` with **Mistake / Symptom / Fix and lesson**. State the generalisable lesson, not just the patch.
   - §4 *Design decisions*: the decision and *why*, plus the alternative you rejected when it matters.
   - §5 *Verification recipes* and §6 *Open questions*: as needed. Remove open questions once they're answered.
3. Bump `Last updated:` to today's date.
4. If the lesson implies a guardrail, also add it where it will be enforced: a test, a note in the `CLAUDE.md` conventions, or a hook in `.claude/hooks/`. Reference the M-number there.
5. Commit the learnings change together with the code change it explains.

## Style

- Exact over vague: "`libEGL.so.1: cannot open shared object file` → `apt-get install -y libegl1 libgles2`", not "install graphics libs".
- No secrets, tokens, personal data or footage descriptions of real people.
