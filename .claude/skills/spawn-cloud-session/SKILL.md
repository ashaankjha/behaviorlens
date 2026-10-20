---
name: spawn-cloud-session
description: Delegate independent, parallelizable BehaviorLens work (audits, experiments, model comparisons, long validations, isolated features) to new Claude Code cloud sessions on their own branches, then monitor, review and merge their results. Use when a task can proceed without this session's context or would block it for a long time.
---

# Spawn and integrate cloud sessions

The project owner wants parallel cloud sessions spawned **whenever necessary**. Use judgement: spawn for work that is independent, long-running or a different concern. Don't spawn for small edits you can make inline, for work that needs this conversation's live context, or for anything that must land on `main` without review.

## When to spawn

- Independent verification (a clean-room audit of the README, tests and claims).
- Experiments that answer an open question in `docs/VALIDATION.md` or `docs/LEARNINGS.md` §6.
- Long compute (pose inference over many clips) that would otherwise block the main thread.
- Self-contained features with a clear acceptance test, such as person tracking or camera-motion detection.

## How

1. Use the `create_session` tool (claude-code-remote MCP) with:
   - `source_url: https://github.com/ashaankjha/behaviorlens` and `source_revision: main`
   - `outcome_branch`: a unique branch such as `audit/...`, `experiment/...` or `feature/...`. **Never `main`.**
   - `permission_mode: auto`, never `plan`, because nobody is watching the child.
   - `tags: ["behaviorlens", "parent:<this session id>"]`
2. Write the prompt as a **standalone brief**, because the child has none of this conversation:
   - which docs to read first (`CLAUDE.md`, `docs/LEARNINGS.md`, the relevant skill);
   - setup commands and the exact question or task;
   - the method, plus what *not* to change (for example "don't retune thresholds");
   - the deliverable file and section;
   - constraints: never commit `data/`, weights or footage of people; push only its branch; no PR; log lessons per `update-learnings`;
   - **collision avoidance when several children run at once**:
     - Reserve a distinct LEARNINGS M-number range per child (for example M26–M29, M30–M33).
     - Give each child its own new module, and keep edits to shared files minimal and additive.
     - Tell children *not* to bump `ALGORITHM_VERSION`; the parent sets it at integration.
     - Re-run the combined validation after merging, because features interact.
   - "Work autonomously; don't ask questions. End with a concise summary."
3. Arm a check-in with `send_later` (about 45 min for an audit or experiment). Children don't report back when they finish cleanly.
   - **Waiting on something external (downloads, quotas, long jobs) must cost no tokens** (LEARNINGS M62). Tell children to wait inside ONE background shell loop that retries with back-off and exits only when the data is complete or a deadline passes, never with hourly `send_later`/trigger self-wake-ups. Ask children to `send_message` the parent when they finish, and keep parent check-ins few (one long fallback, not an hourly series).

## Monitor and integrate

- `get_session(<id>)`: `status_bucket` `failed` means the turn errored; `idle` or completed means it's done. `list_events(<id>, kinds=["assistant","result"])` shows the final summary.
- When it's done: `git fetch origin <branch>`, review the diff adversarially, run `pytest -q` and `ruff check .`, then **merge into `main` with a merge commit** (keeps history; resolve `docs/LEARNINGS.md` M-number clashes by renumbering). Push `main`, then `archive_session` the child.
- If a child fails or goes off-brief, `send_message` it a correction or archive it and do the work yourself. Record what went wrong in LEARNINGS.

## Brief template

```
You are <role> for the BehaviorLens repo (local Streamlit + MediaPipe tool suggesting observable movement
events in staged videos for human review). Work autonomously; do not ask questions.
Read CLAUDE.md, docs/LEARNINGS.md, <skill/doc> first. Setup: <commands>.
Task: <question/goal>. Method: <steps>. Do NOT: <limits>.
Deliverable: <file/section/tests>. Run pytest -q and ruff check . before committing.
Constraints: never commit data/, model weights, or footage. Commit to branch <branch> only and push it;
never push to main; no PR. Log lessons in docs/LEARNINGS.md (continue M-numbering). End with a summary.
```
