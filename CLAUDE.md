# CLAUDE.md: BehaviorLens

Local-first Streamlit and MediaPipe tool that suggests observable movement events in **staged** videos for human review. The spec is `docs/PRD.md`, and `docs/PRD_TRACEABILITY.md` maps each requirement to code and tests.

## Read first

- `docs/LEARNINGS.md` is the **living learnings log**: environment gotchas, past mistakes (M1…), and design decisions. Read it before changing anything non-trivial, and **append to it** whenever you learn something non-obvious or make and fix a mistake (see the `update-learnings` skill).

## Commands

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt -c constraints.txt   # Python 3.11, pinned
sudo apt-get install -y libegl1 libgles2       # Linux: MediaPipe needs EGL even on CPU
.venv/bin/python scripts/download_model.py      # pose model -> assets/models/ (gitignored)
.venv/bin/python -m pytest -q                   # full suite, ~5 s
.venv/bin/ruff check .                          # lint (config in pyproject.toml)
.venv/bin/mypy                                  # strict typing of src/behaviorlens
.venv/bin/python -m pytest -q --cov             # coverage floor (fail_under in pyproject.toml)
.venv/bin/pip-audit -r constraints.txt --no-deps --disable-pip   # dependency CVE scan
.venv/bin/python scripts/generate_soup_list.py  # regenerate docs/regulatory/soup_list.md
.venv/bin/streamlit run app.py                  # UI on :8501
.venv/bin/python scripts/generate_demo.py       # regenerate synthetic sample_data/
.venv/bin/python scripts/fetch_validation_clips.py   # real Charades/UCF101 clips -> data/validation/ (never commit)
.venv/bin/python scripts/validate_real_video.py --split holdout   # real-video metrics (docs/VALIDATION.md)
```

In Claude Code cloud sessions, `.claude/hooks/session-start.sh` does the setup automatically.

## Layout

`app.py` (UI only) → `src/behaviorlens/` (`video_io`, `pose`, `features`, `event_rules`, `pipeline`, `storage`, `export`, `metrics`, `demo_generator`, `config`, `models`). Core logic must stay UI-independent, and the rules must stay runnable on synthetic `PoseSequence`s without video.

## Non-negotiables (from the PRD)

- No real patient data, child footage or cloud video upload. No telemetry or LLM calls.
- No diagnosis, emotion, intent or behavioral-function inference, and no treatment recommendations. Labels describe observable movement only and are flagged as suggestions needing review.
- Missing landmark evidence means *unobservable*, never a guessed classification. `confidence` stays `null`.
- Keep synthetic and real-inference results distinguishable (`data_mode`).
- Exports never include video, paths, source filenames or images.
- Never commit `data/`, model weights or footage of people. `sample_data/` holds synthetic assets only.

## Conventions

- Every rule or threshold change must be measured on real video with the `validate-real-video` skill: select on the tune split, report the held-out half against the previous version, and update `docs/VALIDATION.md`. Synthetic tests alone are not evidence of detection quality (LEARNINGS M15).
- Every rule change needs a synthetic-sequence test in `tests/test_event_rules.py` (use `conftest.make_seq`). Real-inference regressions go in `tests/test_pipeline.py` and must skip when offline or when the model is missing.
- Review actions must stay idempotent (`request_id`), and reruns must never overwrite reviewed events.
- When a Streamlit selectbox `format_func` is used, its labels must be unique (LEARNINGS M2).
- Schema changes: append a migration to `storage.MIGRATIONS` (never edit a released one) and add a migration test. Never write to `review_history` except through `_append_history` (hash chain).
- Log with `log.info("event_name", extra={"fields": {...}})`: IDs and counts only, never paths, filenames, notes or frames.
- Regulatory drafts in `docs/regulatory/` must never claim compliance, clearance or clinical validity, and must be updated with any change they describe.
- Kill background servers with a self-safe pattern: `pkill -f '[s]treamlit run'` (LEARNINGS M1; enforced by a hook).

## Project skills and hooks

- Skills in `.claude/skills/`:
  - `extract-chatgpt-share`: recover a conversation and embedded files from a ChatGPT share link
  - `e2e-verify-streamlit`: launch the app and drive it with Playwright, with screenshots
  - `update-learnings`: how to maintain `docs/LEARNINGS.md`
  - `validate-real-video`: real-footage metrics with tune/holdout discipline
  - `spawn-cloud-session`: delegate independent work to parallel cloud sessions on their own branches, then review and merge
- Hooks in `.claude/settings.json`:
  - SessionStart (cloud setup)
  - PostToolUse (ruff on edited `.py` files)
  - PreToolUse (block self-matching `pkill -f`/`pgrep -f`)

## Parallel cloud sessions

The owner wants new cloud sessions spawned **whenever necessary** for independent work (audits, experiments, long validations, isolated features). Children work on their own `audit/`, `experiment/` or `feature/` branches and never push to `main`. This session reviews, tests and merges their work. See the `spawn-cloud-session` skill.
