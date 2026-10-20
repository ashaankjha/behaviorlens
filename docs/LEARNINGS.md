# BehaviorLens: living learnings log

A running record of what we have learned building BehaviorLens: environment gotchas, library behavior, mistakes and their fixes, and the decisions behind the design. **Append to it whenever you learn something non-obvious.** See "How to update" at the end.

Last updated: 2026-10-20

---

## 1. Project origin

- The project spec came from a ChatGPT share link. The share page is rendered client-side, so `WebFetch` returned only the page title. The conversation, including the full PRD written by a sandbox `cat > BehaviorLens_PRD.md <<'EOF'` tool call, is embedded in the HTML as a React Router turbo-stream (`streamController.enqueue("…")`). Decoding it recovered the PRD verbatim (`docs/PRD.md`). The method is packaged as the `extract-chatgpt-share` skill.
- A `sandbox:/mnt/data/...` download link in a ChatGPT answer is **not** fetchable. Look for the tool call that wrote the file inside the share data instead.

## 2. Environment and tooling

| Topic | Learning |
|---|---|
| Python | The cloud container defaults to Python 3.13, but the PRD targets 3.11. Use `python3.11 -m venv .venv`. |
| MediaPipe | `mediapipe==1.1.1` (newer than 0.10.x) keeps the `mediapipe.tasks.python.vision.PoseLandmarker` API: `PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=…), running_mode=RunningMode.VIDEO, num_poses=…)`, then `detect_for_video(mp.Image(...), timestamp_ms)`. |
| MediaPipe on headless Linux | Even CPU inference loads `libEGL.so.1`. Without it: `OSError: libEGL.so.1: cannot open shared object file`. Fix: `apt-get install -y libegl1 libgles2`. |
| MediaPipe VIDEO mode | Timestamps must be strictly increasing integer milliseconds. We derive them from source frame indices and enforce `max(ts, last + 1)`. |
| Model URL | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task` (~5.8 MB; byte-identical to `/latest/` on 2026-10-20, SHA-256 `59929e1d…` pinned in `config.MODEL_SHA256`). Downloaded, gitignored, never committed. |
| Public test images | The `mediapipe-assets` GCS bucket is listable through the JSON API (`storage.googleapis.com/storage/v1/b/mediapipe-assets/o`). Useful adult photos: `pose.jpg` (yoga lunge, arms horizontal) and `male_full_height_hands.jpg` (standing, full body). Use them only in temp dirs; don't commit them (licence unclear). |
| Wikimedia Commons | The API returned HTTP 429 from this container, so don't rely on it for test media. |
| ffmpeg | `imageio-ffmpeg` bundles a static ffmpeg with libx264. We use it for H.264 previews, the demo video and overlays, so system ffmpeg isn't needed. |
| OpenCV `mp4v` | Browsers can't play MPEG-4 Part 2. Anything shown in a browser must be H.264 (or VP9/WebM). |
| Playwright Chromium | The bundled Chromium (`/opt/pw-browsers/chromium-1194/chrome-linux/chrome`) has **no H.264**: `canPlayType('video/mp4; codecs="avc1…"') == ''`, and `<video>` fails with error code 4. VP9 and AV1 report `"probably"`. To test playback, point a scratch session at a VP9 copy. Real Chrome, Edge, Safari and Firefox play H.264. |
| Streamlit version | 1.66: `width="stretch"` replaces `use_container_width`; `st.plotly_chart(on_select=…, selection_mode="points")` and `st.dataframe(on_select=…, selection_mode="single-row")` work. |
| Claude Code hooks | Project hooks live in `.claude/settings.json` and `.claude/hooks/`. PostToolUse exit code 2 sends stderr back to the model as blocking feedback; PreToolUse exit 2 blocks the tool call. Hooks hot-load mid-session (see M13). `CLAUDE_CODE_REMOTE=true` is set in cloud sessions. `CLAUDE_ENV_FILE` lets a SessionStart hook export `PATH` for later commands. |
| SessionStart cost | A cold `session-start.sh` (venv, requirements, ruff, playwright, model) takes about 40 s; a warm run under 1 s. It runs synchronously so tests never race the install. |
| Real validation data | Hugging Face hosts a 126-clip **Charades** sample with individual MP4s and parquet metadata (labels and `action_timings`): `Aditya02/Charades-Action-Sequence-Sample`. Label 154 is "standing up", 151 is "sitting down"; sitting and door/light activities give per-frame posture ground truth. The full UCF101 zip (`quchenyuan/UCF101-ZIP`, 7 GB) supports HTTP **Range** requests, so `zipfile` over a seekable HTTP reader extracts single classes without downloading the archive (`HttpRangeFile` in `scripts/fetch_validation_clips.py`). |
| Validation cost | 166 clips (20,057 frames) at 5 fps with 4 processes: lite 137 s wall / 491 CPU-s, full 162 s / 579 s, heavy 364 s / 1,323 s; lite at 10 fps 253 s / 910 s. Landmarks are cached in `data/validation/cache/` per clip, model version and fps, so re-scoring rules takes seconds. |
| Pose model sizes and speed | `float16/latest`: lite 5.8 MB, full 9.4 MB, heavy 30.7 MB (same URL pattern, `pose_landmarker_{lite,full,heavy}`). Single-process CPU inference: 24 / 28 / 60 ms per frame. Full is only ~15% slower than lite; heavy is 2.5×. |
| MediaPipe visibility calibration | Visibility scores are **not comparable across pose models**. On Charades, knees with visibility ≥ 0.5: lite 58%, full 55%, heavy 37% of pose frames (median 0.61 / 0.56 / 0.27); hips and shoulders barely differ. |
| Cloud sessions | `create_session` (claude-code-remote MCP) starts a child cloud session from a repo and branch. Set `outcome_branch` to a non-`main` branch and `permission_mode: auto` (`plan` would stall unattended). Children don't report back on a clean finish: poll with `get_session` and `list_events`, or arm a `send_later` check-in. |
| Streamlit network | With `server.headless = true` and no `server.address`, Streamlit 1.66 GETs `checkip.amazonaws.com` on every start (to print an "External URL") and binds `0.0.0.0`. `--server.address localhost` (or `address = "localhost"` in `config.toml`) avoids both. See AUDIT D1. |
| Docker in cloud sessions | The `docker` CLI is installed but no daemon runs (`/var/run/docker.sock` missing), and starting `dockerd` is refused by the sandbox. Verify images in CI instead (`.github/workflows/ci.yml` job `docker`), and pin Dockerfile invariants in a test (`test_dockerfile_keeps_privacy_and_least_privilege_settings`). |
| mypy strict | `strict = true` on `src/behaviorlens` needed only ~25 fixes, mostly bare `dict` annotations, numpy expressions returning `Any` (bind to a typed local), and untyped lambdas (`lerp = lambda …` counts as an untyped call). `mediapipe` and `imageio_ffmpeg` have no stubs (`ignore_missing_imports`); OpenCV 5 and numpy ship their own. |
| Streamlit testing | `streamlit.testing.v1.AppTest` runs the script headlessly and supports clicking buttons. It can't click Plotly charts; use Playwright for that. |

## 3. Mistakes and their fixes

| # | Mistake | Symptom | Fix and lesson |
|---|---|---|---|
| M1 | `pkill -f "streamlit run"` inside a compound Bash command | The tool shell killed itself (exit 144), because the pattern appears in the shell's own command line | Use `pkill -f '[s]treamlit run'` or `kill $(pgrep -f '[.]venv/bin/streamlit')`. The bracket trick stops the pattern matching itself. A PreToolUse hook (`guard_pkill.py`) now blocks the unsafe form. |
| M2 | Session labels in the selectbox weren't unique (same filename, same minute) | Clicking a timeline bar re-rendered a **different** session, so selection looked broken | This Streamlit version's selectbox resolves the choice by its *formatted label*. Make `format_func` output unique (we append the short session ID). |
| M3 | Hand raise measured as wrist above shoulder by 0.02 in normalized image units | Real yoga photo with horizontal arms became `hand_raised` | Measure in body-relative units. We use "head units" (nose-to-shoulder-line height) with a threshold of 0.5. Real inference exposed what the synthetic tests had missed. |
| M4 | Seated test used the **average** of both thighs | A lunge (one thigh horizontal, one leg straight) read as "seated" | Require **both** legs to agree (`max(drop) ≤ seated`, `min(drop) ≥ standing`). Otherwise the frame is ambiguous. |
| M5 | `visible_frame_ratio` counted a frame as visible if *any* arm was observable | A dropped right-wrist frame during a right-hand raise still reported 100% visibility | Compute visibility for the side(s) actually raised. |
| M6 | A session with zero events was marked `reviewed` immediately | The status skipped review-ready after an empty analysis | Treat "no events" as review-ready; the reviewer may still add manual events. |
| M7 | Export set the status to `exported` even with pending events | The stepper showed "✓ Reviewed" while 7 events were pending | `Store.mark_exported` only advances fully reviewed sessions. |
| M8 | Clicking Plotly bars with `locator.click()` timed out | Plotly's `nsewdrag` layer intercepts pointer events | Click at the bar's bounding-box centre with `page.mouse.click(x, y)`, which is what a real user does. |
| M9 | `@st.cache_resource def get_store()` with no arguments | Tests would share one Store across temp data dirs | Key the cached resource by data-dir path: `get_store(str(config.data_dir()))`. |
| M10 | First upload temp file used a fixed name (`_incoming/upload.mp4`) | Concurrent uploads could clobber each other | Use a UUID temp name. |
| M11 | Screenshots with `full_page=True` were cut off | Streamlit scrolls inside its own container, not the page | Use a tall viewport (for example 1440×1750) and `scroll_into_view_if_needed` instead. |
| M12 | First `drive.py` run created a demo session; re-running created another | Duplicate sessions in scratch data caused M2 | E2E scripts should start from a fresh `BEHAVIORLENS_DATA_DIR`. |
| M13 | The first `guard_pkill.py` scanned the raw command text | It blocked a harmless command whose **heredoc body** contained `pkill -f streamlit`, and would have flagged `echo pkill -f …` | Command-scanning hooks must ignore heredoc bodies and only treat words in *command position* (after `sudo`/`exec`/… prefixes) as invocations. Covered by `tests/test_hooks.py`. Hooks in `.claude/settings.json` become active **in the running session** as soon as the file is written, so a buggy hook can block you immediately. Write hook tests to a file (or with the Write tool), not inline in Bash. |
| M14 | The PostToolUse lint hook linted every edited `.py`, including scratch files outside the repo | Noisy "fix before moving on" feedback for throwaway probes | Scope hooks to `$CLAUDE_PROJECT_DIR` (compare `realpath`s). |
| M15 | I declared the PRD finished with the rules validated only on synthetic data and still photos | A goal-check hook rejected "complete": the PRD's core claim (detection on staged footage of real people) had no evidence | **Synthetic tests prove the plumbing, not the claim.** Find real, consented, annotated footage before calling detection done. Charades (crowd-acted home scripts) had it all along on Hugging Face. |
| M16 | v0.1 posture thresholds were tuned to my own synthetic skeleton | On real Charades video: stand-up recall 2/19. Seated drop median was 0.39, against my "≤ 0.35" | Measure feature distributions on labelled real frames *before* picking thresholds. Hip and knee **angles** separate seated from standing far better than vertical drop; the drop cue only suits front-on cameras. |
| M17 | I assumed "visibility ≥ 0.5" meant "observed" | 2.45% of such landmarks lie outside the image; MediaPipe also hallucinates a head inside the frame on torso-only shots | Require landmarks inside the frame (`IN_FRAME_TOLERANCE`). Treat partial-body framing as a known failure mode rather than patching it from one clip. |
| M18 | I used dataset class labels (JumpingJack = arms overhead) as ground truth for an operational definition | 5 "misses" were 0.4 s raises (shorter than the PRD's 0.6 s); 2 squat "false positives" were real hands-above-shoulders | Class labels ≠ event definitions. **Look at frames** for every disagreement before changing rules, and report raw and adjudicated numbers separately, flagging who adjudicated. |
| M19 | I nearly tuned temporal parameters on all the data | The tuning half was dominated by unobservable clips, and the best "all data" settings would have overfit 19 positives | Fix a deterministic tune/holdout split (`split_of`, sha1 of the clip ID) up front. Select on tune, score holdout once, and present the sensitivity table as post-hoc only. |
| M20 | The first per-frame metrics in VALIDATION.md came from my exploratory script, not the shipped code | Numbers were slightly off (for example "6 of 19" should have been 7) | Recompute every reported number with the production code path before publishing. |
| M21 | `validate_real_video.py` cached landmarks under `<clip>_<fps>fps.npz`, with no model in the key | Switching `BEHAVIORLENS_MODEL_PATH` to full/heavy would have scored **lite's** landmarks and reported "no difference" | Every cache key must include everything that changes the cached value: the model version (name + weight hash) and fps (`cache_path`, test `test_validation_cache_key_includes_model_and_fps`). Sanity-check a new run by reproducing a known baseline exactly (the lite rows matched v0.2 to the count). |
| M22 | I expected the larger pose models to help leg visibility, so stand-up recall would rise | Heavy found 0/19 stand-ups (lite 5/19); full 3/19 | Thresholds are coupled to the model that selected them, **including `visibility_threshold`**: heavy is more conservative about occluded legs, so the same 0.5 gate drops most posture frames. A model swap is a threshold change and needs the full tune/holdout protocol. Even at matched observability (diagnostic only), heavy didn't beat lite. |
| M23 | Skill commands used `SCR="$SCRATCHPAD/e2e"`, assuming the variable was set | In a cloud shell `$SCRATCHPAD` was empty, so the next line would have run `rm -rf /e2e` | Never let a path in an `rm -rf` depend on an env var that may be unset. Use `${SCRATCHPAD:?…}` so the command fails loudly. Fixed in the `e2e-verify-streamlit` and `extract-chatgpt-share` skills. |
| M24 | I wrote "the only network access is the model download" from my own code, without checking the framework | The audit found Streamlit binding `0.0.0.0` and calling `checkip.amazonaws.com` on every start (AUDIT D1) | Privacy and network claims must be **observed**: watch the startup log and listening sockets (`/proc/net/tcp` when `ss` is missing), and try a non-loopback connection. Pin the framework behaviour in config (`server.address = "localhost"`) and in a test. |
| M25 | Traceability rows were marked ✅ when the cited test didn't assert the claim (multi-person warning, UI gates, CSV parser) | The clean-room audit downgraded them (AUDIT D2) | A ✅ row must name a test whose assertions cover the claim's exact wording. An independent auditor session is cheap and catches overclaiming that the author can't see. |

| M34 | Ran `pip install -q -r requirements-dev.txt` in the background on a fresh venv | Two attempts sat at 7–10 installed packages for 30+ minutes with no output, while `curl` fetched the same mediapipe wheel at 34 MB/s | Killed it and re-ran with `-v --progress-bar off` to a log file: done in ~2 minutes. **Never leave a long install silent**: log verbosely, watch for progress (site-packages count, log growth), and compare against a direct `curl` before blaming the network. |
| M35 | Nearly set the coverage floor from `coverage report`'s integer total (92 %) | The exact figure was 91.57 %, so `fail_under = 92` would have failed CI on the first run | Read coverage with `--precision=2` and round **down** (floor 91). |
| M36 | Assumed the AUDIT D1 fix (`server.address = "localhost"`) also covered the container | In Docker, Streamlit must listen on `0.0.0.0`, and a wildcard address makes `bootstrap._print_url` GET `checkip.amazonaws.com` again | `--logger.hideWelcomeMessage=true` returns before the lookup (verified by calling `_print_url` with a trapped `get_external_ip`); host exposure is limited by `-p 127.0.0.1:8501:8501`. **Re-verify network/privacy claims for every deployment mode**, not just the dev server (extends M24). |
| M37 | Had subagents draft the regulatory docs from the code while I was still changing it | Within the hour the drafts reported gaps I had since fixed (coverage floor 0, unpinned tools, no load-time model check, untested caps) | Their discrepancy lists were valuable review input, but **re-audit docs against the final code before committing**, and generate what can be generated (`scripts/generate_soup_list.py`). |

## 4. Design decisions (and why)

- **Thresholds come from real data, selected on a tuning split** (rules-v0.2). Each threshold in `RuleParams` cites its source in a comment. The previous version's thresholds stay reproducible through `--params`, so every change can be compared on the held-out half.
- **Pose model stays `pose_landmarker_lite` at 5 fps** (2026-10-18 comparison in `docs/VALIDATION.md`). Full and heavy didn't improve stand-ups with v0.2 thresholds (heavy: 0/19), and heavy's small `hand_raised` gain (1 TP, 1 FP fixed of 40 clips) doesn't justify 2.5× CPU. 10 fps doesn't recover sub-0.6 s JumpingJack raises; it raises stand-up recall but lowers precision, because the v0.2 temporal parameters were selected at 5 fps.
- **Rules before ML.** Interpretable heuristics with evidence and parameters, as the PRD asks. `confidence` stays `null` because no score is a calibrated probability.
- **Tri-state frames** (true / false / unobservable). Missing landmarks never become a negative or positive observation, so an occluded stand-up is a *documented miss*, not a guess. The synthetic demo deliberately contains one.
- **Synthetic demo bypasses inference.** A stick figure doesn't yield reliable MediaPipe landmarks, so `data_mode="synthetic"` is carried on sessions, events and exports, with a 🧪 banner in the UI.
- **Reruns preserve reviewed work.** Pending algorithm events are replaced; confirmed, rejected, edited and manual events are kept; overlapping new suggestions are skipped.
- **Idempotent actions.** Each review action carries a `request_id` with a UNIQUE constraint in `review_history`, so Streamlit reruns can't double-apply an action.
- **Pose cache** keyed by checksum, model version and fps, so threshold tweaks re-run only the cheap rule stage.
- **Exports omit the source filename** (it may contain a person's name) and use the content SHA-256 instead.
- **Before/after thumbnails** are taken 0.5 s outside the event interval (clamped to the video), cached as JPEGs keyed by timestamp so edits regenerate them, and shown only in the UI.
- **Thumbnails and overlays** live under `data/sessions/<id>/` so Delete session removes them, and they're never exported.

- **Independent audit plus parallel experiment paid off:** the audit predicted the validation-cache bug (D3) that the experiment session independently hit and fixed (M21). Both children stayed on-brief with standalone prompts. Merge conflicts only happened in `docs/LEARNINGS.md` (concurrent M-numbers), resolved by renumbering at merge.
- **Schema changes go through migrations** (`storage.MIGRATIONS`, `schema_version` table). Never edit a released migration; append one. Existing databases are backed up (`behaviorlens.db.pre-vN-backup`) before upgrading, and a database from a newer release is refused rather than modified. Pre-versioning databases are detected as v1.
- **Review history is tamper-evident, not tamper-proof.** Per-session SHA-256 chain plus append-only triggers; `verify_integrity` also cross-checks each reviewed event against its last audit record. Anyone with write access can rebuild the chain, so JSON exports carry the chain head as an external anchor. We rejected app-level signing: it would need key management, and we don't roll our own crypto.
- **The pose model is pinned by URL version and SHA-256** (`config.MODEL_SHA256`), checked on download and when loading the default model. An upstream model update fails loudly instead of silently changing inference.
- **Encryption at rest is delegated to the OS** (LUKS, FileVault, BitLocker), not implemented in the app (`docs/regulatory/cybersecurity.md`).
- **Parallel work goes to child cloud sessions on separate branches** and is merged by the parent after review. First use: a clean-room README audit (`audit/clean-room-readme`) and a pose-model comparison (`experiment/pose-model-comparison`).

## 5. Verification recipes that worked

- Full suite: `.venv/bin/python -m pytest -q` (~5 s; the two real-inference tests download sample photos and skip offline).
- Fresh-clone check: clone to scratch, create a venv, `pip install -r requirements.txt`, `python scripts/download_model.py`, `pytest -q`.
- Hooks: `pytest -q tests/test_hooks.py`; cold-start the SessionStart hook in a copy of the tree without `.venv` (`git ls-files -co --exclude-standard | tar …`) and run `CLAUDE_CODE_REMOTE=true ./.claude/hooks/session-start.sh`.
- Browser E2E: see the `e2e-verify-streamlit` skill (start the app on a scratch data dir, drive it with Playwright, check `[data-testid=stException]` count == 0).
- **Real-video validation**: `scripts/fetch_validation_clips.py`, then `scripts/validate_real_video.py --split holdout`. Any rule change must report v0.x against the previous version on the held-out half (see the `validate-real-video` skill and `docs/VALIDATION.md`).
- **Score an old rules version exactly**: `--params` can't switch off v0.2's in-frame gating. Instead, `git worktree add --detach <dir> <old-commit>`, copy in the current `scripts/validate_real_video.py`, symlink the model, and run it with `BEHAVIORLENS_DATA_DIR=<repo>/data` so it reuses the cached landmarks (a few seconds per split). Actual v0.1 matched the documented recall and precision exactly (AUDIT claims 17–18).
- **Clean-room check**: clone into scratch and drop the session `.venv` from `PATH` before `python3.11 -m venv`, or `ruff`/`playwright` silently resolve from the pre-built venv.
- Real-inference sanity: run the rules on `male_full_height_hands.jpg` (expect STANDING and no hand raise) and `pose.jpg` (expect ambiguous posture and no hand raise).

## 6. Open questions and next learning targets

- ~~How do thresholds behave on real moving footage?~~ Answered in `docs/VALIDATION.md`: modest (held-out stand-up recall 0.40) and bounded by leg visibility.
- Do front-on clinic-style recordings (the PRD's intended setup) score much higher than Charades home video? Use the README "own staged clip" procedure.
- Would person tracking plus camera-motion detection remove most of the remaining stand-up false positives (pans, a second person)?
- ~~Whether `pose_landmarker_full` or `pose_landmarker_heavy` improves leg visibility.~~ Answered: not with v0.2 thresholds (M22, `docs/VALIDATION.md`).
- Would 10 fps with temporal parameters re-selected on the tune split beat 5 fps on stand-ups? (Untuned 10 fps: holdout recall 6/10, precision 6/18.)
- Would a model-specific `visibility_threshold`, selected on the tune split, make `heavy` competitive? The post-hoc diagnostic suggests not.
- Side-view cameras: the hip-to-knee drop heuristic assumes a roughly frontal view.

---

## How to update this file

- Add an entry the moment something surprises you: a library behaving unexpectedly, an environment failure, a bug you caused, or a test that caught a wrong assumption.
- Mistakes go in §3 with **symptom → fix → lesson**. Number them sequentially (M13, M14, …).
- Environment facts go in §2. Design trade-offs go in §4. Change existing lines rather than duplicating them.
- Bump "Last updated". The `update-learnings` skill describes the routine.
