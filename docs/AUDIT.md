# Clean-room audit

An independent check of the claims in `README.md`, `docs/PRD_TRACEABILITY.md` and `docs/VALIDATION.md`, run from a fresh clone. The goal was to verify claims, not to add features.

- **Date:** 2026-10-18
- **Commit audited:** `591376c` (`main`)
- **Environment:** Linux cloud container, 4 cores, system `/usr/bin/python3.11` (3.11.17). The audit ran in a new venv outside the session's pre-built `.venv`.
- **Installed by `pip install -r requirements.txt` on this date:** mediapipe 1.1.1, streamlit 1.66.0, opencv-python-headless 5.0.0.93 (plus opencv-contrib-python 5.0.0.93, pulled in by mediapipe), numpy 2.4.6, pandas 3.0.6, plotly 7.1.0, pydantic 2.14.1, pytest 9.1.1, pyarrow 26.0.0.

Key: ✅ claim holds · ⚠️ holds with a caveat, or is only partly verified by what the docs cite · ❌ claim is wrong

## 1. README Quick start, followed literally

| Step | Result | Notes |
|---|---|---|
| `git clone https://github.com/ashaankjha/behaviorlens.git && cd behaviorlens` | ✅ | |
| `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` | ✅ 50 s | Every requirement is an unpinned lower bound (`>=`), so today's install resolved to opencv **5.0**, numpy 2.4, pandas **3.0** and plotly **7.1**. Everything still passes, but nothing pins the "tested" set (see D4). mediapipe also installs `opencv-contrib-python` next to `opencv-python-headless`, so two wheels provide `cv2`. `import cv2` works today. |
| `sudo apt-get install -y libegl1 libgles2` | ✅ | Already present in this image. |
| `python scripts/download_model.py` | ✅ 1 s | 5,777,746 bytes (`pose_landmarker_lite@59929e1d1ee9`), and it prints the model-card link. |
| `streamlit run app.py  # opens http://localhost:8501` | ⚠️ | Serves on :8501 and `/_stcore/health` returns `ok`. But `.streamlit/config.toml` sets `headless = true`, so **no browser opens**. Fixed the README comment. Startup also shows two privacy issues (D1). |
| `pytest -q` | ✅ | `76 passed in 8.67s`, 0 skipped. With the model file removed: `73 passed, 3 skipped`, each skip naming `download_model.py`, as the README says. |
| `ruff check .` (CLAUDE.md) | ⚠️ | `All checks passed!` (ruff 0.17.1). But ruff isn't in `requirements.txt`; only the SessionStart hook installs it. In a clean venv, `.venv/bin/ruff` from CLAUDE.md doesn't exist. Same for `playwright`, which the e2e skill needs. |
| `python scripts/generate_demo.py` | ✅ | Output is byte-identical to the committed `sample_data/` (`git status` stays clean), so "deterministic" holds. |

## 2. Tests and lint (main checkout)

```
$ .venv/bin/python -m pytest -q
76 passed in 7.60s
$ .venv/bin/ruff check .
All checks passed!
```

LEARNINGS says the suite takes "~5 s". Here it took 7.6 to 8.7 s, which is close enough.

## 3. Browser E2E (`e2e-verify-streamlit` skill, `--vp9`)

All 6 checks passed (`drive_demo.py` exited 0). I viewed screenshots `02_event_selected.png` and `04_details.png`: the synthetic banner, selected event, before/after thumbnails, summary table and ground-truth agreement all render. Screenshots stayed in scratch and are synthetic only.

```json
{
  "checks": {
    "timeline_has_bars": {"ok": true, "detail": 9},
    "bar_click_selects_event": {"ok": true, "detail": null},
    "thumbnails_shown": {"ok": true, "detail": 2},
    "video_plays_and_seeks": {"ok": true, "detail": {"readyState": 4, "currentTime": 4, "error": null, "h264": ""}},
    "export_reflects_review": {"ok": true, "detail": ["confirmed", "rejected", "pending", "pending", "pending", "pending", "pending", "pending", "pending"]},
    "no_streamlit_exceptions": {"ok": true, "detail": 0}
  },
  "console_errors": ["Client Error: Video source error - http://localhost:8502/media/d008f08b0ad4970d98332fb0733f0b81.mp4"]
}
```

The one console error is the expected first H.264 load, as documented. The skill's first command, `SCR="$SCRATCHPAD/e2e"`, assumes `$SCRATCHPAD` is set. It isn't set in this cloud shell, so `SCR` would resolve to `/e2e` and the next line runs `rm -rf /e2e`. Fixed in the skill and in `extract-chatgpt-share` with `${SCRATCHPAD:?…}` (LEARNINGS M23).

## 4. Claims checked

All 34 test names cited in `PRD_TRACEABILITY.md` exist. I read the bodies of the 10 below.

| # | Claim (source) | How checked | Result | Evidence |
|---|---|---|---|---|
| 1 | FR1: bad extension, size, empty file, spoofed MIME and AVI signature are rejected with helpful errors (`test_validate_upload_rejects_bad_inputs`) | read test | ✅ | Asserts `Unsupported`, `limit`, `empty`, `does not look like` (an `<html>` body named `.mp4`) and an `AVI` mismatch. |
| 2 | FR1/FR6: exports contain no paths, filenames or video (`test_exports_contain_no_paths_filenames_or_video`) | read test | ✅ | Seeds `Jane_Doe_session.mp4`, then asserts the store root, `Jane_Doe`, `source.mp4` and `sessions/` are absent from both CSV and JSON. "No video bytes" holds only implicitly (text exports). |
| 3 | FR2: timestamps come from frame index and source FPS (`test_sample_frame_indices_use_source_fps`) | read test | ✅ | 30 fps → `[0, 6, 12, 18]` and 15 samples; a 4 fps source keeps all 12 frames. |
| 4 | FR2: cache by checksum + model version + settings (cache-hit assertion in `test_video_pipeline_with_fake_estimator_and_cache`) | read test | ✅ | The second `run_analysis` leaves `FakeEstimator.calls` unchanged. The test also checks timestamps are strictly increasing and `model_version == "fake@1"`. (This covers the app's cache. The **validation** cache is keyed differently; see D3.) |
| 5 | FR2: one primary person; multi-person warning ("rule tests set `num_people`") | grep `tests/` for `people=` or any `num_people > 1` | ❌ | `make_seq(people=…)` exists, but **no test passes `people>1`**. The warning text in `analyze_pose` (`pipeline.py:53`) and the largest-bounding-box selection are untested. The row says ✅ but cites no real verification. |
| 6 | FR2: no crash on model failure (`test_missing_model_is_actionable`) | read and ran test | ✅ | Raises `ModelNotFoundError` mentioning `download_model`, and the session becomes `failed` with that error. |
| 7 | FR3: legs hidden → unobservable, no guess (`test_transition_while_legs_hidden_is_not_guessed`) | read test | ⚠️ | Asserts only that **no `standing_transition`** is produced. Frames being *unobservable* (rather than ambiguous or standing) is asserted elsewhere (`test_upper_body_only_never_infers_sitting`, `test_out_of_frame_landmarks_are_unobservable`). The claim holds across tests, not in the one cited. |
| 8 | FR3: "13 synthetic tests" for `hand_raised` | counted `def test_` in `test_event_rules.py` | ✅ | 13 hand-related tests (lines 30–99 plus `horizontal_arms`, `scale_invariant`), some shared with posture edge cases. |
| 9 | FR4/FR5: idempotent review actions (`test_duplicate_request_ids_are_ignored`) | read test | ✅ | A replayed `req-1` leaves the event `confirmed`; a replayed manual add creates one event; `review_history` has 2 rows. |
| 10 | FR5: rerun replaces only unreviewed suggestions, **after confirmation** (`test_rerun_replaces_only_pending_suggestions`) | read test, grep `test_app.py` | ⚠️ | The storage behaviour is asserted exactly (`added 1, replaced_pending 1, skipped_already_reviewed 2`, reviewed and manual events kept, 2 analyses). The **UI confirmation checkbox** (`app.py:326`) isn't exercised by any test. |
| 11 | FR3/Acc. 3: demo yields 3 `hand_raised` and 2 `standing_transition`, labelled synthetic, with one documented miss (`test_demo_session_end_to_end`) | read and ran test | ✅ | Asserts counts 3 and 2, `data_mode == "synthetic"` on events and JSON, and `fn == 1` for the legs-hidden stand-up. |
| 12 | FR7: precision/recall/F1 only when ground truth exists (`test_demo_session_end_to_end`) | read test, `app.py:483` | ⚠️ | The gating is in `app.py` ("No ground truth supplied…"), but the cited test calls `metrics.match_events` directly. **No test covers `parse_ground_truth_csv`** (user-facing CSV input) or `match_events` tolerance edge cases. |
| 13 | FR2: overlay preview (`test_real_inference_standing_person`) | read and ran test (not skipped) | ✅ | Real MediaPipe on a public adult photo yields `{"sustained_standing"}`, `data_mode == "video_inference"`, and `render_overlay` produces a decodable video. |
| 14 | Acc. 1: works from a fresh clone (Python 3.11, Linux, needs `libegl1 libgles2`) | §1 | ✅ | Reproduced. |
| 15 | README/VALIDATION: fetch ~220 MB in ~2 min; 126 Charades + 40 UCF101 clips | ran `fetch_validation_clips.py` | ✅ | 223 MB in 1 m 49 s: `charades: 126 clips`, `ucf101: 40 clips`. |
| 16 | VALIDATION: v0.2 holdout `standing_transition` recall **0.40 (4/10)**, precision **0.40 (4/10)**, 0 sit-down confusions, 0.50 recall when legs observable | `validate_real_video.py --split holdout` | ✅ | `recall 0.4, detections 10, precision 0.4, detections_during_sit_down 0, recall_when_legs_observable 0.5`. 61 clips, `clip_errors {}`, 149 s cold (doc: "~3 min"). |
| 17 | VALIDATION: v0.1 holdout recall **0.20 (2/10)**, precision **0.50 (2/4)**, legs observable 0.25 | the skill's v0.1 `--params` on holdout | ✅ | `recall 0.2, detections 4, precision 0.5, … 0.25`. I also ran the **actual v0.1 code** (`git worktree` at `8886c05`, same cached landmarks): identical. |
| 18 | VALIDATION: all-split rows (v0.2 5/19, 5/12, legs observable 0.42; v0.1 2/19, 2/4, 0.17); UCF101 TP 10, FP 3, FN 6, TN 21, precision 0.77, recall 0.63, per-class 7/8, 3/8, 0/8, 2/8, 1/8; "7 of the 19" stand-up clips <30% observable | `--split all`, both versions | ⚠️ | All reproduce exactly through `--params`. But the **actual v0.1 code** gives 0.15 (not 0.17) for v0.1 all-split "recall when legs observable". The `--params` emulation keeps v0.2's hard-coded in-frame gating (`features.IN_FRAME_TOLERANCE`), which changes which clips count as observable. Recall and precision are unaffected. |
| 19 | VALIDATION: 95% Wilson intervals 0.17–0.69, 0.12–0.49, 0.06–0.51 | recomputed | ✅ | (0.17, 0.69), (0.12, 0.49), (0.06, 0.51). |
| 20 | README: local-only; no telemetry; "the only network access is the one-time download of the pose model" | read Streamlit 1.66 `bootstrap.py` and `net_util.py`; observed startup log | ❌ | See D1. On every start Streamlit GETs `checkip.amazonaws.com` (it printed `External URL: http://18.227.…:8501`) and binds `0.0.0.0`. Usage stats are off (`gatherUsageStats = false`) ✅. |

## 5. Discrepancies, by priority

**D1 (high, privacy claim): the app makes an outbound request and listens on all interfaces.** `.streamlit/config.toml` sets `server.headless = true` but no `server.address`. In Streamlit 1.66 that combination calls `net_util.get_external_ip()` (an HTTP GET to `checkip.amazonaws.com`) on every start, and the server binds `0.0.0.0`, so anyone on the same network can open the app. That contradicts the README's "the only network access is the one-time download of the pose model", and it exposes uploaded session videos and review data to the LAN.
*Fix (not applied, behavior change):* add `address = "localhost"` under `[server]` in `.streamlit/config.toml`. A non-wildcard address skips the external-IP lookup (`bootstrap.py` branch for a manually set `server.address`) and binds only to loopback. I checked this: `streamlit run app.py --server.address localhost` logged `Uvicorn server started on localhost:8503` with no External URL. Add a test that parses the config and asserts it, and document how to opt into LAN access.

**D2 (medium, traceability overclaims): some rows cite tests that don't verify the claim.**
- FR2 multi-person warning (claim 5, ❌): no test with `people>1`. Add one `analyze_pose` test with `make_seq(people=2)` that asserts the "More than one person" warning, and a pipeline test of largest-bounding-box selection.
- FR5 "after confirmation" and FR7 "only when ground truth exists" (claims 10 and 12, ⚠️): the UI gates are untested, and `parse_ground_truth_csv` has no tests at all. Add AppTest cases and parser tests for bad columns, negative or reversed times, and unknown types.
- FR3 legs-hidden row (claim 7, ⚠️): also cite `test_out_of_frame_landmarks_are_unobservable` and `test_upper_body_only_never_infers_sitting`, or add an `UNKNOWN`-state assertion to the cited test.
- Until then, these rows should be 🟡 rather than ✅.

**D3 (medium, validation integrity): the validation landmark cache ignores the model version.** `scripts/validate_real_video.py` caches at `data/validation/cache/{dataset}_{clip}_{fps}fps.npz`. If the pose model changes (for example the planned `pose_landmarker_full`/`heavy` comparison), stale `lite` landmarks are silently reused and the report still prints the new `model_version`. *Fix:* include `model_version()` in the cache filename, or store it in the `.npz` and verify it on load. Until then, delete `data/validation/cache/` whenever the model changes (noted in LEARNINGS §2).

**D4 (medium, reproducibility): "tested with Python 3.11", but nothing pins the tested versions.** `requirements.txt` uses only `>=` bounds. Today's install pulled major versions newer than the ones in LEARNINGS (opencv 5, pandas 3, plotly 7). Dev tools (`ruff`, `playwright`) that CLAUDE.md commands depend on aren't listed. *Fix:* commit a `constraints.txt` or lock file from a known-good venv (`pip freeze`), and a `requirements-dev.txt` with `ruff` and `playwright`. Optionally, add a CI job that runs the README Quick start.

**D5 (low, labelling): the "v0.1" comparison is an emulation.**
- `--params` can't reproduce v0.1 exactly, because in-frame gating is a module constant. The difference only shows in one cell (claim 18).
- The JSON output of a `--params` run is labelled `"algorithm_version": "rules-v0.2"`.
- `RuleParams.from_dict` silently drops unknown keys, so a typo in `--params` silently scores the current defaults.

*Fix:* add a footnote in VALIDATION.md ("v0.1 rows = v0.1 thresholds on v0.2 code; actual v0.1 code gives identical recall and precision, and 0.15 for all-split legs-observable"). Make `from_dict` raise on unknown keys in the script, and record `"params_override": true` in the JSON.

**D6 (low, fixed in this branch):**
- The README said `streamlit run app.py` "opens" the URL. In headless mode it doesn't.
- Skill commands assumed `$SCRATCHPAD` was set, so `rm -rf "$SCR"` would target `/e2e`.

Both were corrected (README line 38; `e2e-verify-streamlit` and `extract-chatgpt-share` SKILL.md files).

**D7 (low, environment note):** mediapipe 1.1.1 installs `opencv-contrib-python` alongside `opencv-python-headless`, so both wheels write the same `cv2` package and whichever installs last wins. It works today. If `cv2` import errors appear after an upgrade, reinstall `opencv-python-headless` last, or remove the extra wheel.

## 6. What was not checked

- Real upload through the browser and pose-overlay rendering in the UI. Only the AppTest and pytest paths were run, plus the synthetic e2e run.
- Visual review of the validation disagreements (VALIDATION §5–6 adjudications).
- Delete-session file cleanup beyond reading that `test_delete_session_removes_records_and_files` exists and passes.

No validation clips, frames, model weights or `data/` contents are committed. `data/validation/` is gitignored, and audit screenshots stayed in the session scratch directory.

## Resolution (parent session, 2026-10-18)

| # | Resolution | Verified by |
|---|---|---|
| D1 | `.streamlit/config.toml` sets `address = "localhost"`. The README privacy text now describes loopback-only binding and how to opt into LAN access deliberately. | `test_server_binds_localhost_only`; a live start logged `Uvicorn server started on localhost` with no External URL; `/proc/net/tcp` showed `LISTEN 127.0.0.1`; a connection to the container's non-loopback IP was refused. |
| D2 | Added tests for the multi-person warning, primary-person selection (extracted `pose.select_primary`), the UI re-run confirmation gate, ground-truth gating in the UI, and `parse_ground_truth_csv`/`match_events` edge cases (`tests/test_metrics.py`). The traceability rows cite them. | The new tests pass. |
| D3 | Fixed independently by the pose-model-comparison session: the cache key includes the model version and fps (`cache_path`). | `test_validation_cache_key_includes_model_and_fps`; the lite rows reproduce v0.2 exactly. |
| D4 | `constraints.txt` (exact tested versions) and `requirements-dev.txt` (adds `ruff` and `playwright`). The README, CLAUDE.md and the SessionStart hook install with `-c constraints.txt`. | Cold-start hook run from a clean copy (see LEARNINGS §5). |
| D5 | `--params` rejects unknown keys (`parse_params`), and overridden runs record `params_override` in the JSON. VALIDATION.md has a footnote on how the v0.1 rows were produced. | `test_validation_params_reject_typos`. |
| D6 | Merged as written (README headless note; `${SCRATCHPAD:?}` guards; lesson renumbered **M23** at merge). | — |
| D7 | Documentation only; no change. | — |
