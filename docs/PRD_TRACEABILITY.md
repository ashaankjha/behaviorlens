# PRD traceability

Maps every requirement in [`PRD.md`](PRD.md) to where it is implemented and how it is verified. Keep this updated when behavior changes.

Status key: ✅ implemented and verified · 🟡 implemented with a documented limitation · ⛔ out of scope by the PRD

## Functional requirements

| Req | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|
| FR1 | Upload `.mp4/.mov/.avi`, ≤100 MB, ≤5 min, helpful errors | ✅ | `video_io.validate_upload`, `probe_video`; `.streamlit/config.toml` `maxUploadSize` | `test_validate_upload_rejects_bad_inputs`, `test_short_clip_rejected`, `test_bad_fps_rejected`, `test_too_large_upload_rejected`, `test_corrupted_video_rejected_and_cleaned_up` |
| FR1 | One-click built-in sample session | ✅ | sidebar **Load synthetic demo session** → `pipeline.create_demo_session` | `test_demo_review_flow` (AppTest) |
| FR1 | Session ID, checksum, filename, timestamp, fps, size, duration, status | ✅ | `storage.sessions` table, `pipeline.create_upload_session` | `test_upload_valid_video_creates_session` |
| FR1 | Sanitize filenames; don't trust MIME; no absolute paths in exports | ✅ | `sanitize_filename`, container signature check, relative paths in DB | `test_sanitize_filename`, `test_exports_contain_no_paths_filenames_or_video` |
| FR2 | MediaPipe Pose Landmarker VIDEO mode, local model, OpenCV decode | ✅ | `pose.MediaPipePoseEstimator`, `video_io.iter_sampled_frames` | `test_real_inference_standing_person` |
| FR2 | 5 fps default (10 optional), timestamps from frame index and source FPS | ✅ | `sample_frame_indices`, UI sampling-rate radio | `test_sample_frame_indices_use_source_fps`, monotonic check in `test_video_pipeline_with_fake_estimator_and_cache` |
| FR2 | Store landmarks with visibility/presence; `no_pose` frames | ✅ | `PoseSequence` (N, 33, 5), NaN rows, `.npz` per analysis | `test_real_inference_without_person_yields_no_pose` |
| FR2 | One primary person; warn on multiple | ✅ | All people detected (`num_poses = AnalysisSettings.max_people`, default 4; `MediaPipePoseEstimator.detect_all`), tracked across frames (`tracking.track_people`: landmark-box IoU + centre distance after constant-velocity prediction, 2 s re-identification window, no appearance or identity model). Events are computed on one subject track only (`pipeline.analyze_multi_pose`; default = greatest coverage × mean box size); frames without the subject are unobservable. Warning when >1 substantial track; `stats.tracking` and event `evidence.tracking` record the subject, track count and identity-switch statistics | `test_k5crm_pattern_no_false_stand_up_from_identity_switch`, `test_crossing_paths_keep_their_ids`, `test_crossing_with_occlusion_reidentifies_after_the_gap`, `test_short_absence_keeps_the_id_long_absence_gets_a_new_one`, `test_two_people_of_different_sizes_and_subject_choice`, `test_multi_person_video_is_tracked_and_subject_is_selectable`; real video in [VALIDATION.md](VALIDATION.md#person-tracking). Legacy single-person path: `test_multi_person_frames_raise_a_warning`, `test_primary_person_is_the_largest_pose` |
| FR2 | Reviewer chooses the subject when several people are tracked; overlay labels track numbers | ✅ | **Subject (tracked person)** selectbox in Analysis controls (unique labels, M2), `AnalysisSettings.subject_track_id`; `render_overlay` draws `#n` labels, subject in green | `test_reviewer_can_pick_the_subject_track` (AppTest), `test_reviewer_can_choose_the_other_subject`, `test_overlay_labels_tracks_and_is_invalidated_by_reanalysis` |
| FR2 | Progress, no crash on decode/model failure | ✅ | `st.progress` callback, `run_analysis` sets `failed` + error | `test_missing_model_is_actionable` |
| FR2 | Cache by checksum + model version + settings | ✅ | `_pose_cache_path` (checksum, model version, fps, `max_people`, `CAMERA_MOTION_VERSION`); tracking runs after the cache, so changing the subject re-uses landmarks | cache-hit assertion in `test_video_pipeline_with_fake_estimator_and_cache`, `test_pose_cache_key_includes_max_people`, `test_multi_person_video_is_tracked_and_subject_is_selectable` |
| FR2 | Toggled pose-overlay preview | ✅ | `pipeline.render_overlay`, overlay toggle | `test_real_inference_standing_person` |
| FR3 | `standing_transition` with dwell and visibility gating; unobservable when legs hidden | 🟡 | `features.posture_features` (rules-v0.2: hip/knee angles plus drop, in-frame gating), `event_rules.detect_standing` | synthetic: `test_standing_transition_basic`, `test_upper_body_only_never_infers_sitting`, `test_transition_while_legs_hidden_is_not_guessed`, `test_lunge_is_neither_seated_nor_standing`, `test_angled_seated_pose_is_seated_via_hip_angle`; unobservable state asserted in `test_upper_body_only_never_infers_sitting` and `test_out_of_frame_landmarks_are_unobservable`. **Real video, shipped rules-v0.3** (tracking on, camera-motion masking off by default, chosen by a pre-registered rule): Charades held-out recall 0.20 (2/10), precision 0.40 (2/5), 3 false positives, against rules-v0.2's 0.40, 0.40, 6 ([VALIDATION.md](VALIDATION.md), *rules-v0.3*). Integration regression tests: `test_run_analysis_reports_camera_motion_on_the_tracking_path`, `test_multi_person_extraction_records_camera_motion_and_tracks_carry_it`. Works end to end on real footage; accuracy is modest and limited by leg visibility. **Fixed-camera, full-body footage** (UTKinect-Action3D + PKU-MMD, 506 videos, 195 stand-ups; `scripts/datasets.py`, `validate_real_video.py --dataset`): recall 0.71 (138/195), precision 0.56 (138/248); held-out half 0.65 / 0.55. A pre-registered v0.4 grid did not improve the holdout (VALIDATION *Fixed-camera datasets*). `pose_landmarker_full`/`heavy` don't improve it with v0.2 thresholds (heavy 0/19); lite stays the default (VALIDATION *Pose model comparison*). |
| FR3 | Moving camera → posture unobservable (never guessed), surfaced to the reviewer | 🟡 | `camera_motion.py` (OpenCV feature tracking outside the person box + RANSAC similarity, phase-correlation fallback; tri-state static/moving/unknown), computed in `pipeline.extract_pose`, stored in `PoseSequence.camera_motion` (old `.npz` without it load and mask nothing); `features.posture_features` masks moving frames; `RuleParams.camera_*` thresholds; stats `camera_moving_ratio`/`camera_motion_intervals`, warning in `analyze_pose`, `camera_moving_frames` in event evidence; UI *Camera motion* timeline lane and Analysis-details caption | estimator accuracy and static-camera-with-moving-person: `tests/test_camera_motion.py` (20 tests incl. `test_static_camera_with_moving_person_stays_static`, `test_old_npz_without_motion_still_loads_and_masks_nothing`, `test_extract_pose_records_camera_motion_from_video`); rules: `test_pan_induced_transition_is_suppressed`, `test_camera_motion_does_not_mask_hand_raised`, `test_sequences_without_motion_data_mask_nothing`, `test_camera_motion_is_reported_in_stats_and_warnings`; UI: `test_camera_motion_lane_and_details`. **Real video:** Charades holdout false positives 6 → 3 (both HIOY0 pans and 9HGNV removed), recall 4/10 → 3/10 (a stand-up filmed by a following camera is lost), precision 0.40 → 0.50; per-frame balanced accuracy 0.78 → 0.80 ([VALIDATION.md](VALIDATION.md#camera-motion)). Limitation: cannot tell a pan from a camera tracking a real stand-up. On 506 fixed-camera videos no frame is flagged as moving and masking changes nothing (inert, as intended). |
| FR3 | `hand_raised`: wrist above same-side shoulder ≥0.6 s, confidence-gated, both hands = one event | ✅ | `features.hand_raise_features`, `event_rules.detect_hand_raised` | 13 synthetic tests in `test_event_rules.py`. **Real video:** UCF101 PullUps 7/8 clips detected, JumpRope 0/8; reviewed disagreements in [VALIDATION.md](VALIDATION.md). Fixed-camera (UTKinect + PKU-MMD) against an independent Kinect-skeleton reference (`scripts/skeleton_adjudication.py`), held-out: precision 0.88 (835/946), recall ≥ 0.79 (821/1038) |
| FR3 | Optional `sustained_standing` ≥2 s | ✅ | `detect_standing` | `test_sustained_standing_optional` |
| FR3 | Smoothing/debounce → segments → merged intervals; edge-case tests | ✅ | `find_segments` gap bridging + minimum durations | occlusion, noise, spike, missing-pose tests |
| FR3 | `algorithm_version`, parameters and evidence; no fake probabilities | ✅ | `Event.evidence`, `confidence=None` | `test_hand_raised_basic_right` |
| FR4 | Title, local-only notice, session selector/upload, status | ✅ | `app.py` header and sidebar, `status_stepper` | `test_app_starts_with_empty_state` |
| FR4 | Video player, controls, metadata, counts | ✅ | Review tab | browser run (see LEARNINGS) |
| FR4 | Timeline chart with click/select; replay at or just before the event | 🟡 | Plotly bars with `on_select`, dataframe row select, `st.video(start_time=…)` | browser check: selecting a 25.0 s event positions the player at 24.0 s. Seek precision is browser-dependent (disclosed in the UI) |
| FR4 | Review pane: evidence, accept/reject/edit, manual add, note; no duplicate actions | ✅ | review pane, request IDs | `test_duplicate_request_ids_are_ignored`, `test_demo_review_flow` |
| FR4 | Export pane keeps rejected events unless filtered | ✅ | Export tab multiselect | `test_status_filter_keeps_rejected_unless_filtered` |
| FR5 | SQLite sessions, landmarks, events, review history | ✅ | `storage.py` | `test_review_actions_persist_across_restart` |
| FR5 | Stable IDs, statuses, provenance, timestamps | ✅ | `Event` model | `test_original_proposal_kept_after_second_edit` |
| FR5 | Preserve the original proposal for edited events | ✅ | `original_proposal` + `review_history` | same |
| FR5 | Rerun replaces only unreviewed suggestions, after confirmation | ✅ | `save_suggestions`, UI confirmation checkbox | `test_rerun_replaces_only_pending_suggestions` (storage), `test_rerun_requires_explicit_confirmation` (UI gate, AppTest) |
| FR6 | CSV minimum fields | ✅ | `export.CSV_FIELDS` | `test_csv_fields_and_statuses` |
| FR6 | JSON with metadata, settings, events, warning | ✅ | `export.to_json_dict` | `test_json_structure` |
| FR6 | No video, paths, identities or crops | ✅ | export omits filename and paths | `test_exports_contain_no_paths_filenames_or_video` |
| FR7 | Threshold controls | ✅ | **Analysis controls** sliders | `test_threshold_is_configurable` |
| FR7 | Before/after thumbnails | ✅ | `pipeline.event_thumbnails` | `test_event_thumbnails_are_local_and_deleted_with_session` |
| FR7 | Overlay video | ✅ | `render_overlay` | see FR2 |
| FR7 | Counts and durations, proposed vs confirmed | ✅ | `summary_counts` | browser run |
| FR7 | Precision/recall/F1 only when ground truth exists | ✅ | `metrics.match_events`, `metrics.parse_ground_truth_csv`, ground-truth CSV upload | `test_ground_truth_metrics_only_when_ground_truth_exists` (UI gating), `tests/test_metrics.py` (CSV validation, tolerance, one-to-one matching), `test_demo_session_end_to_end` |

## Sections 7–9: data, UX, privacy

| Requirement | Status | Where |
|---|---|---|
| Synthetic fixture clearly separated from real inference | ✅ | `data_mode` on sessions, events and exports; 🧪 banner; `demo_generator` docstring |
| Exact disclaimer banner and neutral vocabulary | ✅ | `config.DISCLAIMER`; "Candidate observations", "Review status" |
| States ready → processing → review-ready → reviewed/exported | ✅ | `status_stepper`, `refresh_review_status`, `mark_exported` (`test_export_status_only_after_full_review`) |
| Local processing; telemetry off; no remote endpoints; no frames in logs | ✅ | `.streamlit/config.toml` (`gatherUsageStats = false`, `address = "localhost"`, so no 0.0.0.0 bind and no checkip.amazonaws.com call); no frame logging | `test_server_binds_localhost_only`; verified loopback-only listen and refused LAN connection (AUDIT D1) |
| Delete session removes DB rows and derived files | ✅ | `Store.delete_session` (`test_delete_session_removes_records_and_files`) |
| Missing evidence → unknown/no event | ✅ | tri-state features |
| No emotion, intent, stereotypy, aggression or treatment features | ⛔ | not implemented by design |
| User-directed local-video test procedure | ✅ | README, "Optional: test with your own staged clip" |

## Production-readiness controls (beyond the PRD, 2026-10-20)

| Control | Status | Implementation | Verification |
|---|---|---|---|
| Schema versioning, transactional migrations, backup before upgrade, refuse newer schema | ✅ | `storage.MIGRATIONS`, `Store._migrate` | `test_legacy_v1_database_is_migrated_without_data_loss`, `test_failed_migration_rolls_back`, `test_database_from_newer_release_is_refused` |
| WAL, `synchronous=FULL`, foreign keys, `secure_delete`, serialized writes | ✅ | `Store._connect`, `Store._conn(write=True)` | `test_new_database_is_at_latest_schema_with_wal_and_foreign_keys`, `test_concurrent_duplicate_request_is_applied_once` |
| Tamper-evident review history (append-only, hash chain, event cross-check, export anchor, UI banner) | ✅ | `storage._append_history`, `Store.verify_integrity`, `export.to_json_dict["audit"]`, app audit expander | `tests/test_storage_integrity.py`, `test_audit_trail_status_is_shown_and_flags_tampering` |
| Structured logs without PHI, frames or paths | ✅ | `logging_setup.py` | `test_logs_are_json_and_contain_no_paths_or_filenames`, `test_failed_analysis_logs_exception_type_only` |
| Resource limits: bounded upload copy, 4K frames, decode cap, ground-truth CSV caps | ✅ | `pipeline._copy_bounded`, `video_io`, `metrics` | `tests/test_robustness.py` |
| Model supply chain: versioned URL, pinned SHA-256 (download and load) | ✅ | `config.MODEL_SHA256`, `pose.download_model`, `pose.verify_model` | `test_model_download_verifies_checksum`, `test_tampered_default_model_is_refused` |
| CI: ruff, mypy strict, pytest with coverage floor (91 %), pip-audit, container smoke test | ✅ (container job runs only in CI) | `.github/workflows/ci.yml`, `pyproject.toml` | [`regulatory/verification_report.md`](regulatory/verification_report.md) |
| Container: non-root, loopback publish, no checkip call | 🟡 not built locally | `Dockerfile`, `.dockerignore` | `test_dockerfile_keeps_privacy_and_least_privilege_settings`, CI `docker` job |

Draft regulatory-readiness documents (not a submission) are in [`regulatory/`](regulatory/README.md).

## Section 11: acceptance tests

| # | Acceptance test | Result |
|---|---|---|
| 1 | `pip install -r requirements.txt`; documented model command and license notice | ✅ verified from a fresh clone (Python 3.11, Linux; needs `libegl1 libgles2`) |
| 2 | `streamlit run app.py` starts without uncaught exceptions | ✅ AppTest and a live browser run |
| 3 | Demo shows ≥1 `hand_raised` and ≥1 `standing_transition`, labelled synthetic | ✅ `test_demo_session_end_to_end` (3 and 2) |
| 4 | New MP4 upload runs decode and inference, or fails with an actionable message | ✅ real-inference tests, browser upload, and 166 real moving-person clips (Charades and UCF101) run through decode → MediaPipe → rules with zero errors (`scripts/validate_real_video.py`) |
| 5 | Selecting an event shows metadata and seeks the video | ✅ browser seek check (VP9 copy, see LEARNINGS) |
| 6 | Confirm/reject/edit/manual add persist after restart | ✅ `test_review_actions_persist_across_restart` |
| 7 | CSV/JSON fields match reviewed statuses | ✅ `test_export.py` |
| 8 | Invalid format, too large, missing model, missing pose, short clip, bad fps, corrupted video | ✅ `test_pipeline.py` |
| 9 | `pytest -q` passes | ✅ |
| 10 | No patient footage, keys, private assets or local data committed | ✅ `.gitignore`; `sample_data/` is synthetic only |

## Real-video validation (PRD §7, §11)

The PRD asks for detection on staged footage of consenting adults. Charades (crowd workers acting out scripts at home, with consent) is used as that footage: 126 clips, plus 40 UCF101 clips for arm movements. Every clip was processed with real MediaPipe inference and scored against independent annotations. The protocol, results and limits are in [`VALIDATION.md`](VALIDATION.md), and both scripts are reproducible.

## Known limitations (documented, not hidden)

- Stand-up detection is modest on real home video (held-out recall 0.40, precision 0.40 for rules-v0.2; 0.30 and 0.50 with camera-motion masking). Most misses happen when legs are hidden. Camera-motion masking removes the pan false positives but also hides stand-ups filmed by a camera that follows the person. The remaining false positives come from partial-body framing, jitter and a second person.
- Samples are small (10 held-out stand-ups), so the improvement over v0.1 isn't statistically conclusive.
- Validation footage isn't from ABA settings. The README "own staged clip" procedure covers the intended camera setup.

## Independent audit

A clean-room audit by a separate session (`docs/AUDIT.md`, 2026-10-18) reproduced the README setup, the tests, the e2e check and every number in `VALIDATION.md`. All of its discrepancies have been resolved; see the *Resolution* section of `AUDIT.md`.
