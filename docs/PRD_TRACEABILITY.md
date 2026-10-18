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
| FR2 | One primary person; warn on multiple | ✅ | `num_poses=2`, largest bounding box, `analyze_pose` warning | rule tests set `num_people`; warning text in `analyze_pose` |
| FR2 | Progress, no crash on decode/model failure | ✅ | `st.progress` callback, `run_analysis` sets `failed` + error | `test_missing_model_is_actionable` |
| FR2 | Cache by checksum + model version + settings | ✅ | `_pose_cache_path` | cache-hit assertion in `test_video_pipeline_with_fake_estimator_and_cache` |
| FR2 | Toggled pose-overlay preview | ✅ | `pipeline.render_overlay`, overlay toggle | `test_real_inference_standing_person` |
| FR3 | `standing_transition` with dwell and visibility gating; unobservable when legs hidden | ✅ | `features.posture_features`, `event_rules.detect_standing` | `test_standing_transition_basic`, `test_upper_body_only_never_infers_sitting`, `test_transition_while_legs_hidden_is_not_guessed`, `test_lunge_is_neither_seated_nor_standing` |
| FR3 | `hand_raised`: wrist above same-side shoulder ≥0.6 s, confidence-gated, both hands = one event | ✅ | `features.hand_raise_features`, `event_rules.detect_hand_raised` | 11 tests in `test_event_rules.py` |
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
| FR5 | Rerun replaces only unreviewed suggestions, after confirmation | ✅ | `save_suggestions`, UI confirmation checkbox | `test_rerun_replaces_only_pending_suggestions` |
| FR6 | CSV minimum fields | ✅ | `export.CSV_FIELDS` | `test_csv_fields_and_statuses` |
| FR6 | JSON with metadata, settings, events, warning | ✅ | `export.to_json_dict` | `test_json_structure` |
| FR6 | No video, paths, identities or crops | ✅ | export omits filename and paths | `test_exports_contain_no_paths_filenames_or_video` |
| FR7 | Threshold controls | ✅ | **Analysis controls** sliders | `test_threshold_is_configurable` |
| FR7 | Before/after thumbnails | ✅ | `pipeline.event_thumbnails` | `test_event_thumbnails_are_local_and_deleted_with_session` |
| FR7 | Overlay video | ✅ | `render_overlay` | see FR2 |
| FR7 | Counts and durations, proposed vs confirmed | ✅ | `summary_counts` | browser run |
| FR7 | Precision/recall/F1 only when ground truth exists | ✅ | `metrics.match_events`, ground-truth CSV upload | `test_demo_session_end_to_end` |

## Sections 7–9: data, UX, privacy

| Requirement | Status | Where |
|---|---|---|
| Synthetic fixture clearly separated from real inference | ✅ | `data_mode` on sessions, events and exports; 🧪 banner; `demo_generator` docstring |
| Exact disclaimer banner and neutral vocabulary | ✅ | `config.DISCLAIMER`; "Candidate observations", "Review status" |
| States ready → processing → review-ready → reviewed/exported | ✅ | `status_stepper`, `refresh_review_status`, `mark_exported` (`test_export_status_only_after_full_review`) |
| Local processing; telemetry off; no frames in logs | ✅ | `.streamlit/config.toml`, no frame logging |
| Delete session removes DB rows and derived files | ✅ | `Store.delete_session` (`test_delete_session_removes_records_and_files`) |
| Missing evidence → unknown/no event | ✅ | tri-state features |
| No emotion, intent, stereotypy, aggression or treatment features | ⛔ | not implemented by design |
| User-directed local-video test procedure | ✅ | README, "Optional: test with your own staged clip" |

## Section 11: acceptance tests

| # | Acceptance test | Result |
|---|---|---|
| 1 | `pip install -r requirements.txt`; documented model command and license notice | ✅ verified from a fresh clone (Python 3.11, Linux; needs `libegl1 libgles2`) |
| 2 | `streamlit run app.py` starts without uncaught exceptions | ✅ AppTest and a live browser run |
| 3 | Demo shows ≥1 `hand_raised` and ≥1 `standing_transition`, labelled synthetic | ✅ `test_demo_session_end_to_end` (3 and 2) |
| 4 | New MP4 upload runs decode and inference, or fails with an actionable message | ✅ real-inference tests and browser upload |
| 5 | Selecting an event shows metadata and seeks the video | ✅ browser seek check (VP9 copy, see LEARNINGS) |
| 6 | Confirm/reject/edit/manual add persist after restart | ✅ `test_review_actions_persist_across_restart` |
| 7 | CSV/JSON fields match reviewed statuses | ✅ `test_export.py` |
| 8 | Invalid format, too large, missing model, missing pose, short clip, bad fps, corrupted video | ✅ `test_pipeline.py` |
| 9 | `pytest -q` passes | ✅ |
| 10 | No patient footage, keys, private assets or local data committed | ✅ `.gitignore`; `sample_data/` is synthetic only |

## Known gaps

- Hand-raise and stand-up rules have not been checked against real *moving* staged footage, because none was available to the agent. Real inference has only been run on still photos of adults. Use the README procedure above to close this gap.
