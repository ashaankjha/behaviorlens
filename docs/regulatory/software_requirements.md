# Software Requirements Specification

> **DRAFT — requires review by qualified regulatory/QA personnel; not a regulatory submission.**

| Field | Value |
|---|---|
| Document ID | BL-REG-SRS-001 |
| Version | 0.1 (draft) |
| Date | 2026-10-20 |
| Status | Draft |
| Owner | TBD (manufacturer) |
| Applies to | BehaviorLens `0.1.0`, `rules-v0.2`, schema version 2 |

Related documents: [Intended use](intended_use.md) · [Software development plan](software_development_plan.md) · [Architecture](architecture.md) · [Risk management](risk_management.md) · [Verification report](verification_report.md) · [Cybersecurity](cybersecurity.md) · [SOUP list](soup_list.md)

---

## 1. Purpose and conventions

This document restates the product requirements in [PRD.md](../PRD.md) as verifiable software requirements with stable identifiers, and adds requirements introduced since the PRD (data integrity, resource limits, logging). It is intended to support traceability from requirement to implementation to verification, alongside the existing [PRD_TRACEABILITY.md](../PRD_TRACEABILITY.md).

- **ID scheme:** `SRS-<group>-<nn>`. Groups: `FR1`–`FR7` (functional, matching PRD §4), `PERF` (performance and resource limits), `DATA` (data integrity), `SEC` (privacy and security), `USE` (usability and labeling). IDs are never reused; a withdrawn requirement keeps its ID with status *withdrawn*.
- **Source:** PRD section, or `LEARNINGS`/`AUDIT` item, or *Engineering* for requirements added during hardening.
- **Verification:** names of tests that exist in `tests/` (`file::test`). *Inspection* means verified by reading configuration or code; *Analysis* means verified by the real-video evaluation in [VALIDATION.md](../VALIDATION.md). Tests marked † skip when offline or when the pose model is missing.
- **PRD_TRACEABILITY row:** the matching row in PRD_TRACEABILITY.md (by requirement column), or **none** where that document has no row yet.
- **RC** marks requirements expected to act as risk controls in [risk_management.md](risk_management.md).

## 2. Functional requirements

### FR1: Upload and session management

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-FR1-01 | The system shall accept only `.mp4`, `.mov` and `.avi` files and reject others with an actionable message. | PRD FR1 | `video_io.validate_upload`, `config.ALLOWED_EXTENSIONS` | `test_pipeline::test_validate_upload_rejects_bad_inputs` | FR1 · Upload `.mp4/.mov/.avi` |
| SRS-FR1-02 | The system shall verify the container signature (`ftyp` for MP4/MOV, `RIFF…AVI ` for AVI) and shall not rely on the client MIME type. (RC) | PRD FR1 | `video_io.validate_upload` | `test_pipeline::test_validate_upload_rejects_bad_inputs` | FR1 · Sanitize filenames; don't trust MIME |
| SRS-FR1-03 | The system shall reject empty files and corrupted or undecodable videos with an actionable message and remove any partial files. | PRD FR1, §11.8 | `validate_upload`, `probe_video`, cleanup in `pipeline.create_upload_session` | `test_pipeline::test_validate_upload_rejects_bad_inputs`, `test_pipeline::test_corrupted_video_rejected_and_cleaned_up` | FR1 · Upload… helpful errors |
| SRS-FR1-04 | The system shall create, per upload, a unique session ID, SHA-256 content checksum, sanitized source filename, creation timestamp, fps, dimensions, frame count, duration and processing status. | PRD FR1 | `storage.Store.create_session`, `pipeline.create_upload_session` | `test_pipeline::test_upload_valid_video_creates_session`, `test_storage::test_session_roundtrip` | FR1 · Session ID, checksum… |
| SRS-FR1-05 | The system shall sanitize filenames to a basename with a restricted character set and bounded length. | PRD FR1 | `video_io.sanitize_filename` | `test_pipeline::test_sanitize_filename` | FR1 · Sanitize filenames |
| SRS-FR1-06 | The system shall provide a one-click synthetic demo session. | PRD FR1 | `pipeline.create_demo_session`; sidebar button in `app.py` | `test_app::test_demo_review_flow`, `test_pipeline::test_demo_session_end_to_end` | FR1 · One-click built-in sample |

### FR2: Pose extraction

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-FR2-01 | The system shall estimate pose locally with MediaPipe Pose Landmarker in VIDEO mode using a locally stored model and decode frames with OpenCV. | PRD FR2 | `pose.MediaPipePoseEstimator`, `video_io.iter_sampled_frames` | `test_pipeline::test_real_inference_standing_person` † | FR2 · MediaPipe… VIDEO mode |
| SRS-FR2-02 | The system shall sample at 5 fps by default (10 fps optional) and derive strictly increasing timestamps from source frame indices and source fps. | PRD FR2 | `video_io.sample_frame_indices`, `PoseSequence` invariant | `test_pipeline::test_sample_frame_indices_use_source_fps`, `test_pipeline::test_video_pipeline_with_fake_estimator_and_cache` | FR2 · 5 fps default… |
| SRS-FR2-03 | The system shall store per-frame landmarks (x, y, z, visibility, presence) and mark frames with no detected person as `no_pose` (all-NaN). | PRD FR2 | `models.PoseSequence`, `pose._to_array` | `test_pipeline::test_real_inference_without_person_yields_no_pose` †, `test_event_rules::test_missing_pose_frames_never_create_events` | FR2 · Store landmarks… `no_pose` |
| SRS-FR2-04 | The system shall analyze one primary person (largest visible bounding box) and warn when more than one person is detected that attribution is not reliable. (RC) | PRD FR2 | `pose.select_primary`, `pipeline.analyze_pose` | `test_event_rules::test_primary_person_is_the_largest_pose`, `test_event_rules::test_multi_person_frames_raise_a_warning` | FR2 · One primary person |
| SRS-FR2-05 | The system shall report a missing pose model with the command to obtain it, and set the session to `failed` on decode or model failure without crashing the UI. | PRD FR2, §11.8 | `pose.ModelNotFoundError`, `pipeline.run_analysis` | `test_pipeline::test_missing_model_is_actionable`, `test_robustness::test_failed_analysis_logs_exception_type_only` | FR2 · Progress, no crash |
| SRS-FR2-06 | The system shall cache pose results keyed by content checksum, model version and sampling rate. | PRD FR2 | `pipeline._pose_cache_path` | `test_pipeline::test_video_pipeline_with_fake_estimator_and_cache` | FR2 · Cache by checksum… |
| SRS-FR2-07 | The system shall offer a local low-resolution pose-overlay preview and remove partial output on failure. | PRD FR2, FR7 | `pipeline.render_overlay`, `video_io.FrameWriter.abort` | `test_pipeline::test_real_inference_standing_person` †, `test_robustness::test_overlay_writer_cleans_up_on_failure` | FR2 · Toggled pose-overlay |

### FR3: Candidate event detection

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-FR3-01 | The system shall suggest `hand_raised` when a wrist is above the same-side shoulder by ≥ `hand_raise_min_height` head units for ≥ 0.6 s, with required landmarks visible; simultaneous raises form one event with side metadata. | PRD FR3.2 | `features.hand_raise_features`, `event_rules.detect_hand_raised` | `test_event_rules::test_hand_raised_basic_right`, `::test_wrist_above_shoulder_uses_image_coordinates`, `::test_exactly_minimum_duration_counts`, `::test_short_spike_below_minimum_is_ignored`, `::test_both_hands_form_one_contiguous_event`, `::test_horizontal_arms_are_not_hand_raised`, `::test_hand_raise_threshold_is_scale_invariant` | FR3 · `hand_raised` |
| SRS-FR3-02 | The system shall suggest `standing_transition` only when a seated-like state and a standing-like state are each held for the dwell duration, within `max_transition_s`; sitting down shall not produce it. | PRD FR3.1 | `features.posture_features`, `event_rules.detect_standing` | `test_event_rules::test_standing_transition_basic`, `::test_sitting_down_is_not_a_standing_transition`, `::test_brief_standing_spike_is_not_a_transition`, `::test_insufficient_dwell_before_transition`, `::test_angled_seated_pose_is_seated_via_hip_angle`, `::test_lunge_is_neither_seated_nor_standing` | FR3 · `standing_transition` |
| SRS-FR3-03 | When required landmarks are not visible or lie outside the frame, the frame state shall be *unobservable*; the system shall never infer sitting from upper-body visibility alone. (RC) | PRD FR3.1, §9 | tri-state features, `features._visible` with `IN_FRAME_TOLERANCE` | `test_event_rules::test_upper_body_only_never_infers_sitting`, `::test_transition_while_legs_hidden_is_not_guessed`, `::test_out_of_frame_landmarks_are_unobservable`, `::test_occluded_wrist_produces_no_event` | FR3 · `standing_transition`…; §7–9 · Missing evidence → unknown |
| SRS-FR3-04 | The system shall optionally suggest `sustained_standing` for an upright pose held ≥ 2 s. | PRD FR3.3 | `event_rules.detect_standing` | `test_event_rules::test_sustained_standing_optional` | FR3 · Optional `sustained_standing` |
| SRS-FR3-05 | The system shall bridge short gaps, split on long gaps, enforce minimum durations, and keep events within the video duration. | PRD FR3 | `event_rules.find_segments`, `segment_bounds` | `test_event_rules::test_find_segments_bridges_short_gaps_only`, `::test_find_segments_empty`, `::test_single_dropped_frame_is_bridged`, `::test_long_gap_splits_events`, `::test_noise_robustness`, `::test_posture_gap_bridging_is_independent_of_hand_gap`, `::test_events_stay_within_duration` | FR3 · Smoothing/debounce… |
| SRS-FR3-06 | Each algorithmic event shall carry `algorithm_version`, rule name, parameters and an evidence summary; `confidence` shall be null. (RC) | PRD FR3 | `event_rules._event`, `models.Event` | `test_event_rules::test_hand_raised_basic_right` | FR3 · `algorithm_version`… no fake probabilities |
| SRS-FR3-07 | Rule thresholds shall be configurable; the UI shall expose a subset as controls. | PRD FR7 | `config.RuleParams`; Analysis controls in `app.py` | `test_event_rules::test_threshold_is_configurable` | FR7 · Threshold controls |
| SRS-FR3-08 | The system shall warn when posture is observable in < 50% of frames or a person is detected in < 50% of frames. (RC) | PRD §9 | `pipeline.analyze_pose` | `test_pipeline::test_demo_session_end_to_end` (observability stats); warning text by *Inspection* | none |

### FR4: Review UI

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-FR4-01 | The UI shall show the title, disclaimer banner, session selector, upload control and status stepper. | PRD FR4, §8 | `app.py` header/sidebar, `status_stepper` | `test_app::test_app_starts_with_empty_state` | FR4 · Title, local-only notice… |
| SRS-FR4-02 | The UI shall show a timeline and table of events; selecting one shall re-render the player about 1 s before the event and disclose seek precision limits. | PRD FR4 | `timeline_chart`, `events_table`, `action_select` | *Inspection*; browser check via `e2e-verify-streamlit` skill | FR4 · Timeline chart… |
| SRS-FR4-03 | The UI shall allow confirm, reject, edit (type, times, note) and manual event insertion. | PRD FR4 | Review pane in `app.py`; `Store.set_status`, `edit_event`, `add_manual_event` | `test_app::test_demo_review_flow`, `test_storage::test_review_actions_persist_across_restart`, `test_storage::test_event_validation` | FR4 · Review pane |
| SRS-FR4-04 | Repeated submission of the same review action (same `request_id`) shall be applied once, including across concurrent store instances; a replayed manual add shall return the stored event. (RC) | PRD FR4 | `request_id UNIQUE`, check inside `BEGIN IMMEDIATE` in `Store._apply_review` and `Store.add_manual_event` | `test_storage::test_duplicate_request_ids_are_ignored`, `test_storage_integrity::test_concurrent_duplicate_request_is_applied_once`, `test_storage_integrity::test_replayed_manual_add_returns_the_stored_event` | FR4 · …no duplicate actions |
| SRS-FR4-05 | Re-running analysis with pending suggestions shall require explicit confirmation in the UI. | PRD FR5 | confirmation checkbox in `app.py` | `test_app::test_rerun_requires_explicit_confirmation` | FR5 · Rerun replaces only unreviewed… |

### FR5–FR7: Persistence, export, polish

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-FR5-01 | Sessions, analyses, events and review history shall persist in SQLite across restarts. | PRD FR5 | `storage.Store` | `test_storage::test_review_actions_persist_across_restart` | FR5 · SQLite sessions… |
| SRS-FR5-02 | Edits to algorithmic events shall preserve the original proposal, also after repeated edits. (RC) | PRD FR5 | `Store.edit_event` (`original_proposal`) | `test_storage::test_original_proposal_kept_after_second_edit` | FR5 · Preserve the original proposal |
| SRS-FR5-03 | Re-running analysis shall replace only pending algorithmic suggestions and preserve reviewed and manual events. (RC) | PRD FR5 | `Store.save_suggestions` | `test_storage::test_rerun_replaces_only_pending_suggestions` | FR5 · Rerun replaces only unreviewed |
| SRS-FR5-04 | Session status shall move ready → processing → review_ready → reviewed → exported, and become `exported` only when no events are pending. | PRD §8 | `refresh_review_status`, `mark_exported` | `test_storage::test_status_moves_back_to_review_ready_when_pending_exists`, `test_storage::test_export_status_only_after_full_review` | §7–9 · States ready → … |
| SRS-FR6-01 | CSV export shall include at least the PRD FR6 fields plus `analysis_id`, `data_mode` and original-proposal fields. | PRD FR6 | `export.CSV_FIELDS` | `test_export::test_csv_fields_and_statuses` | FR6 · CSV minimum fields |
| SRS-FR6-02 | JSON export shall include session metadata, analysis settings, events, the not-clinically-validated warning and `data_mode`. | PRD FR6 | `export.to_json_dict` | `test_export::test_json_structure`, `test_export::test_synthetic_sessions_are_labelled` | FR6 · JSON with metadata… |
| SRS-FR6-03 | Export shall keep rejected events with their status unless the user filters them out. | PRD FR4 | Export tab filter | `test_export::test_status_filter_keeps_rejected_unless_filtered` | FR4 · Export pane keeps rejected |
| SRS-FR7-01 | Precision/recall/F1 shall be shown only when ground truth exists; ground-truth CSVs shall be validated (columns, numeric, finite, `0 ≤ start < end`, type length, row cap). | PRD FR7 | `metrics.match_events`, `metrics.parse_ground_truth_csv` | `test_app::test_ground_truth_metrics_only_when_ground_truth_exists`, `test_metrics::*` (5 tests), `test_robustness::test_ground_truth_rejects_non_finite_and_blank`, `test_robustness::test_ground_truth_row_cap` | FR7 · Precision/recall/F1 only… |
| SRS-FR7-02 | Before/after thumbnails shall be stored locally under the session directory and deleted with the session. | PRD FR7 | `pipeline.event_thumbnails` | `test_pipeline::test_event_thumbnails_are_local_and_deleted_with_session` | FR7 · Before/after thumbnails |

## 3. Performance and resource limits

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-PERF-01 | Uploads larger than 100 MB shall be rejected, both by declared size and while copying, regardless of client-reported size. (RC) | PRD FR1 | `config.MAX_UPLOAD_BYTES`, `pipeline._copy_bounded`, `.streamlit/config.toml` `maxUploadSize = 100` | `test_pipeline::test_too_large_upload_rejected`, `test_robustness::test_upload_copy_is_bounded` | FR1 · ≤100 MB (bounded copy: none) |
| SRS-PERF-02 | Videos shorter than 1 s or longer than 5 min shall be rejected. | PRD FR1 | `config.MIN_DURATION_S`, `MAX_DURATION_S`, `probe_video` | `test_pipeline::test_short_clip_rejected` (max duration: *Inspection* only) | FR1 · ≤5 min |
| SRS-PERF-03 | Videos reporting a non-finite frame rate or one outside 1–240 fps shall be rejected. | PRD §11.8 | `config.MIN_FPS`, `MAX_FPS`, `probe_video` | `test_pipeline::test_bad_fps_rejected` (lower bound; upper bound by *Inspection*) | FR1 · helpful errors |
| SRS-PERF-04 | Videos whose header or first decoded frame exceeds 3840×2160 pixels shall be rejected; decoding shall stop if any later frame exceeds it. (RC) | Engineering | `config.MAX_FRAME_PIXELS`, `probe_video`, `iter_sampled_frames` | `test_robustness::test_oversized_resolution_rejected` (probe), `test_robustness::test_mid_stream_oversized_frame_is_refused` (mid-stream) | none |
| SRS-PERF-05 | No more than `MAX_DECODED_FRAMES` (5 min × 240 fps = 72,000) source frames shall be decoded per analysis, independent of the header frame count. | Engineering | `config.MAX_DECODED_FRAMES`, `iter_sampled_frames` | `test_robustness::test_decode_is_capped_independently_of_header` | none |
| SRS-PERF-06 | A truncated file shall stop decoding with a warning rather than an error. | PRD §11.8 | `iter_sampled_frames` | *Inspection* | none |
| SRS-PERF-07 | Pose model downloads larger than 64 MB shall be aborted. | Engineering | `pose.MAX_MODEL_BYTES` | *Inspection* | none |

## 4. Data integrity

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-DATA-01 | The database schema shall be versioned and upgraded by ordered, transactional migrations; a failed migration shall leave the database unchanged. | Engineering | `storage.MIGRATIONS`, `Store._migrate` | `test_storage_integrity::test_new_database_is_at_latest_schema_with_wal_and_foreign_keys`, `::test_failed_migration_rolls_back` | none |
| SRS-DATA-02 | Existing databases shall be backed up before upgrade and migrated without data loss. | Engineering | backup `behaviorlens.db.pre-v<N>-backup` | `test_storage_integrity::test_legacy_v1_database_is_migrated_without_data_loss` | none |
| SRS-DATA-03 | A database written by a newer schema version shall be refused without modification, and the UI shall show the error. | Engineering | `SchemaVersionError`; `app.py` startup | `test_storage_integrity::test_database_from_newer_release_is_refused` | none |
| SRS-DATA-04 | Connections shall use WAL journaling, `synchronous=FULL` and foreign keys; writes shall use `BEGIN IMMEDIATE`. | Engineering | `Store._connect`, `_conn` | `test_storage_integrity::test_new_database_is_at_latest_schema_with_wal_and_foreign_keys` | none |
| SRS-DATA-05 | `review_history` shall be append-only (no update; delete only through cascade of a deleted session). (RC) | PRD FR5 | triggers in `_migrate_v2` | `test_storage_integrity::test_review_history_is_append_only` | FR5 · …review history (append-only: none) |
| SRS-DATA-06 | Each `review_history` row shall be SHA-256 hash-chained per session; `verify_integrity` shall detect modified, removed, reordered or inserted rows. (RC) | Engineering | `history_row_hash`, `_append_history`, `_verify_session` | `test_storage_integrity::test_review_history_is_hash_chained_and_verifies`, `::test_tampering_with_history_is_detected` | none |
| SRS-DATA-07 | `verify_integrity` shall flag events changed outside the audit trail and reviewed/manual events without an audit record; the UI shall display failures. (RC) | Engineering | `_verify_session`; *Review history* expander in `app.py` | `test_storage_integrity::test_editing_events_outside_the_audit_trail_is_detected`, `::test_reviewed_event_without_audit_record_is_detected`, `test_app::test_audit_trail_status_is_shown_and_flags_tampering` (UI banner) | none |
| SRS-DATA-08 | JSON exports shall carry the review-history row count, chain head and integrity problem count. | Engineering | `export.to_json_dict` `audit` block | `test_storage_integrity::test_json_export_carries_chain_head` | none |
| SRS-DATA-09 | Session deletion shall record session ID, time, history row count and chain head (no content) in `deletion_log`, leaving other chains valid. | PRD §9 | `Store.delete_session` | `test_storage_integrity::test_delete_session_is_logged_and_keeps_chain_valid` | §7–9 · Delete session (log: none) |
| SRS-DATA-10 | Event intervals shall satisfy `0 ≤ start_s < end_s ≤ duration + 0.5 s`; manual/edited types shall be from the allowed list; a failed edit shall change nothing. | PRD §6 | `models.Event` validator, `check_within` | `test_storage::test_event_validation` | FR5 · Stable IDs… |
| SRS-DATA-11 | Pose model downloads shall be verified against the pinned SHA-256 and discarded on mismatch; only HTTPS URLs are allowed. A model at the default path (`config.default_model_path()`) shall be re-verified at load and refused on mismatch (`ModelIntegrityError`); a model at an override path (`BEHAVIORLENS_MODEL_PATH`) is not checked against the pin, but its hash prefix shall be recorded in `model_version`. (RC) | Engineering | `pose.download_model`, `pose.MediaPipePoseEstimator.__init__`, `config.MODEL_SHA256` | `test_robustness::test_model_download_verifies_checksum`, `test_robustness::test_pinned_model_matches_downloaded_file` †, `test_robustness::test_tampered_default_model_is_refused` | none |
| SRS-DATA-12 | `PoseSequence` shall reject wrong shapes, non-increasing timestamps, invalid `sample_fps` and unknown `data_mode`. | Engineering | `PoseSequence.__post_init__` | `test_robustness::test_pose_sequence_rejects_invalid_metadata` | none |

## 5. Privacy and security

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-SEC-01 | The server shall bind to loopback only and Streamlit usage statistics shall be disabled. (RC) | PRD §9; AUDIT D1 | `.streamlit/config.toml` (`address = "localhost"`, `gatherUsageStats = false`); Docker docs require `-p 127.0.0.1:`; Docker `CMD` passes `--browser.gatherUsageStats=false` | `test_app::test_server_binds_localhost_only`, `test_robustness::test_dockerfile_keeps_privacy_and_least_privilege_settings`; CI Docker smoke test | §7–9 · Local processing; telemetry off |
| SRS-SEC-02 | The software shall make no telemetry, cloud upload or LLM calls; the only outbound request is the pose-model download. | PRD §1, §9 | design; `pose.download_model`; Docker `CMD` passes `--logger.hideWelcomeMessage=true` so Streamlit does not look up an external IP (`checkip.amazonaws.com`) when bound to `0.0.0.0` in the container | *Inspection*; AUDIT D1 resolution (observed startup); `test_robustness::test_dockerfile_keeps_privacy_and_least_privilege_settings` (flag present in `Dockerfile`) | §7–9 · Local processing |
| SRS-SEC-03 | Exports shall not contain video bytes, filesystem paths, source filenames or images. (RC) | PRD FR1, FR6 | `export.py` (uses `content_sha256`, omits filename) | `test_export::test_exports_contain_no_paths_filenames_or_video` | FR6 · No video, paths… |
| SRS-SEC-04 | Logs shall be JSON lines to stderr, contain no frames, filenames or absolute paths, and record exception types only (no tracebacks). (RC) | PRD §9 | `logging_setup.RedactingFilter`, `JsonFormatter` | `test_robustness::test_logs_are_json_and_contain_no_paths_or_filenames`, `::test_failed_analysis_logs_exception_type_only`, `::test_redact_removes_paths` | none |
| SRS-SEC-05 | Stored paths shall be relative to the data directory and resolved paths outside it refused; IDs used as directory names shall be restricted to `[A-Za-z0-9_-]{1,64}`. (RC) | Engineering | `Store.path`, `_check_id` | `test_storage_integrity::test_stored_paths_cannot_escape_the_data_dir` | none |
| SRS-SEC-06 | Delete session shall remove DB records, video, landmarks, overlays, thumbnails and the pose cache when no other session shares the checksum. | PRD §9 | `Store.delete_session`; connections set `PRAGMA secure_delete = ON` and deletion is followed by `PRAGMA wal_checkpoint(TRUNCATE)` (overwrite of deleted rows: *Inspection*, no dedicated test) | `test_storage::test_delete_session_removes_records_and_files`, `test_pipeline::test_event_thumbnails_are_local_and_deleted_with_session` | §7–9 · Delete session |
| SRS-SEC-07 | Uploaded data shall be stored under the gitignored data directory, never under web-served paths. | PRD §9 | `config.data_dir()`, `.gitignore`, `.dockerignore` | *Inspection* | FR1 · Sanitize…; §11.10 |
| SRS-SEC-08 | The container shall run as a non-root user. | Engineering | `Dockerfile` `USER behaviorlens` | `test_robustness::test_dockerfile_keeps_privacy_and_least_privilege_settings` (Dockerfile content); CI Docker smoke test (image built only in CI) | none |

## 6. Usability and labeling

| ID | Requirement | Source | Implementation | Verification | PRD_TRACEABILITY row |
|---|---|---|---|---|---|
| SRS-USE-01 | Every page shall show: "Research/demo prototype. Not for clinical decision-making. Use only synthetic or staged footage with informed consent." (RC) | PRD §8 | `config.DISCLAIMER` | `test_app::test_app_starts_with_empty_state` | §7–9 · Exact disclaimer banner |
| SRS-USE-02 | Synthetic data shall be labeled as such on sessions, events, the UI and exports (`data_mode`). (RC) | PRD §7 | `data_mode`; 🧪 banner; `data_mode_note` | `test_app::test_demo_review_flow`, `test_export::test_synthetic_sessions_are_labelled`, `test_pipeline::test_demo_session_end_to_end` | §7–9 · Synthetic fixture clearly separated |
| SRS-USE-03 | Algorithmic events shall be presented as suggestions requiring review, with no calibrated confidence. (RC) | PRD §1, FR3 | review pane caption; `EXPORT_WARNING` | `test_export::test_json_structure` (warning); caption by *Inspection* | FR3 · …no fake probabilities |
| SRS-USE-04 | Vocabulary shall describe observable movement only; no diagnosis, emotion, intent or treatment terms. | PRD §8, §9 | labels in `app.py`, `config.EVENT_TYPES` | *Inspection* | §7–9 · No emotion, intent… |
| SRS-USE-05 | Error and empty states shall state the next action. | PRD §8 | messages in `video_io`, `pose`, `app.py` | `test_pipeline::test_missing_model_is_actionable`, `test_pipeline::test_validate_upload_rejects_bad_inputs` | FR1/FR2 error rows |

## 7. Verification gaps identified

| Requirement | Gap |
|---|---|
| SRS-PERF-02 | No test for the 5-minute maximum. |
| SRS-PERF-03 | No test for the 240 fps upper bound. |
| SRS-PERF-04 | Closed (2026-10-20): `test_robustness::test_mid_stream_oversized_frame_is_refused`. |
| SRS-PERF-05 | Closed (2026-10-20): `test_robustness::test_decode_is_capped_independently_of_header`. |
| SRS-PERF-06, -07 | Truncated-tail warning and model-size cap untested. |
| SRS-FR3-08 | Observability warning text not asserted. |
| SRS-SEC-06 | `secure_delete` and the post-deletion WAL checkpoint are verified by *Inspection* only. |
| SRS-FR4-02 | Timeline selection and seek verified only by the on-demand browser skill. |
| Detection performance | No requirement-level acceptance criteria; current results are reported in [VALIDATION.md](../VALIDATION.md) and summarized in [intended_use.md](intended_use.md) §6. |
