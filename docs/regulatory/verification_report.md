# Software verification report

> **DRAFT — requires review by qualified regulatory/QA personnel; not a regulatory submission.**

| Field | Value |
|---|---|
| Document ID | BL-REG-VER-001 |
| Version | 0.1 (draft) |
| Date | 2026-10-20 |
| Status | Draft |
| Owner | TBD (manufacturer) |
| Software | BehaviorLens, `ALGORITHM_VERSION = rules-v0.2`, schema version 2, model `pose_landmarker_lite@59929e1d1ee9` |
| Environment | Python 3.11.17, Linux (cloud container), dependencies exactly as `constraints.txt` |

This report records **what is verified, by which mechanism, and the results observed on 2026-10-20**. Software verification shows that the code does what its requirements say. It is **not clinical validation**: nothing here shows that the suggestions are accurate enough for any clinical purpose (see [clinical_validation_plan.md](clinical_validation_plan.md), which is a proposal and has not been executed). Requirement IDs refer to [software_requirements.md](software_requirements.md); hazard IDs refer to [risk_management.md](risk_management.md).

## 1. Verification activities and results

| Activity | Tool / command | Gate | Result (2026-10-20) |
|---|---|---|---|
| Lint | `ruff check .` (ruff 0.17.1; rules E, F, W, B) | zero findings | **Pass**: "All checks passed!" |
| Static typing | `mypy` (mypy 2.4.0, `strict = true`, pydantic plugin, `src/behaviorlens`) | zero errors | **Pass**: "no issues found in 13 source files". Only `mediapipe` and `imageio_ffmpeg` (no published stubs) use `ignore_missing_imports`; there are no `# type: ignore` comments in `src/` except one `attr-defined` marker on a logging handler attribute. |
| Unit, integration and UI tests | `python -m pytest -q --cov` | all pass | **Pass**: 124 passed, 0 skipped, 0 failed (~10 s). |
| Coverage | pytest-cov 7.1.0 / coverage 7.16.2, line + branch over `src/behaviorlens` | `fail_under = 91` | **Pass**: 91.57 %. The floor is the measured value rounded down. |
| Dependency vulnerabilities | `pip-audit -r constraints.txt --no-deps --disable-pip` (pip-audit 2.10.1) | zero known vulnerabilities | **Pass**: "No known vulnerabilities found" (88 pinned components). See [soup_list.md](soup_list.md). |
| Browser end-to-end | `e2e-verify-streamlit` skill (Playwright + headless Chromium against a live `streamlit run` on a scratch data dir) | all 6 checks pass | **Pass**: `timeline_has_bars` (9), `bar_click_selects_event`, `thumbnails_shown` (2), `video_plays_and_seeks` (readyState 4, seek to 4.0 s), `export_reflects_review`, `no_streamlit_exceptions` (0). The one console "Video source error" is the expected H.264 load in Chromium (LEARNINGS §2). The server log contained only JSON log lines with IDs and counts. |
| Container build and smoke test | CI job `docker` in `.github/workflows/ci.yml` | image builds; health endpoint answers on `127.0.0.1`; non-root UID; pinned model verifies; no traceback in logs | **Not run locally**: the Docker daemon is not available in the authoring environment. The job runs on GitHub-hosted runners on every push. The Dockerfile's privacy and least-privilege settings are asserted statically by `test_dockerfile_keeps_privacy_and_least_privilege_settings`. |
| Real-video performance | `scripts/validate_real_video.py` (tune/holdout protocol) | n/a (measurement, not a gate) | **Not re-run**: this change set doesn't touch rules, thresholds, features or the model (`ALGORITHM_VERSION` unchanged). The latest figures are in [VALIDATION.md](../VALIDATION.md): held-out `standing_transition` recall 0.40 and precision 0.40 (n = 10, 95 % CI 0.17–0.69). |

All gates except the container job and real-video validation run in CI (`.github/workflows/ci.yml`, job `test`) on every push and pull request, with the same pinned versions.

## 2. Test inventory

| File | Tests | Scope |
|---|---|---|
| `tests/test_event_rules.py` | 30 | Rule logic on synthetic `PoseSequence`s: hand raise, standing transition, sustained standing; occlusion, noise, spikes, missing pose, out-of-frame landmarks, multi-person warning, primary-person selection |
| `tests/test_pipeline.py` | 19 | Upload validation, probe limits (short clip, bad fps, too large, corrupted), frame sampling, pose cache, missing model, **real MediaPipe inference** on public adult photos, thumbnails, validation-harness helpers |
| `tests/test_storage.py` | 9 | Persistence across restart, original-proposal preservation, idempotent actions, rerun preserves reviewed work, event validation, delete session, status transitions |
| `tests/test_storage_integrity.py` | 14 | Schema version, WAL, foreign keys, legacy v1 → v2 migration without data loss, rollback of a failed migration, refusal of a newer schema, hash-chain verification, append-only triggers, tamper detection (history edit, history deletion, event edited outside the audit trail, reviewed event with no audit record), deletion log, export chain anchor, path traversal, concurrent duplicate requests, manual-add replay |
| `tests/test_robustness.py` | 16 | JSON logs without paths or filenames, exception-type-only failure logs, redaction, bounded upload copy, 4K resolution limit (probe and mid-stream), decode cap, `PoseSequence` metadata validation, ground-truth CSV limits, model checksum on download and at load, overlay writer cleanup, Dockerfile invariants |
| `tests/test_export.py` | 5 | CSV fields, status filter, JSON structure, no paths/filenames/video, synthetic labelling |
| `tests/test_metrics.py` | 10 | Ground-truth CSV parsing, tolerance matching, one-to-one matching |
| `tests/test_app.py` | 6 | Streamlit AppTest: start-up and disclaimer, demo review flow, loopback binding config, rerun confirmation gate, ground-truth gating, audit-integrity banner (verified and tampered) |
| `tests/test_hooks.py` | 15 | Development guard-rail hooks (not product code) |
| **Total** | **124** | |

Real-inference tests (`tests/test_pipeline.py`) need the pose model and network access to MediaPipe's public sample photos. They skip, rather than fail, when either is missing; in this run none skipped.

## 3. Verification of risk controls added in this change set

| Risk control | Hazard | Verified by |
|---|---|---|
| Versioned schema, ordered transactional migrations, pre-migration backup, refusal of a newer schema | H-07 data loss/corruption | `test_new_database_is_at_latest_schema_with_wal_and_foreign_keys`, `test_legacy_v1_database_is_migrated_without_data_loss`, `test_failed_migration_rolls_back`, `test_database_from_newer_release_is_refused` |
| WAL, `synchronous=FULL`, foreign keys, `BEGIN IMMEDIATE` writes | H-07, H-10 | `test_new_database_is_at_latest_schema_with_wal_and_foreign_keys`, `test_concurrent_duplicate_request_is_applied_once` |
| Append-only, hash-chained `review_history`; `verify_integrity`; export chain anchor; UI banner | H-08 audit-trail tampering | `test_review_history_is_hash_chained_and_verifies`, `test_review_history_is_append_only`, `test_tampering_with_history_is_detected`, `test_editing_events_outside_the_audit_trail_is_detected`, `test_reviewed_event_without_audit_record_is_detected`, `test_json_export_carries_chain_head`, `test_audit_trail_status_is_shown_and_flags_tampering` |
| Deletion log, `secure_delete`, WAL checkpoint on delete | H-13 | `test_delete_session_is_logged_and_keeps_chain_valid`; `secure_delete`/checkpoint verified by code review only |
| Stored-path containment and safe IDs | path traversal | `test_stored_paths_cannot_escape_the_data_dir` |
| Structured logging with redaction | H-12 privacy via logs | `test_logs_are_json_and_contain_no_paths_or_filenames`, `test_failed_analysis_logs_exception_type_only`, `test_redact_removes_paths` |
| Resource limits (bounded copy, 4K, decode cap, CSV row cap, finite values) | H-14 malicious/oversized input | `test_upload_copy_is_bounded`, `test_oversized_resolution_rejected`, `test_mid_stream_oversized_frame_is_refused`, `test_decode_is_capped_independently_of_header`, `test_ground_truth_rejects_non_finite_and_blank`, `test_ground_truth_row_cap` |
| Model supply chain (versioned URL, pinned SHA-256 on download and at load, https only, size cap) | H-05 | `test_model_download_verifies_checksum`, `test_tampered_default_model_is_refused`, `test_pinned_model_matches_downloaded_file` |
| Loopback-only network exposure, including Docker (no checkip call) | H-11 LAN exposure | `test_server_binds_localhost_only`, `test_dockerfile_keeps_privacy_and_least_privilege_settings`; the Streamlit `_print_url` behaviour was checked directly (a wildcard address calls `get_external_ip` unless `logger.hideWelcomeMessage` is set) |

## 4. Known verification gaps

- **Coverage holes:** `demo_generator.py` 70 % (the stick-figure video renderer is exercised only when sample assets are regenerated) and `video_io.py` 84 % (ffmpeg-unavailable fallbacks, transcode failure paths, truncated-tail warning).
- **No dedicated tests** for the 5-minute maximum duration, the 240 fps upper bound, the truncated-tail decode warning, or the 64 MB model download cap.
- **Container image** is verified only in CI, not in this environment.
- **Not tested at all:**
  - concurrency beyond two `Store` instances;
  - power-loss durability (relies on SQLite's documented WAL + `synchronous=FULL` behaviour, which is SOUP);
  - browser compatibility beyond headless Chromium;
  - usability / human factors (no IEC 62366 activities have been performed).
- **Real-video detection performance** is the binding limitation. See [intended_use.md](intended_use.md) and [VALIDATION.md](../VALIDATION.md).

## 5. How to reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt -c constraints.txt
sudo apt-get install -y libegl1 libgles2
.venv/bin/python scripts/download_model.py
.venv/bin/ruff check . && .venv/bin/mypy
.venv/bin/python -m pytest -q --cov
.venv/bin/pip-audit -r constraints.txt --no-deps --disable-pip
.venv/bin/python scripts/generate_soup_list.py     # regenerates soup_list.md with a fresh audit
```
