# Cybersecurity and privacy engineering (draft)

> **DRAFT — requires review by qualified regulatory/QA personnel; not a regulatory submission.**

| Field | Value |
|---|---|
| Document ID | BL-REG-CYB-001 |
| Version | 0.1 (draft) |
| Date | 2026-10-20 |
| Status | Draft |
| Owner | TBD (manufacturer) |

This document records the threat model, implemented mitigations and known gaps for BehaviorLens as built. It is structured after common premarket cybersecurity expectations (threat modelling, SBOM, vulnerability management, security architecture). It is intended to support a future cybersecurity file, and it does not claim conformity with FDA premarket cybersecurity guidance, IEC 81001-5-1 or any other standard. Risks that overlap patient safety or privacy are cross-referenced to [risk_management.md](risk_management.md).

## 1. System context

BehaviorLens is a **single-user, local** Python application: a Streamlit UI (`app.py`) over a core library (`src/behaviorlens/`), SQLite (`data/behaviorlens.db`) and files under `data/sessions/<session_id>/` and `data/cache/`. It runs on the user's workstation, either natively or in the provided Docker image. See [architecture.md](architecture.md).

**There is no authentication or access control.** Anyone who can reach the HTTP port or read `data/` has full access. That is acceptable only for the current labelled use (one person, their own machine, synthetic or staged footage). Authentication, authorisation and per-user audit attribution are **mandatory requirements before any multi-user, shared-workstation or networked deployment**.

## 2. Assets

| Asset | Location | Sensitivity |
|---|---|---|
| Uploaded videos, browser previews, overlays, thumbnails | `data/sessions/<id>/` | High: identifiable images of people (potentially PHI in a clinical setting) |
| Pose landmarks | `data/sessions/<id>/landmarks_*.npz`, `data/cache/*.npz` | Medium: derived biometric-like movement data |
| Review data (events, notes, status) | `events` table | Medium to high: free-text notes may contain identifiers or clinical observations |
| Audit trail | `review_history` (hash-chained), `deletion_log` | Integrity-critical |
| Session metadata, including the sanitised source filename | `sessions` table | Medium: filenames may contain names (never exported) |
| Pre-migration DB backup | `data/behaviorlens.db.pre-vN-backup` | Same as the DB; not removed by Delete session |
| Pose model weights | `assets/models/pose_landmarker_lite.task` | Integrity-critical (pinned SHA-256) |
| Exports | User's downloads folder | Medium: no video, paths or filenames, but notes and the content hash |
| Logs | stderr / Streamlit console | Low by design (redacted) |

## 3. Threat actors

1. **Other users on the LAN** who can reach an exposed port.
2. **Local malware or other OS users** on the same workstation, who can reach the loopback port or read `data/`.
3. **A malicious video file**, crafted to exploit decoders or exhaust resources.
4. **A compromised upstream dependency or model**: PyPI package, model bucket or container base image.
5. **Careless export sharing**: exports, logs or `data/` copies emailed, synced or attached to tickets.

## 4. Trust boundaries

```
 [Browser] --HTTP (loopback)--> [Streamlit server / app.py] --> [behaviorlens core]
                                                                   |-> SQLite + data/ files (local FS)
                                                                   |-> OpenCV/FFmpeg decode (native code; untrusted input)
                                                                   |-> MediaPipe inference (native code; pinned model)
 [Google model bucket] --HTTPS, once, SHA-256 pinned--> assets/models/
 [PyPI] --pip, exact pins--> environment          [User] <--CSV/JSON download-- app
```

- **TB1, browser to server:** HTTP without TLS, which is acceptable only on loopback. No authentication.
- **TB2, uploaded file to decoder:** the file is untrusted, and is checked before and during decoding (§6, Tampering).
- **TB3, filesystem:** the DB and files are trusted by the app. Integrity checks detect, but cannot prevent, outside modification.
- **TB4, supply chain:** model and packages are fetched from the internet at install or build time only.
- **TB5, export:** data leaves the app's control at download.

## 5. Network exposure

- **Loopback by default:** `.streamlit/config.toml` sets `server.address = "localhost"`, verified by `test_server_binds_localhost_only` and by the live check recorded in [AUDIT.md](../AUDIT.md) D1 (listening on 127.0.0.1 only; a LAN connection was refused).
- **No telemetry:** `browser.gatherUsageStats = false` in config, and `--browser.gatherUsageStats=false` in the Docker CMD.
- **Streamlit `checkip.amazonaws.com` call avoided:** in Streamlit 1.66, a headless server without `server.address` fetches its external IP on every start. Setting a non-wildcard address skips that call (AUDIT D1, LEARNINGS M24).
- **Docker:** inside the container Streamlit must bind `0.0.0.0`. Host exposure is controlled by publishing to loopback only: `docker run -p 127.0.0.1:8501:8501 …`. A plain `-p 8501:8501` exposes the app to the network. The CI smoke test uses the loopback form.
- **Docker and the external-IP lookup:** with `--server.address=0.0.0.0`, Streamlit 1.66's `bootstrap._print_url` would GET `checkip.amazonaws.com` to print an "External URL". The Docker `CMD` therefore also passes `--logger.hideWelcomeMessage=true`, which returns before that call (checked by calling `_print_url` with a trapped `get_external_ip`). `test_dockerfile_keeps_privacy_and_least_privilege_settings` asserts the flag, `--browser.gatherUsageStats=false`, the non-root user and the loopback publish instruction are present in the `Dockerfile`. The image itself was not built locally while drafting (no Docker daemon available); it is built and smoke-tested only by the CI `docker` job.
- **Outbound traffic:** only the one-time model download (HTTPS to `storage.googleapis.com`) and package installation. The app makes no LLM calls, cloud uploads or analytics requests.
- **Residual:** a user can override the bind address. No TLS. Any local process can reach the loopback port. Streamlit's own defaults (for example XSRF protection) were not separately assessed.

## 6. STRIDE analysis

| Threat | Example | Mitigations implemented | Residual |
|---|---|---|---|
| **S**poofing | A LAN or local user opens the UI and acts as the reviewer | Loopback bind (§5) | **No authentication.** Local OS users and processes are not distinguished, and audit records carry no user identity. |
| **T**ampering: input | Crafted video exploits the decoder or exhausts resources | Extension allow-list; container signature check (`ftyp` / `RIFF…AVI `); MIME not trusted (`video_io.validate_upload`); 100 MB limit and bounded copy (`pipeline._copy_bounded`, `maxUploadSize = 100`); duration ≤5 min, fps 1–240, 4K pixel limit on the header and on every decoded frame (`probe_video`, `iter_sampled_frames`); `MAX_DECODED_FRAMES`; ffmpeg timeout 600 s; UUID temp names, cleaned up on failure. Tests: `test_validate_upload_rejects_bad_inputs`, `test_upload_copy_is_bounded`, `test_too_large_upload_rejected`, `test_oversized_resolution_rejected`, `test_mid_stream_oversized_frame_is_refused`, `test_decode_is_capped_independently_of_header`, `test_corrupted_video_rejected_and_cleaned_up` | OpenCV/FFmpeg parse untrusted data in native code. No sandbox beyond the container's non-root user. The 5-minute maximum and 240 fps upper bound have no dedicated tests. |
| **T**ampering: stored data | DB rows or files edited outside the app | Append-only triggers and per-session SHA-256 chain on `review_history`; `verify_integrity` (chain plus event-vs-audit check, `PRAGMA integrity_check`, `foreign_key_check`); integrity status in the UI; chain head in the JSON export; `deletion_log`. Tests: `test_tampering_with_history_is_detected`, `test_editing_events_outside_the_audit_trail_is_detected`, `test_review_history_is_append_only`, `test_audit_trail_status_is_shown_and_flags_tampering` (UI banner) | Tamper-*evident* only. An attacker with write access can drop the triggers and recompute the chain. Detection then depends on an externally kept export head. |
| **T**ampering: paths | A crafted ID or stored relpath escapes `data/` | `_check_id` (`[A-Za-z0-9_-]{1,64}`); `Store.path` refuses paths that resolve outside the root; `sanitize_filename`. Test: `test_stored_paths_cannot_escape_the_data_dir` | Low |
| **T**ampering: supply chain | Swapped model or malicious package | Versioned model URL, https only, pinned `MODEL_SHA256`, size cap, atomic replace (`pose.download_model`); Docker build fails on mismatch; CI runs `verify_model()` in the container; a model at the default path is re-hashed at load and refused with `ModelIntegrityError` on mismatch (`MediaPipePoseEstimator.__init__`); `constraints.txt` exact pins; `pip-audit` in CI. Tests: `test_model_download_verifies_checksum`, `test_pinned_model_matches_downloaded_file`, `test_tampered_default_model_is_refused` | A model at `BEHAVIORLENS_MODEL_PATH` (used for full/heavy comparisons) is not checked against the pin; only its hash prefix is recorded in `model_version`. Pip installs are pinned by version, not by hash (`--require-hashes` not used). The base image is not digest-pinned. |
| **R**epudiation | Reviewer denies a decision; record silently changed | Hash-chained `review_history` with before/after JSON, action and timestamp; idempotent `request_id` | No user identity or signatures. Local clock timestamps. |
| **I**nformation disclosure: network | Video viewed from the LAN | Loopback bind; Docker loopback publish | User override; no TLS |
| **I**nformation disclosure: at rest | Stolen laptop, other OS user, backups | Guidance only (§8); Delete session removes rows and files; `PRAGMA secure_delete = ON` on every connection and `PRAGMA wal_checkpoint(TRUNCATE)` after deletion, so deleted rows are overwritten in the DB file and not kept in the WAL | **No app-level encryption.** Deleted data may remain in the pre-migration backup, in OS backups and snapshots, and as filesystem-level remnants of deleted video files (`rmtree` does not shred). `data/` permissions follow the umask. |
| **I**nformation disclosure: exports and logs | Export or log shared | Exports exclude video, paths, filenames and images (`test_exports_contain_no_paths_filenames_or_video`). Logs are JSON with redaction and exception type only (`test_logs_are_json_and_contain_no_paths_or_filenames`, `test_failed_analysis_logs_exception_type_only`, `test_redact_removes_paths`) | Free-text notes are exported verbatim. `content_sha256` links an export to a specific file. |
| **D**enial of service | Huge or long file, decompression bomb | Size, duration, resolution and decode-frame caps; per-frame pixel check; transcode timeout | Single process. A long, valid 5-minute 4K file still takes minutes of CPU. |
| **E**levation of privilege | Exploit in a decoder or dependency | Docker image runs as non-root uid 10001 (verified in CI); no shell or subprocess built from user strings (ffmpeg receives argument lists) | Native-code vulnerabilities in mediapipe, OpenCV or FFmpeg. Native installs run as the user. |

## 7. Update and patch process

1. Monitor advisories (§10). For a fix, bump the affected pin in `constraints.txt`. Never loosen a pin to an unpinned range.
2. Run the full gate: `ruff check .`, `mypy`, `pytest` with coverage, `pip-audit -r constraints.txt`, and the Docker build and smoke test (CI).
3. **Inference-affecting dependencies** are `mediapipe`, `opencv-python-headless`, `opencv-contrib-python` and `numpy`, plus the pose model itself (`MODEL_URL`/`MODEL_SHA256`). Changing any of them also requires a rerun of the `validate-real-video` skill (tune/holdout, compared with the previous version), an update to [VALIDATION.md](../VALIDATION.md), and a review of [risk_management.md](risk_management.md) H-05. A model change is a threshold change (LEARNINGS M22).
4. Changes to the storage schema need a new migration (released migrations are never edited), a migration test, and confirmation that older releases refuse the new schema.
5. Regenerate [soup_list.md](soup_list.md) with `python scripts/generate_soup_list.py`, and record the change, its verification evidence and the regenerated list in the release notes.

## 8. Encryption at rest (guidance)

BehaviorLens does **not** encrypt data itself and will not implement its own cryptography. Protect data with OS or platform encryption:

- **Linux:** LUKS/dm-crypt full-disk or volume encryption for the disk that holds `data/`.
- **macOS:** FileVault.
- **Windows:** BitLocker.
- **Docker:** put the `behaviorlens-data` volume (mounted at `/data`) on an encrypted host filesystem, or use an encrypted volume driver.
- **Permissions:** restrict `data/` to the running user (for example `chmod 700 data`). Do not keep `data/` in synced folders (cloud drives) or world-readable locations.
- **Backups:** encrypt any backup of `data/`. This includes **`behaviorlens.db.pre-vN-backup`**, which a schema upgrade creates next to the database. It contains every session as of the upgrade, and **Delete session does not remove it**. Delete it once the upgrade is confirmed good, if retention rules allow.
- **Deletion:** Delete session removes DB rows and session files. Storage connections set `PRAGMA secure_delete = ON`, so deleted rows are overwritten in the database file, and `delete_session` runs `PRAGMA wal_checkpoint(TRUNCATE)` afterwards, so they are not kept in the WAL. Remaining residuals: the pre-migration backup file, OS backups or snapshots (outside the app's control), and filesystem-level remnants of deleted video and derived files (`shutil.rmtree` unlinks, it does not shred; on SSDs and copy-on-write filesystems overwriting would not be reliable anyway). Disk encryption is the primary control.

## 9. SBOM

- The current component inventory is [soup_list.md](soup_list.md), generated by `scripts/generate_soup_list.py` from `constraints.txt` (exact versions installed by `pip install -r requirements-dev.txt -c constraints.txt`) plus a pip-audit run. On 2026-10-20: pip-audit 2.10.1, 88 components, no known vulnerabilities. This is a human-readable inventory, not a machine-readable SBOM.
- **Recommended future CI artifact:** a machine-readable CycloneDX SBOM built from the same pins, for example:
  - `pip-audit -r constraints.txt --no-deps --disable-pip --format cyclonedx-json --output sbom.cdx.json`, or
  - `cyclonedx-py requirements constraints.txt -o sbom.cdx.json` (cyclonedx-bom), or `cyclonedx-py environment` inside the built image.
  - Also record the model artifact (`pose_landmarker_lite`, URL, SHA-256), the Docker base image digest and OS packages (for example with a container SBOM tool).
- No machine-readable SBOM is produced today.

## 10. Vulnerability monitoring

- **Implemented:** the CI job runs `pip-audit -r constraints.txt --no-deps --disable-pip` on every push and pull request. A known vulnerability fails the build.
- **Gaps:** there is no scheduled run, so a new advisory against unchanged pins goes unnoticed until the next push. Recommendation: a weekly scheduled workflow. OS packages in the Docker image are not scanned. Native libraries bundled in wheels (FFmpeg in `imageio-ffmpeg`, the codecs in OpenCV) are only covered to the extent that their Python packages have advisories.
- Triage: assess exploitability in the local, single-user context, record the decision, and patch through §7.

## 11. Coordinated vulnerability disclosure

*Placeholder.* The manufacturer must publish a disclosure policy (contact address, PGP key or secure form, acknowledgement and remediation timelines, safe-harbour statement) and add a `SECURITY.md`. No such channel exists yet.

## 12. Logging policy

- Structured JSON lines to **stderr only** (`logging_setup.configure_logging`). The module writes no log files and sends nothing anywhere.
- Records hold an event name plus non-identifying fields: IDs, counts, durations, codec, fps, `model_version`, `algorithm_version`, and exception **type**.
- `RedactingFilter` removes absolute and home paths, Windows paths, the data directory and the project root. It truncates fields to 300 characters, drops binary, array and large values, and strips tracebacks (`exc_info`, `stack_info`).
- Frames, landmarks, filenames, notes and video content are never logged.
- Level comes from `BEHAVIORLENS_LOG_LEVEL` (default INFO).
- Residual: Streamlit's and third-party libraries' own log output is not filtered by this module. If a deployment captures stderr to disk, retention and access must be governed there.

## 13. Requirements before any multi-user or networked deployment

1. Authentication and role-based authorisation, with user identity recorded in audit records.
2. TLS for any non-loopback access, behind a hardened reverse proxy.
3. Encryption at rest enforced by the deployment, plus key management.
4. A scheduled vulnerability scan, a CycloneDX SBOM per release, and hash-pinned installs.
5. Removal or gating of `BEHAVIORLENS_MODEL_PATH` in production builds (default-path models are already re-verified at load; override-path models are not).
6. A HIPAA/GDPR data-protection impact assessment, and retention and deletion procedures that cover backups and the pre-migration copy.
7. A penetration test and a formal threat-model review by qualified security personnel.
