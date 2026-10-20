# Software Development Plan

> **DRAFT — requires review by qualified regulatory/QA personnel; not a regulatory submission.**

| Field | Value |
|---|---|
| Document ID | BL-REG-SDP-001 |
| Version | 0.1 (draft) |
| Date | 2026-10-20 |
| Status | Draft |
| Owner | TBD (manufacturer) |
| Applies to | BehaviorLens `0.1.0`, `rules-v0.3`, schema version 2 (`storage.SCHEMA_VERSION`) |

Related documents: [Intended use](intended_use.md) · [Software requirements](software_requirements.md) · [Architecture](architecture.md) · [SOUP list](soup_list.md) · [Risk management](risk_management.md) · [Verification report](verification_report.md) · [Cybersecurity](cybersecurity.md) · [Clinical validation plan](clinical_validation_plan.md)

---

## 1. Purpose and scope

This plan describes how BehaviorLens software is developed, verified, released and maintained. It is **structured after** the activities of IEC 62304 (medical device software lifecycle processes) so that, if the manufacturer later decides to pursue a clinical intended use, existing practice can be mapped to a recognized lifecycle and the remaining gaps are visible. It is not a claim of compliance with IEC 62304 or any other standard.

For each lifecycle activity the plan records (a) what the activity requires, (b) what the repository already does, and (c) gaps. Gaps are consolidated in §15. Section headings cite the corresponding IEC 62304 clause numbers for orientation only.

Scope: everything in the repository that ships or affects shipped behavior: `app.py`, `src/behaviorlens/`, `scripts/download_model.py`, `.streamlit/config.toml`, `Dockerfile`, dependency manifests, and the tests and validation scripts that produce verification evidence.

## 2. Proposed software safety classification

**Proposed class: B** (if the software were ever used clinically as decision support with mandatory human review). The classification is the manufacturer's decision, made through its risk management process ([risk_management.md](risk_management.md)); this section is an engineering proposal to feed that process.

### 2.1 Rationale

IEC 62304 assigns class A when the software cannot contribute to a hazardous situation, or can only contribute to one that does not result in unacceptable risk after external risk controls; class B when it can contribute to a hazardous situation resulting in unacceptable risk where the possible harm is non-serious injury; class C when serious injury or death is possible.

- **Current use (research/demo, staged consenting adults):** there is no patient and no care decision, so the software does not contribute to patient harm. The relevant hazards today are privacy and data-integrity hazards (exposure of footage, corrupted review records), handled as security and quality matters.
- **Hypothetical clinical decision-support use:** a missed or false movement suggestion that a reviewer accepts could distort behavioral data that later informs an intervention plan. Plausible harm is indirect and non-serious (for example an inappropriate adjustment of a behavior program, delayed recognition of a change, or loss of confidentiality of session video). That points to **class B**.
- **Why not C:** the tool suggests generic gross-motor movements only; it does not detect self-injury, elopement, seizures or other safety-critical events, and it is not used in real time. Serious injury would require an implausible chain of failures plus independent clinical misjudgment. This argument depends on the intended use staying narrow and must be re-assessed if event types change.
- **Why A is only arguable:** class A would require showing that no hazardous situation can result in unacceptable risk, typically by crediting **external** risk controls. Mandatory human review is the main candidate, but known failure modes (hidden legs → silent misses, second-person attribution, camera pans) produce plausible-looking outputs that a reviewer may not catch, and over-reliance is a recognized human-factors risk. Until a usability evaluation shows that human review reliably catches these errors, crediting it to reach class A is weak.

### 2.2 Consequences of class B for this plan

Class B adds, relative to A: documented architecture with SOUP identification, detailed design to the level needed for unit verification, unit verification acceptance criteria, integration testing, and regression testing on change. These are addressed below.

## 3. Software development planning (5.1)

| Element | Current practice | Gap |
|---|---|---|
| Plan and lifecycle model | Iterative development from [PRD.md](../PRD.md); vertical slice first, then P1 features (PRD §10). | No formally approved plan before this draft; no defined release milestones. |
| Standards, methods, tools | Python 3.11, ruff (`pyproject.toml`), mypy strict on `src/behaviorlens`, pytest, pytest-cov, pip-audit; CI in `.github/workflows/ci.yml`. None open for tool pinning: ruff, mypy 2.4.0, pytest, pytest-cov 7.1.0, coverage 7.16.2 and pip-audit 2.10.1 are pinned in `constraints.txt` (regenerated by `pip freeze`). CI upgrades `pip` itself without a pin. |
| Development knowledge base | [LEARNINGS.md](../LEARNINGS.md) records mistakes (M1–M25) and decisions; CLAUDE.md requires appending to it. | Not a controlled document (no review/approval). |
| Roles | A primary development session plus child sessions for audits and experiments on separate branches, merged after review (CLAUDE.md *Parallel cloud sessions*). | No named human roles (developer, reviewer, QA, release authority). |

## 4. Software requirements analysis (5.2)

- **Current practice:** requirements originate in [PRD.md](../PRD.md) (FR1–FR7, §§7–11) and are traced to code and tests in [PRD_TRACEABILITY.md](../PRD_TRACEABILITY.md). This folder adds [software_requirements.md](software_requirements.md) with stable `SRS-*` IDs covering functional, performance/resource, data-integrity, privacy/security and usability requirements, each mapped to verification.
- **Risk-control requirements:** several requirements are already risk controls (unobservable-not-guessed, `confidence = null`, loopback binding, export exclusions, append-only audit trail). They should be tagged against hazards in [risk_management.md](risk_management.md).
- **Gaps:** no formal requirements review/approval record; data-integrity and robustness requirements added on 2026-10-20 are not yet in PRD_TRACEABILITY; no requirement-level acceptance criteria for detection performance (VALIDATION.md reports results, not pass/fail criteria).

## 5. Software architectural design (5.3)

- **Current practice:** [architecture.md](architecture.md) describes components, data flows, storage layout, trust boundaries and SOUP boundaries. The core is UI-independent by rule (CLAUDE.md; `pipeline.analyze_pose` runs on any `PoseSequence` without video), which supports unit-level verification of the rules on synthetic data.
- **Segregation:** logical (module) segregation only; all components run in one Python process. No segregation is relied on as a risk control.
- **Gaps:** architecture review not recorded; no explicit interface specification for SOUP failure behavior beyond the wrappers' exception types.

## 6. Software detailed design (5.4)

- **Current practice:** module and function docstrings document behavior and the reasons for thresholds (`config.RuleParams` comments cite their data source; `features.posture_features` documents each cue). Data structures are typed (`models.py`, pydantic `Event`, `PoseSequence` invariants).
- **Gaps:** no separate detailed-design document per unit. For class B this can likely be satisfied by the architecture document plus docstrings, if the manufacturer accepts that approach.

## 7. Unit implementation and verification (5.5)

| Element | Current practice |
|---|---|
| Coding standard | ruff rules `E, F, W, B`, line length 120; mypy `strict = true`; enforced in CI and by a PostToolUse hook on edited `.py` files. |
| Unit tests | `tests/test_event_rules.py` (synthetic sequences via `conftest.make_seq`), `test_storage.py`, `test_storage_integrity.py`, `test_export.py`, `test_metrics.py`, `test_robustness.py`. Every rule change needs a synthetic-sequence test (CLAUDE.md). |
| Acceptance criteria | Tests pass in CI; lint and type checks pass. |
| Coverage | `pytest --cov` with branch coverage in CI; `pyproject.toml` `[tool.coverage.report] fail_under = 91` (measured 91.57% line+branch on 2026-10-20), so CI fails below the floor. |

**Gaps:** no code-review checklist; unit acceptance criteria are implicit in test assertions rather than documented.

## 8. Software integration and integration testing (5.6)

- **Current practice:** `tests/test_pipeline.py` exercises upload → probe → pose (fake estimator and real MediaPipe) → rules → storage → export, including the pose cache; real-inference tests skip when offline or when the model is missing. `tests/test_app.py` drives the Streamlit script headlessly with `AppTest`. The `e2e-verify-streamlit` skill drives the real UI in Chromium with Playwright.
- **Gaps:** e2e browser checks are run on demand, not in CI; real-inference tests can silently skip in CI if photo downloads fail; no integration test plan document.

## 9. Software system testing (5.7)

- **Current practice:** PRD §11 acceptance tests are tracked in PRD_TRACEABILITY; a clean-room audit ([AUDIT.md](../AUDIT.md)) reproduced setup, tests and every validation number from a fresh clone. CI's `docker` job builds the image and smoke-tests the health endpoint, non-root user, model checksum and absence of tracebacks.
- **Real-video performance testing:** `scripts/validate_real_video.py` with a fixed tune/holdout split ([VALIDATION.md](../VALIDATION.md)).
- **Gaps:** no system test protocol/report with pass/fail criteria per requirement (see [verification_report.md](verification_report.md)); no usability (human factors) testing.

## 10. Software release (5.8)

- **Current practice:** `pyproject.toml` version `0.1.0`; `ALGORITHM_VERSION` and the pose model version (`name@sha256[:12]`) are recorded in every analysis and export; CI must pass.
- **Gaps:** no release procedure, tags or release notes; no documented list of known residual anomalies per release; no archived build artifacts (container image digest, constraint set) tied to a version.

## 11. Software maintenance (6)

- **Current practice:** dependency vulnerability scan (`pip-audit -r constraints.txt`) in CI; LEARNINGS logs field and development findings; the pinned model cannot change silently (§13.1).
- **Gaps:** no maintenance plan defining how user feedback is collected, triaged and fed into problem resolution; no defined review cadence for SOUP security advisories outside CI runs.

## 12. Software problem resolution (9)

- **Current practice:** problems are recorded as LEARNINGS mistakes with *symptom → fix → lesson*; audit discrepancies (D1–D7) were tracked to resolution with verifying evidence in AUDIT.md; regression tests accompany fixes (e.g. `test_server_binds_localhost_only` for D1).
- **Gaps:** no problem-report form with severity, safety impact assessment and trend analysis; no link from problem reports to risk management.

## 13. Configuration management (8) and change control

### 13.1 Configuration items and identification

| Item | Mechanism |
|---|---|
| Source, docs, tests, scripts | git. Child sessions work on `audit/`, `experiment/` or `feature/` branches and never push to `main`; the parent session reviews, tests and merges. |
| Python dependencies | `requirements.txt` (lower bounds), `requirements-dev.txt`, and `constraints.txt` with **exact** tested versions (installed with `-c constraints.txt` by README, CI, Dockerfile and SessionStart hook). |
| Algorithm | `config.ALGORITHM_VERSION` (`rules-v0.3`), bumped when thresholds change (validate-real-video skill step 6); stored on every event and analysis. |
| Rule parameters | Full `RuleParams` stored as `settings_json` on every analysis and exported. |
| Pose model | Versioned URL (`…/float16/1/…`) plus pinned `config.MODEL_SHA256`; download fails on mismatch (`pose.download_model`, `test_model_download_verifies_checksum`); a default-path model is re-verified at load (`test_tampered_default_model_is_refused`); `model_version = name@sha256[:12]` stored per analysis. |
| Database schema | `schema_version` table and ordered, append-only `storage.MIGRATIONS`; backup before upgrade; databases from newer releases are refused (`test_database_from_newer_release_is_refused`). |
| Container | `Dockerfile` on `python:3.11-slim` with pinned Python packages. |

### 13.2 Change control

1. Changes are made on a branch and must pass CI (ruff, mypy, pytest with coverage, pip-audit, Docker smoke test) before merge. The Docker image is built and smoke-tested only by the CI `docker` job; this plan does not rely on local Docker builds.
2. **Rule, threshold, feature or pose-model changes** must follow the `validate-real-video` skill: thresholds are selected on the **tune** split only, the **held-out** split is scored once and reported against the previous version, `docs/VALIDATION.md` is updated with recomputed numbers and Wilson intervals, `ALGORITHM_VERSION` is bumped, and a synthetic regression test is added to `tests/test_event_rules.py` (CLAUDE.md *Conventions*; LEARNINGS M15, M19, M22).
3. A model swap counts as a threshold change, because visibility calibration differs between models (M22).
4. Schema changes are new migrations only; released migrations are never edited.
5. Traceability rows must cite tests whose assertions cover the claim (M25).

### 13.3 Gaps

No formal change request record or impact analysis template; no approval signature beyond merge; `RuleParams.from_dict` silently ignores unknown keys (the validation script's `parse_params` rejects them, but the library does not); `pip` itself is upgraded unpinned in CI; container base image is pinned by tag, not digest.

### 13.4 SOUP management

SOUP items are listed with versions, purpose, known anomalies and failure handling in [soup_list.md](soup_list.md). Current controls: exact pins in `constraints.txt`; checksum-pinned model; `pip-audit` in CI; the SOUP list is generated by `scripts/generate_soup_list.py` from `constraints.txt` plus a pip-audit run (pip-audit 2.10.1: 88 components, no known vulnerabilities on 2026-10-20); wrapper modules isolate each SOUP (see [architecture.md](architecture.md) §6). Known SOUP anomalies are recorded in LEARNINGS §2 (e.g. Streamlit's default `0.0.0.0` bind and external IP lookup, MediaPipe visibility calibration differences, two `cv2` wheels from mediapipe). **Gaps:** no documented evaluation of SOUP anomaly lists per version; the ffmpeg binary bundled by `imageio-ffmpeg` is not separately version-tracked.

## 14. Risk management integration (7)

Risk management is performed per [risk_management.md](risk_management.md) (structured after ISO 14971). Software contributes by: implementing risk controls as requirements (tagged in [software_requirements.md](software_requirements.md)); verifying each control with a named test; re-running the risk analysis when changes touch rules, storage, export, network behavior or SOUP; and recording residual anomalies at release.

## 15. Consolidated gap list

| # | Gap | Proposed action |
|---|---|---|
| G1 | No approved plan, roles or release authority | Name roles; approve this plan. |
| G2 | Closed (2026-10-20): `fail_under = 91` (measured 91.57%) is set in `pyproject.toml` and enforced by `pytest --cov` in CI. | Raise the floor as coverage improves. |
| G3 | Closed (2026-10-20): mypy 2.4.0, pytest-cov 7.1.0, coverage 7.16.2 and pip-audit 2.10.1 are pinned in `constraints.txt`. | None. |
| G4 | E2E browser test not in CI | Add a CI job using the skill's driver. |
| G5 | No detection-performance acceptance criteria | Define in [clinical_validation_plan.md](clinical_validation_plan.md). |
| G6 | No release procedure, tags or anomaly list | Define release checklist and tag releases. |
| G7 | Problem reports lack safety assessment | Add a problem-report template linked to risk management. |
| G8 | No usability evaluation | Plan formative/summative human-factors work. |
| G9 | New integrity/robustness requirements missing from PRD_TRACEABILITY | Add rows or reference [software_requirements.md](software_requirements.md). |
| G10 | Partly closed (2026-10-20): `MediaPipePoseEstimator.__init__` re-verifies a model at `config.default_model_path()` against `MODEL_SHA256` and raises `ModelIntegrityError` on mismatch (`test_tampered_default_model_is_refused`). A model at `BEHAVIORLENS_MODEL_PATH` is not checked against the pin; only its hash prefix is recorded in `model_version`. | Pin or flag override models. |
| G11 | Closed (2026-10-20): `test_decode_is_capped_independently_of_header` and `test_mid_stream_oversized_frame_is_refused` (`tests/test_robustness.py`). Still without dedicated tests: truncated-tail warning, 5-min maximum duration, 240 fps upper bound, 64 MB model cap. | Add unit tests for the remaining limits (see [software_requirements.md](software_requirements.md)). |
| G12 | Base image pinned by tag only | Pin `python:3.11-slim` by digest. |
