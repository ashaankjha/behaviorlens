# Intended Use and Limitations

> **DRAFT — requires review by qualified regulatory/QA personnel; not a regulatory submission.**

| Field | Value |
|---|---|
| Document ID | BL-REG-IU-001 |
| Version | 0.1 (draft) |
| Date | 2026-10-20 |
| Status | Draft |
| Owner | TBD (manufacturer) |
| Applies to | BehaviorLens `0.1.0` (`pyproject.toml`), rules `rules-v0.2` (`config.ALGORITHM_VERSION`), pose model `pose_landmarker_lite` (SHA-256 `59929e1d…574a`, `config.MODEL_SHA256`) |

Related documents: [README](README.md) · [Software development plan](software_development_plan.md) · [Software requirements](software_requirements.md) · [Architecture](architecture.md) · [Risk management](risk_management.md) · [Clinical validation plan](clinical_validation_plan.md) · [Verification report](verification_report.md) · [Cybersecurity](cybersecurity.md) · [SOUP list](soup_list.md)

---

## 1. Purpose of this document

This document records how BehaviorLens is **currently** intended to be used and what it must not be used for. It is the reference point for the requirements, risk analysis and verification documents in this folder. It also sketches, as a clearly hypothetical exercise, what a future clinical intended use might look like and what would have to be true before such a claim could be made.

Nothing in this document is a claim of regulatory clearance, conformity, certification or clinical validity.

## 2. Current regulatory and clinical status

**BehaviorLens is a research/demo prototype. It is not for clinical decision-making and does not diagnose.**

- The application displays the banner *"Research/demo prototype. Not for clinical decision-making. Use only synthetic or staged footage with informed consent."* on every page (`config.DISCLAIMER`, verified by `test_app_starts_with_empty_state`).
- Every JSON export carries `config.EXPORT_WARNING`, which states that suggestions describe observable movement only, are not clinically validated, do not represent diagnoses, intent, emotion, behavioral function or treatment recommendations, and require human review.
- No clinical performance study has been performed. The only real-video evidence is a small engineering evaluation on public research datasets ([VALIDATION.md](../VALIDATION.md)), which that report itself describes as "not clinical validation".
- The software has not been placed on any market and is not intended to be at this stage.

## 3. Intended use (current, non-clinical)

BehaviorLens is a local, single-user software tool that helps a human reviewer navigate **staged** video recordings by suggesting time intervals in which specific **visually observable body movements** may occur. The reviewer inspects each suggestion against the video and confirms, rejects or edits it, adds events the tool missed, and exports the reviewed records.

The tool suggests three movement event types, defined geometrically from 2D pose landmarks:

| Event type | Operational definition (rules-v0.2) |
|---|---|
| `hand_raised` | A wrist is at least 0.5 "head units" above the same-side shoulder for ≥ 0.6 s, with the nose, both shoulders and the wrist visible (visibility ≥ 0.5) and in frame. A "head unit" is the vertical nose-to-shoulder-line distance. Both hands raised together form one event. |
| `standing_transition` | A seated-like posture held ≥ 0.8 s followed within 5 s by an upright posture held ≥ 0.8 s, using hip/knee angles, hip-to-knee drop and torso tilt, with both shoulders, hips and knees required to be visible and in frame. |
| `sustained_standing` (optional) | An upright posture held ≥ 2 s. |

These are engineering heuristics. They are **not** behavioral, developmental or clinical categories.

### 3.1 Intended users

| User | Description |
|---|---|
| Primary | A Board Certified Behavior Analyst (BCBA) or behavior-analytics researcher reviewing **staged demonstration footage** in a research or method-development context. |
| Secondary | An ML or software engineer evaluating event-detection methods with reproducible labeled data. |

Users are expected to be able to judge from the video whether a movement occurred. The tool assumes the user, not the software, makes every determination.

### 3.2 Intended input data

- Staged recordings of **consenting adults**, or the bundled **synthetic** fixture (`data_mode = "synthetic"`).
- Local MP4/MOV/AVI files, ≤ 100 MB, 1 s to 5 min long, 1–240 fps, at most 3840×2160 pixels.
- One person in view, ideally front-on with the whole body visible.

Not intended inputs: real patient recordings, recordings of children, footage containing protected health information, or footage of people who have not consented.

### 3.3 Use environment

- A personal workstation running the app locally (`streamlit run app.py`) or in the provided container. The server binds to loopback only (`.streamlit/config.toml` `address = "localhost"`; the container must be published to `127.0.0.1`).
- No network connection is needed after the one-time pose-model download. No cloud processing, telemetry or LLM calls. (In the container, Streamlit's external-IP lookup is suppressed with `--logger.hideWelcomeMessage=true`; see [cybersecurity.md](cybersecurity.md) §5.)
- Single user per data directory. There is no authentication, user management or multi-user access control.
- Not a networked clinical system, not connected to an EHR, and not used in real time during sessions.

## 4. What BehaviorLens does not do

BehaviorLens does **not**:

- diagnose any condition, estimate severity, or screen for any disorder;
- infer emotion, intent, motivation or the function of a behavior;
- classify stereotypy, aggression, self-injury or any clinically defined behavior;
- recommend, select or adjust any treatment or intervention;
- produce calibrated probabilities (`confidence` is always `null`);
- track or attribute events across multiple people;
- operate autonomously: every suggestion is created with `review_status = "pending"` and remains a suggestion until a human acts on it.

## 5. Contraindications (do not use)

Do not use BehaviorLens:

1. To inform any clinical, educational, legal, insurance or care decision about an individual.
2. On recordings of real patients, children, or anyone who has not given informed consent to recording and processing.
3. As the sole or primary source of behavioral data for any record. Exports are reviewer-curated research data only.
4. On scenes with more than one person when attribution matters (see §6.4).
5. Over a network or shared server. The app has no authentication, and exposing it beyond loopback would expose stored videos and review data.

## 6. Known limitations (from real-video evaluation)

All figures come from [VALIDATION.md](../VALIDATION.md) (run date 2026-10-18, `pose_landmarker_lite`, 5 fps, rules-v0.2). The datasets are small public research samples, so every number below has wide uncertainty.

### 6.1 `standing_transition` performance is modest

| Measure (Charades, event level) | Held-out half | All clips |
|---|---|---|
| Recall | **0.40 (4/10)**, 95% Wilson [0.17–0.69] | 0.26 (5/19), [0.12–0.49] |
| Precision | **0.40 (4/10)**, [0.17–0.69] | 0.42 (5/12), [0.19–0.68]; 0.50 (6/12) after visual review by the rule author |
| Recall when legs observable (posture observable in ≥ 30% of frames) | 0.50 (4/8) | 0.42 (5/12) |
| Confused with sitting down | 0 | 0 |

Per-frame posture balanced accuracy on held-out frames is 0.79. The improvement over rules-v0.1 (held-out recall 0.20, precision 0.50) is **not statistically conclusive** at this sample size.

### 6.2 `hand_raised` performance

On 40 UCF101 clips, measured per clip against class labels: **precision 0.77, recall 0.63** (TP 10, FP 3, FN 6, TN 21). PullUps 7/8, JumpingJack 3/8, JumpRope 0/8, BodyWeightSquats 2/8, WalkingWithDog 1/8.

- Arm raises shorter than 0.6 s (for example most JumpingJack raises, about 0.4 s) are not suggested by design.
- MediaPipe can hallucinate a head and shoulders when only a torso is visible, which produced the one confirmed false positive.
- Adjudication of disagreements was done by the same agent that built the rules, so adjudicated figures are indicative only.

### 6.3 Hidden legs make standing events unobservable

When hips or knees are not visible (furniture, desks, blankets, out of frame), posture is marked **unobservable** and no standing event is suggested. This is deliberate: missing evidence is never turned into a guess. The consequence is that real transitions are silently missed in those intervals. In 7 of the 19 Charades stand-up clips, posture was observable in fewer than 30% of frames. The app warns when posture is observable in less than 50% of sampled frames (`pipeline.analyze_pose`), but reviewers must still scan unobservable intervals themselves.

### 6.4 Other documented failure modes

| Condition | Effect |
|---|---|
| **Camera pans / handheld camera** | Produced 2 of the standing false positives. There is no camera-motion detection. |
| **Second person / multi-person scenes** | Events are attributed to the largest detected person (`pose.select_primary`). Attribution can switch between people; one false positive came from a second person. The app warns when more than one person is detected, but does not track identities. |
| **Partial-body framing / close-ups** | Jitter on partially framed standing people produced 3 false positives; torso-only shots can yield hallucinated landmarks. |
| **Side-view cameras** | The hip-to-knee drop cue assumes a roughly frontal view. Side-view behavior has not been evaluated. |
| **Non-ABA footage** | Validation used home (Charades) and sports (UCF101) video, not ABA session rooms. Camera placement, furniture and participants differ. |
| **Model or sampling changes** | Thresholds are coupled to the pose model's visibility calibration. `pose_landmarker_heavy` found 0/19 stand-ups with the same thresholds; 10 fps raised recall but lowered precision. Only the default (lite, 5 fps) has been characterized. The pinned lite model is re-verified against its SHA-256 at load; a model supplied through `BEHAVIORLENS_MODEL_PATH` is not checked against the pin (its hash prefix is recorded in `model_version`). |
| **User-adjusted thresholds** | The UI exposes sliders for visibility threshold, minimum hand-raise duration, dwell and gap bridging. Non-default values have not been validated on real video. |
| **Small samples** | 19 annotated stand-up windows (10 held out) and 40 action clips. All intervals are wide. |
| **Seek precision** | The player re-renders about 1 s before an event; precision depends on browser and keyframes. |

### 6.5 Synthetic demo

The built-in demo session is a programmatic pose sequence with a stick-figure video. It bypasses pose inference and demonstrates only the rules and review workflow. It is labeled SYNTHETIC in the UI and in exports (`data_mode`), and must never be presented as evidence of real-world detection performance.

## 7. User responsibilities

- Review every suggestion against the video; treat unobservable intervals as unreviewed, not as "no event".
- Use only consented staged footage and keep the data directory on an access-controlled machine. Deleting a session in the app overwrites its database rows (`secure_delete`, WAL checkpoint) but does not remove the pre-migration backup (`behaviorlens.db.pre-vN-backup`), external or OS backups and snapshots, or filesystem-level remnants of deleted video files; the app does not encrypt data. Use OS-level disk encryption ([cybersecurity.md](cybersecurity.md) §8).
- Do not alter the configuration to expose the app on a network.

## 8. Hypothetical future clinical intended use (requires validation)

> This section is a **planning exercise only**. It describes a use that BehaviorLens is **not** intended for today and makes no claim that the software is suitable for it. Any such use would require the manufacturer's own regulatory strategy, a formal risk management process, and clinical validation as outlined in [clinical_validation_plan.md](clinical_validation_plan.md).

### 8.1 Possible future statement (illustrative)

*"BehaviorLens is software intended to assist qualified behavior analysts in locating candidate intervals of specified observable gross-motor movements (e.g., hand raising, sit-to-stand transitions) in recorded sessions, for subsequent human review and annotation. All suggestions require confirmation by the qualified user. The software does not diagnose, assess, or recommend treatment."*

### 8.2 Conditions that would have to be met first

| Area | What would be needed |
|---|---|
| Performance | Prospective evaluation on footage representative of the intended setting (camera placement, rooms, participant population), with pre-specified acceptance criteria, adequate sample size and independent annotation. The current Charades/UCF101 evidence would not be sufficient. |
| Population | Explicit decision on whether children or specific populations are included, with ethics approval and consent processes; current development has used adults only. |
| Workflow | Human-factors evaluation showing reviewers do not over-rely on suggestions or treat unobservable intervals as negatives. |
| Multi-person scenes | Person tracking and attribution, or an enforced single-person restriction. |
| Security and privacy | Authentication, encryption at rest, access logging and a documented data-protection approach if real patient data were ever processed (see [cybersecurity.md](cybersecurity.md)). |
| Lifecycle | Full execution of the lifecycle in [software_development_plan.md](software_development_plan.md), including closure of the gaps listed there. |
| Regulatory determination | A documented determination by the manufacturer of whether the software is a medical device in each target jurisdiction, and of its classification. |

Until these conditions are addressed, the current intended use in §3 and the contraindications in §5 apply.
