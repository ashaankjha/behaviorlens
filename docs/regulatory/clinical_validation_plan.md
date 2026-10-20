# Clinical validation plan (protocol proposal)

> **DRAFT — requires review by qualified regulatory/QA personnel; not a regulatory submission.**

| Field | Value |
|---|---|
| Document ID | BL-REG-CVP-001 |
| Version | 0.1 (draft) |
| Date | 2026-10-20 |
| Status | Draft |
| Owner | TBD (manufacturer) |

> **Status: PROPOSAL ONLY. This protocol has NOT been executed.** No participants have been recruited, no clinical recordings exist, and no ethics approval has been sought. BehaviorLens has **no clinical validation**. The figures in §3 are engineering evidence from public research clips, not clinical performance.

This plan is intended to support a future clinical performance evaluation. A qualified clinical/regulatory lead, a biostatistician and an ethics committee or IRB must review and finalise it before any data collection.

## 1. Objectives

**Primary objective.** Estimate the event-level sensitivity and positive predictive value (PPV) of BehaviorLens candidate suggestions for `standing_transition` and `hand_raised`, against a blinded expert reference standard, in recordings made under the intended camera protocol.

**Secondary objectives.**
1. Per-frame agreement between the rule states (seated-like / standing-like / ambiguous / unobservable; hand raised / not / unobservable) and reference annotation.
2. Rate of *unobservable* frames and events under the intended camera protocol, and how often an annotated event falls in an unobservable stretch.
3. Review efficiency: time to a complete reviewed event record with BehaviorLens suggestions versus manual scrubbing, and agreement of each workflow's output with the reference standard.
4. Reviewer behaviour: the proportion of suggestions confirmed without replaying the clip (input to the usability engineering file; see [risk_management.md](risk_management.md) H-16).
5. Performance in pre-specified subgroups (§9).

## 2. Intended-use claim under evaluation

The claim to be tested, worded after [intended_use.md](intended_use.md) and to be finalised by the manufacturer:

> BehaviorLens suggests time-stamped candidate occurrences of defined, visually observable movements (hand raised above shoulder level for at least 0.6 s; transition from seated to standing) in single-person video recorded with a fixed camera, for review, correction and confirmation by a qualified reviewer. It does not diagnose, infer intent, emotion or behavioral function, or recommend treatment, and its suggestions are not an exhaustive record.

The study evaluates a **review aid**. It does not evaluate autonomous detection. Any claim about time saved depends on secondary endpoint 3.

## 3. Current engineering evidence and its limits

From [VALIDATION.md](../VALIDATION.md) (`pose_landmarker_lite`, 5 fps). Figures below are rules-v0.2 (2026-10-18); the shipped rules-v0.3 has held-out recall 0.20 (2/10) and precision 0.40 (2/5), see VALIDATION *rules-v0.3*:

| Measure | Result | Limits |
|---|---|---|
| `standing_transition`, Charades held-out half | Recall 0.40 (4/10, Wilson 95% CI 0.17–0.69); precision 0.40 (4/10) | 10 held-out events. Home videos, not ABA settings. Incomplete annotations. |
| `standing_transition`, all clips | Recall 0.26 (5/19, CI 0.12–0.49); precision 0.42 (5/12) | Includes the tuning half |
| Per-frame posture, held-out balanced accuracy | 0.79 | Activity-derived labels |
| `hand_raised`, UCF101 per clip | Precision 0.77, recall 0.63 vs class labels (TP 10, FP 3, FN 6, TN 21) | Class labels are not operational definitions. Adjudicated by the developer, not blinded raters. |

Key limitations: tiny samples; non-clinical settings; no blinded expert reference standard; most misses are unobservable legs; the remaining false positives come from camera pans, partial framing and a second person. Person tracking and camera-motion detection are **planned, not in this branch**. This evidence justifies running a study. It does not justify any performance claim.

## 4. Study design

- **Type:** prospective, non-interventional diagnostic-accuracy-style study with blinded reference annotation. The design is structured after STARD reporting principles. Results do not influence care.
- **Phase A, staged recordings (first):** adult volunteers (≥18 years) act out scripted sequences that include target movements, near-miss distractors (arms at shoulder height, sitting down, lunges, partial standing) and pre-specified challenge conditions (occlusion by furniture, a second person entering, lighting changes). Phase A tests the camera protocol and analysis pipeline, and locks the algorithm.
- **Phase B, clinical recordings (later, separately approved):** recordings from real ABA sessions under the same camera protocol, only after Phase A acceptance criteria are met and the regulatory strategy is set.
- **Ethics:** IRB or ethics committee approval before each phase. Written informed consent covering recording, local processing, annotation, retention and the right to withdraw. **No minors** unless a separate protocol with parental consent, child assent and specific ethics approval is approved. Phase B data counts as health data, subject to HIPAA/GDPR assessment (see [cybersecurity.md](cybersecurity.md)).
- **Sites:** at least two sites (staged lab plus clinic) to reduce single-room bias. Each site must have a data-protection plan before enrolment.

## 5. Setting and camera protocol

| Item | Requirement (proposed) |
|---|---|
| Camera | Fixed on a tripod or wall mount; **no panning or handheld** (pans caused false positives in VALIDATION). |
| View | Roughly frontal, or ≤45° off-axis; full body including knees and ankles in frame for the whole session. |
| Height | About 1.0–1.5 m. Record the angle and distance for each recording. |
| Format | MP4/H.264, ≥720p, 25–30 fps constant frame rate, ≤5 min per clip (`config.MAX_DURATION_S`), ≤100 MB. |
| Scene | One target participant. Other people allowed only in the pre-specified multi-person subgroup, recorded as such. |
| Lighting | Even, no strong backlight. Record the conditions. |
| Deviations | Logged per clip (camera bumped, participant left the frame, and so on). Clips are analysed, not dropped, and deviations are reported. |

## 6. Reference standard

- **Annotators:** at least **2 independent Board Certified Behavior Analysts (BCBAs)**, trained on the written operational definitions below. They are blinded to BehaviorLens output, to each other's annotations and to the study hypotheses where feasible.
- **Operational definitions** (to be finalised and piloted):
  - *Hand raised:* at least one wrist above the same-side shoulder by clearly more than shoulder-to-chin distance, sustained ≥0.6 s. Onset and offset are marked to 0.1 s.
  - *Standing transition:* a change from a seated position (weight on a seat surface) to standing upright with weight on the feet. Onset is when the buttocks leave the seat; offset is when upright stance is reached.
  - Each event also gets an *observability* flag: the annotator records whether the relevant body parts were visible.
- **Tool:** annotation with a neutral tool (not BehaviorLens) to avoid anchoring.
- **Adjudication:** a third BCBA, blinded to BehaviorLens output, resolves disagreements: events marked by only one annotator, or matched events whose onsets differ by more than the tolerance. The adjudicated set is the reference standard.

## 7. Inter-rater reliability

- **Event level:** interval-by-interval interobserver agreement (IOA) and exact-count-per-interval IOA, using 10 s intervals, plus event matching at the endpoint tolerance (§8).
- **Frame level:** Cohen's kappa on per-second state labels (seated/standing/other; raised/not).
- **Target:** IOA ≥ 0.80 and kappa ≥ 0.80 per event type, measured on a pilot of ≥10% of recordings before main annotation. Below target, definitions are refined and annotators retrained, and the pilot clips are re-annotated. The pilot clips are not used to tune the algorithm.
- Report raw (pre-adjudication) agreement in all cases.

## 8. Endpoints

| Endpoint | Definition |
|---|---|
| **Primary 1:** event-level sensitivity, per event type | TP / (TP + FN). A suggestion is a TP when it overlaps a reference event of the same type, with onset within ±1.0 s (tolerance pre-specified; sensitivity analyses at ±0.5 s and ±2.0 s). One-to-one matching (`metrics.match_events` logic, start and end within tolerance, or the overlap rule used in VALIDATION; one rule to be chosen and locked before unblinding). |
| **Primary 2:** event-level PPV, per event type | TP / (TP + FP) |
| Secondary: per-frame agreement | Balanced accuracy and Cohen's kappa of rule states vs reference, on frames the reference marks observable |
| Secondary: unobservable rate | Proportion of reference events falling in stretches that BehaviorLens marks unobservable |
| Secondary: review time saved | Paired comparison of reviewer minutes per recording-minute (assisted vs manual, crossover with washout and counterbalanced order), with the accuracy of each final record vs the reference |
| Secondary: confirm-without-replay rate | From app interaction logs (counts only; no content) |

Sensitivity is also reported **conditional on observability**, but the primary analysis counts unobservable misses as misses.

## 9. Subgroup analyses (pre-specified)

- Camera angle: frontal (≤20°) vs oblique (20–45°).
- Occlusion: legs or wrists occluded for >30% of the event window vs not.
- Multi-person: a second person visible during the event vs single person.
- Seat type (chair, sofa, floor) and clothing (loose vs fitted lower garments), as recorded.
- Site.

Subgroups are descriptive with 95% CIs. No subgroup acceptance criteria unless the manufacturer adds them.

## 10. Sample-size justification

**Precision-based estimate for sensitivity.** To estimate a proportion *p* with 95% CI half-width *d*:

n = z² · p(1 − p) / d² = 1.96² × 0.80 × 0.20 / 0.10² = 3.8416 × 0.16 / 0.01 = 61.5 → **62 reference-positive events per event type**.

At *p* = 0.50 (closer to the current stand-up recall of 0.40), the worst case is 1.96² × 0.25 / 0.01 = **97 events**.

**Clustering.** Events are nested in participants. Inflate with a design effect DE = 1 + (m − 1) · ICC, where *m* is the mean number of events per participant. With *m* = 5 and an assumed ICC = 0.10, DE = 1.4: 62 × 1.4 = **87 events (about 18 participants)**, or 97 × 1.4 = **136 events (about 28 participants)** in the worst case. Estimate the ICC in Phase A and re-check *n* before Phase B.

**Hypothesis-test alternative (illustrative).** To show that sensitivity exceeds a performance goal p₀ = 0.70 when the true value is p₁ = 0.85, with one-sided α = 0.05 and 80% power:

n = [1.645·√(0.70·0.30) + 0.842·√(0.85·0.15)]² / 0.15² = (0.754 + 0.301)² / 0.0225 ≈ **50 events**, before the design effect (×1.4 ≈ 70).

**PPV** needs the same number of *suggested* events. Plan recruitment so that the expected number of suggestions also reaches the target.

**Comparison with current evidence.** The held-out set in VALIDATION has **n = 10** stand-up events, giving a CI half-width of about ±0.26 (0.17–0.69). The proposed study needs about **6–14× more events** per type for a ±0.10 half-width.

**Over-recruitment.** Add about 20% for withdrawals, unusable recordings and annotation failures. Script Phase A so that each participant performs a pre-specified number of target events (for example 5 stand-ups and 5 hand raises) and distractors.

## 11. Pre-specified acceptance criteria (placeholders)

These criteria are **placeholders for the manufacturer**. They must be justified from clinical need, the state of the art and the risk analysis before the study starts.

| Endpoint | Proposed criterion (placeholder) |
|---|---|
| `hand_raised` sensitivity | Lower bound of the 95% CI ≥ **[X = e.g. 0.70]** |
| `hand_raised` PPV | Lower bound of the 95% CI ≥ **[Y = e.g. 0.70]** |
| `standing_transition` sensitivity | Lower bound of the 95% CI ≥ **[X₂]** |
| `standing_transition` PPV | Lower bound of the 95% CI ≥ **[Y₂]** |
| Reference-standard reliability | IOA and kappa ≥ 0.80 (prerequisite for interpreting the study) |
| Assisted review | Final-record accuracy non-inferior to manual review (margin **[Δ]**) |

If a criterion fails, the outcome is reported as a failure. Thresholds or definitions are not changed after unblinding.

## 12. Algorithm lock and unblinding discipline

The same tune/holdout discipline as [VALIDATION.md](../VALIDATION.md) and LEARNINGS M19 applies:

1. Before any study data are analysed, **freeze** `config.ALGORITHM_VERSION`, every `RuleParams` default, the sampling rate, `config.MODEL_SHA256` / `model_version`, and the pinned `constraints.txt`. Record the git commit and a container image digest in the statistical analysis plan (SAP).
2. Phase A may set aside a deterministic **tune split** (hashed by participant, not clip, to avoid leakage) for one round of threshold selection. The **evaluation split** stays sealed until the lock in step 1, and is scored exactly once.
3. Disable the UI threshold sliders for the study build, or verify each analysis's stored `settings` against the locked values ([risk_management.md](risk_management.md) H-06).
4. Any change after unblinding produces a new algorithm version that needs a new evaluation set.
5. An independent analyst runs the locked scoring script, and its output is archived with the data.

## 13. Data management and privacy

- **Local processing only:** analysis runs on site workstations, with no cloud upload, telemetry or LLM calls (product constraint). Workstations use full-disk encryption and loopback binding ([cybersecurity.md](cybersecurity.md)).
- **De-identification:** filenames are coded IDs, never names. Video is identifiable by nature, so access is limited to the study team. Exports (CSV/JSON) contain no video, paths or filenames, but free-text notes must not contain identifiers (study SOP).
- **Data flow:** recordings → encrypted site storage → BehaviorLens (assisted arm) and a separate annotation tool (reference arm) → coded event tables → central analysis. Only coded event tables leave the site.
- **Retention:** video is retained for the period the approved protocol and local law require, then securely deleted. Note that BehaviorLens **Delete session** does not remove OS backups or the `behaviorlens.db.pre-vN-backup` file (see the risk file, H-13).
- **Audit:** keep each JSON export's `audit.review_history_chain_head_sha256` in the study master file so that later alteration of review data is detectable.

## 14. Statistical analysis plan (outline)

- Sensitivity and PPV with 95% CIs. Use a cluster-robust method (for example a logit GEE with exchangeable correlation, or a participant-level cluster bootstrap with 2,000 resamples). Report Wilson intervals without clustering for comparison with VALIDATION.
- Per-frame kappa with bootstrap CIs.
- Review time: paired mixed-effects model (participant and reviewer random effects, order effect).
- Missing data: unusable recordings are reported with reasons. The primary analysis covers all recordings with a reference standard.
- No interim efficacy analysis. A blinded check of event yield after 25% of enrolment may adjust recruitment only.
- Report following STARD 2015 and, where relevant, TRIPOD+AI / CONSORT-AI items.

## 15. Open items before execution

Regulatory strategy and the intended-use statement; the final operational definitions; the acceptance criteria; ethics approval; data-protection assessment; usability study plan; site agreements; annotation tool selection; and SAP sign-off by a biostatistician.
