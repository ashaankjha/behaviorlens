# Real-video validation

The synthetic demo only exercises the rules. This report covers **real MediaPipe inference on real videos of moving people**, measured against independent ground truth. Run date: 2026-10-18. Model `pose_landmarker_lite`, sampling at 5 fps (see *Pose model comparison* for `full`, `heavy` and 10 fps).

> These numbers come from small public research samples (19 annotated stand-ups and 40 action clips). They are engineering evidence about the heuristics, **not clinical validation**.

## Data

| Dataset | What | Used for | Ground truth |
|---|---|---|---|
| **Charades** (Sigurdsson et al., 2016), public 126-clip sample on Hugging Face (`Aditya02/Charades-Action-Sequence-Sample`) | Home videos acted out from scripts by **consenting crowd workers**. The closest public match to the PRD's "staged footage of consenting adults". | `standing_transition` | Temporal labels: 154 *Someone is standing up from somewhere* (19 windows), 151 *going from standing to sitting* (20 windows, hard negatives). Per-frame posture labels come from activities: sitting in a chair, at a table or on a sofa versus opening doors, using light switches, walking through doorways. |
| **UCF101** (Soomro et al., 2012), 5 classes × 8 clips pulled by HTTP range requests from the full archive | Real action clips | `hand_raised` (per clip) | Class-level: JumpingJack and PullUps marked "arms overhead"; JumpRope, BodyWeightSquats and WalkingWithDog marked not. |

Clips are downloaded to the gitignored `data/validation/` and **never committed or redistributed**. Both datasets are for research use.

**Reproduce:**
```bash
python scripts/fetch_validation_clips.py          # ~220 MB, ~2 min
python scripts/validate_real_video.py --split holdout --json results.json   # ~3 min the first time, then cached
```

**Protocol.** Clips are split 50/50 by a hash of the clip ID into *tune* and *holdout* halves (`validate_real_video.split_of`). Every threshold change in rules-v0.2 was chosen using the **tuning half only**, and the held-out half was scored once with the chosen parameters. A detected stand-up counts as correct if it overlaps an annotated window widened by 1 s.

## Results

### `standing_transition` (Charades, event level)

| Rules | Split | Recall | Precision | Confused with sitting down | Recall when legs observable |
|---|---|---|---|---|---|
| v0.1 (synthetic-tuned) | holdout | 0.20 (2/10) | 0.50 (2/4) | 0 | 0.25 |
| **v0.2** | **holdout** | **0.40 (4/10)** | **0.40 (4/10)** | 0 | **0.50** |
| v0.1 | all | 0.11 (2/19) | 0.50 (2/4) | 0 | 0.17 |
| v0.2 | all | 0.26 (5/19) | 0.42 (5/12); **0.50 (6/12) after visual review** | 0 | 0.42 |

*How the v0.1 rows were produced:* v0.1 thresholds were run on v0.2 code through `--params`, which can't switch off v0.2's in-frame gating. The clean-room audit also re-ran the **actual v0.1 code** (commit `8886c05`) on the same landmarks. Recall and precision are identical; the only difference is v0.1 all-split "recall when legs observable", which is 0.15 with the real code against the 0.17 above ([AUDIT.md](AUDIT.md) claims 17–18).

**Per-frame posture**, computed with the production `posture_features` on observable frames labelled by Charades activities (tune: 515 seated and 167 standing frames; holdout: 363 and 351). Balanced accuracy, with the share of seated/standing frames given the *opposite* label in brackets:

| Rules | Tune | Holdout |
|---|---|---|
| v0.1: drop ≤ 0.35 seated, ≥ 0.6 standing | 0.53 (7.0% / 9.6%) | 0.67 (0.0% / 10.8%) |
| v0.2: hip/knee angles plus drop | 0.64 (8.5% / 11.4%) | **0.79** (0.8% / 12.8%) |

### `hand_raised` (UCF101, per clip)

| | Clips with ≥1 `hand_raised` |
|---|---|
| PullUps (arms overhead, sustained) | **7/8**. The miss: no frame where the hand rule could observe the arms (`hand_observable` = 0), so nothing was guessed. |
| JumpingJack (arms overhead, *brief*) | 3/8 |
| JumpRope | 0/8 ✅ |
| BodyWeightSquats | 2/8 |
| WalkingWithDog | 1/8 |

Measured against the raw class labels: precision 0.77, recall 0.63 (TP 10, FP 3, FN 6, TN 21). This is unchanged from v0.1.

## Pose model comparison

Run date: 2026-10-18. **Question:** do `pose_landmarker_full` or `pose_landmarker_heavy` improve real-video results over `pose_landmarker_lite`, and at what CPU cost? Secondary: does 10 fps sampling help brief `hand_raised` movements?

**Method.** rules-v0.2 defaults unchanged (no retuning). The same 166 clips and the same scorer as above. Each model is selected with `BEHAVIORLENS_MODEL_PATH`. The landmark cache is keyed by clip, model version (name + weight hash) and fps (`validate_real_video.cache_path`, LEARNINGS M21), so no run reuses another model's landmarks. Models: `float16/latest` from Google's bucket. `lite@59929e1d1ee9` 5.8 MB, `full@4eaa5eb7a983` 9.4 MB, `heavy@64437af838a6` 30.7 MB. CPU: 4-core cloud container.

```bash
BEHAVIORLENS_MODEL_PATH=assets/models/pose_landmarker_heavy.task \
  python scripts/validate_real_video.py --split holdout --json heavy.json   # also --split all, --fps 10
```

### `standing_transition` (Charades), 95% Wilson intervals in brackets

| Model, fps | Split | Recall | Precision | Confused with sitting down | Recall when legs observable | Median posture observability |
|---|---|---|---|---|---|---|
| **lite, 5** (default) | holdout | **0.40 (4/10)** [0.17–0.69] | 0.40 (4/10) [0.17–0.69] | 0 | 0.50 (4/8) | 0.41 |
| full, 5 | holdout | 0.30 (3/10) [0.11–0.60] | 0.38 (3/8) [0.14–0.69] | 0 | 0.50 (3/6) | 0.35 |
| heavy, 5 | holdout | 0.00 (0/10) [0.00–0.28] | 0.00 (0/1) | 0 | 0.00 (0/5) | 0.18 |
| lite, 10 | holdout | 0.60 (6/10) [0.31–0.83] | 0.33 (6/18) [0.16–0.56] | 0 | 0.71 (5/7) | 0.41 |
| **lite, 5** (default) | all | **0.26 (5/19)** [0.12–0.49] | 0.42 (5/12) [0.19–0.68] | 0 | 0.42 (5/12) | 0.34 |
| full, 5 | all | 0.16 (3/19) [0.06–0.38] | 0.33 (3/9) [0.12–0.65] | 0 | 0.30 (3/10) | 0.25 |
| heavy, 5 | all | 0.00 (0/19) [0.00–0.17] | 0.00 (0/1) | 0 | 0.00 (0/8) | 0.14 |
| lite, 10 | all | 0.42 (8/19) [0.23–0.64] | 0.32 (8/25) [0.17–0.52] | 0 | 0.64 (7/11) | 0.34 |

The lite 5 fps rows reproduce the rules-v0.2 results above exactly, a sanity check on the new cache key. "Recall when legs observable" uses each model's own observability, so its denominators differ.

### `hand_raised` (UCF101, clips with ≥1 event, all 40 clips)

| Model, fps | PullUps | JumpingJack | JumpRope | BodyWeightSquats | WalkingWithDog | TP / FP / FN / TN vs class labels |
|---|---|---|---|---|---|---|
| **lite, 5** | 7/8 | 3/8 | 0/8 | 2/8 | 1/8 | 10 / 3 / 6 / 21 |
| full, 5 | 7/8 | 3/8 | 0/8 | 3/8 | 0/8 | 10 / 3 / 6 / 21 |
| heavy, 5 | **8/8** | 3/8 | 0/8 | 2/8 | **0/8** | **11 / 2 / 5 / 22** |
| lite, 10 | 7/8 | 3/8 | 0/8 | 3/8 | 1/8 | 10 / 4 / 6 / 20 |

Heavy fixes both documented `hand_raised` errors: the PullUps clip where lite never observed the arms (hand observability 0 → 0.56), and the hallucinated-head WalkingWithDog false positive. Full also removes the WalkingWithDog false positive but adds a BodyWeightSquats clip. The extra squat clips at 10 fps and under full are a class-label "FP", like the two already adjudicated above; they were not reviewed visually here. Overall the differences are 1–2 clips out of 40 and within noise.

### CPU cost

| Model | Single-process inference, ms per frame (1,056 frames, decode excluded) | Full validation extraction, 166 clips / 20,057 frames, 4 workers (wall / CPU incl. decode) | 5-min video at 5 fps (1,500 frames), single process |
|---|---|---|---|
| lite | 24 | 137 s / 491 s | ≈ 36 s |
| full | 28 (1.15×) | 162 s / 579 s | ≈ 41 s |
| heavy | 60 (2.5×) | 364 s / 1,323 s | ≈ 90 s |
| lite at 10 fps | 24 | 253 s / 910 s (2× frames) | ≈ 72 s |

### Why the larger models score lower on stand-ups

Not because they see legs worse. They **report lower visibility for legs**, so the fixed `visibility_threshold` (0.5, chosen with lite) gates more frames as unobservable. In Charades frames with a detected pose, knees have visibility ≥ 0.5 in 58% (lite), 55% (full) and 37% (heavy) of frames, with median knee visibility 0.61, 0.56 and 0.27. Hips and shoulders hardly change (hips 93–94% for all three). Heavy also detects a pose in slightly *more* frames (90.2% vs 88.5%). For SFPJ7, a stand-up lite finds and heavy misses, the frames show the person under a blanket on a sofa, then standing behind a coffee table. The legs really are mostly hidden, so heavy's low visibility looks more honest than wrong. The PRD's design choice (missing evidence → unobservable) then yields no event.

*Diagnostic only, tune split, not a selection* (the default stays 0.5): with a lower visibility threshold the larger models do not overtake lite at matched observability. Median observability, then tune-split recall:

| Visibility threshold | lite | full | heavy |
|---|---|---|---|
| 0.5 (default) | 0.22 → 1/9 | 0.22 → 0/9 | 0.08 → 0/9 |
| 0.3 | 0.42 → 2/9 | 0.32 → 1/9 | 0.18 → 0/9 |
| 0.2 | 0.57 → 3/9 | 0.43 → 1/9 | 0.22 → 1/9 |

### 10 fps

- **`hand_raised`: no change where it was expected.** JumpingJack stays at 3/8. The misses are arm raises shorter than the PRD's 0.6 s minimum (see above), so a higher sampling rate cannot recover them. Only the event definition could, and that is not a model or fps question.
- **`standing_transition` changes, and not in a validated direction.** Recall rises (holdout 6/10 vs 4/10) but precision falls (6/18 vs 4/10), with twice the detections. The v0.2 temporal parameters (dwell 0.8 s, gap bridging 0.4 s) were selected at 5 fps. A plausible (unverified) mechanism: at 10 fps the same durations are reached from more, noisier frames, and short gaps fall inside the 0.4 s bridge more often. Treat this as an fps–parameter interaction to tune, on the tune split, before using it, not as a free gain.

### Recommendation

**Keep `pose_landmarker_lite` at 5 fps as the default.**

- Neither larger model improves `standing_transition` with the current thresholds. Full is slightly worse (3/10 vs 4/10 holdout) and heavy finds nothing (0/10). Paired on the 19 annotated windows, lite finds 5 that heavy misses and heavy finds none that lite misses (exact McNemar p = 0.06 all, p = 0.13 holdout). That is suggestive at this sample size, not conclusive.
- Heavy's small `hand_raised` gain (1 TP and 1 FP fixed out of 40 clips) does not justify 2.5× CPU together with the stand-up loss.
- Full is nearly free (+15% CPU) but shows no benefit.
- Thresholds are coupled to the model's visibility calibration. Switching models would require re-selecting at least `visibility_threshold` (and probably the posture thresholds) on the tune split, then scoring the holdout once. Even the post-hoc diagnostic above does not suggest that would beat lite.

Caveats as above: 19 stand-up windows and 40 action clips, home and sports footage rather than clinic recordings, and incomplete Charades annotations. All intervals are wide.

## Camera motion

Run date: 2026-10-20. **Question:** do handheld camera pans, tilts and zooms cause false `standing_transition` suggestions, and does treating strongly moving-camera frames as unobservable for posture remove them without costing real stand-ups?

**Why it matters.** Landmarks are image coordinates. When the camera moves, the apparent body geometry changes even if the person does not, and a pan can swing the view away from the person entirely. In the v0.2 review (*What we learned*, item 6), two false positives came from camera pans in clip HIOY0 (≈ 20.4 s and ≈ 33.4 s).

### Method

`src/behaviorlens/camera_motion.py`, computed inside the pose frame loop (`pipeline.extract_pose`) and stored with the landmarks (`PoseSequence.camera_motion`; older `.npz` files without it still load and mask nothing).

1. Each sampled frame is downscaled to 320 px wide grayscale.
2. Corners (`goodFeaturesToTrack`) are picked in the previous frame **outside the detected person's padded landmark box**, tracked with pyramidal Lucas–Kanade, checked forward–backward, and matches that land in the current person box are dropped.
3. A RANSAC similarity fit (`estimateAffinePartial2D`) gives image-centre translation (fraction of frame width/height), rotation, log scale change and the inlier ratio.
4. With too few features or inliers, phase correlation on the frames (person box blanked) gives a translation-only estimate if its peak response is ≥ 0.15. Otherwise the frame is *unknown*.
5. Rules (`camera_motion_state`): a reliable estimate (phase correlation, or features with inlier ratio ≥ 0.5) is **moving** if translation > 0.2 frame/s, rotation > 10°/s or |log scale| > 0.1/s, otherwise **static**; anything else is **unknown**. Unknown never masks: missing evidence about the camera is not evidence that it moved. Moving frames become `UNKNOWN` for posture (`features.posture_features`). Raw estimates are stored, so thresholds can change without re-extraction.

On the synthetic tests (`tests/test_camera_motion.py`: a textured world warped by known pans, tilts, rolls and zooms, with and without an independently moving textured "person"), translation is recovered within 0.004 frame widths, rotation within 0.2°, log scale within 0.006, and a static camera with a moving person stays static.

**Speed.** 3.7 ms per sampled frame on average (median 3.3, p95 6.0; 2,415 frames from 20 Charades/UCF101 clips at 5 fps, single CPU process, decode excluded). That is about 15 % on top of `pose_landmarker_lite` inference (24 ms/frame). 98 % of frame pairs used the feature path and 0.7 % phase correlation; 1.3 % were unknown (mostly first frames). The full validation extraction (166 clips, pose and motion, 4 workers) took 205 s wall.

**Data.** Camera motion is common but mostly brief: the median Charades clip has the camera moving in 1–2 % of sampled frames, and 35 of 126 clips exceed 10 %.

### Threshold selection (tuning half only)

A grid over translation {0.15, 0.2, 0.3, 0.5, 0.75, off} frame/s × rotation {5, 10, 20, off} °/s × scale {0.1, 0.2, 0.3, off}/s × padding {0, 1} frames was scored on the **tune** split. The tune half has 9 stand-up windows, of which v0.2 finds 1 (V1PRP), and 2 detections (1 false, JLQLC, unrelated to the camera). It therefore shows only the *cost* of masking. The pre-declared rule was: **the most sensitive setting that loses no tune stand-up and adds no tune false positive**. Translation 0.15 /s, rotation 5 °/s, or a one-frame pad at 0.2 /s each lose V1PRP (its person stands up just as the camera starts to follow), so the selection is translation 0.2 /s, rotation 10 °/s, scale 0.1 /s, no padding. It masks 6.8 % of tune frames.

The tune split's per-frame data agrees on the direction: activity-labelled posture frames with camera translation above 0.2 /s were misclassified 25 % of the time against 12.5 % otherwise (16 frames, so weak evidence on its own).

**Hand rules are not masked.** On the tune UCF101 clips, masking moving-camera frames for `hand_raised` changed no clip's outcome and only split one PullUps event in two. The hand cue (wrist above shoulder in the same frame) does not depend on where the camera points. Camera motion is still reported in hand-event evidence (`camera_moving_frames`).

### Results (held-out half, scored once)

`standing_transition`, 95 % Wilson intervals in brackets:

| Rules | Split | Recall | Precision | False positives | Confused with sitting down | Recall when legs observable |
|---|---|---|---|---|---|---|
| v0.2 (`mask_camera_motion: false`) | holdout | 0.40 (4/10) [0.17–0.69] | 0.40 (4/10) [0.17–0.69] | 6 | 0 | 0.50 (4/8) |
| **v0.2 + camera motion** | **holdout** | **0.30 (3/10)** [0.11–0.60] | **0.50 (3/6)** [0.19–0.81] | **3** | 0 | 0.43 (3/7) |
| v0.2 | all | 0.26 (5/19) [0.12–0.49] | 0.42 (5/12) [0.19–0.68] | 7 | 0 | 0.42 (5/12) |
| v0.2 + camera motion | all | 0.21 (4/19) [0.09–0.43] | 0.50 (4/8) [0.22–0.78] | 4 | 0 | 0.36 (4/11) |

The v0.2 rows reproduce the earlier tables exactly from the new caches (pose landmarks are unchanged; only motion data was added).

`hand_raised` (UCF101) is unchanged by construction: holdout TP/FP/FN/TN 5/3/1/9, all 10/3/6/21.

**Per-frame posture** (production `posture_features`, Charades activity labels from `eval_charades_frames`: seated = sitting at a table, in a chair or on a sofa; standing = doors, doorways, light switches; frames inside stand-up/sit-down windows excluded). Balanced accuracy, with the share of observed seated/standing frames given the *opposite* label in brackets:

| Rules | Tune | Holdout |
|---|---|---|
| v0.2 | 0.664 (8.5 % / 12.3 %), 515 + 122 frames | 0.783 (0.8 % / 13.3 %), 363 + 278 frames |
| v0.2 + camera motion | 0.680 (8.8 % / 10.7 %), 502 + 103 frames | **0.801** (0.8 % / 9.8 %), 356 + 234 frames |

These standing labels use a slightly different activity set from the original v0.2 table above, so the v0.2 row here differs from 0.64 / 0.79; the comparison within this table is paired. On the holdout, frames with camera translation above 0.2 /s are misclassified **35.5 %** of the time against 5.6 % otherwise (31 of 552 observed frames), which is the mechanism the masking targets.

### What changed, clip by clip (holdout)

| Clip | v0.2 | With camera motion | What the frames show |
|---|---|---|---|
| HIOY0 20.4 s | false positive | **suppressed** | The camera points at the floor and a cushion; MediaPipe reports a "seated" pose with no person in view. The camera then pans back to the standing person. |
| HIOY0 33.4 s | false positive | **suppressed** | Same pattern with a TV cabinet, then a pan back to the person. |
| 9HGNV 9.2 s | false positive | **suppressed** | A person walks while carrying bedding; the camera follows and zooms. Legs are hidden under the bedding and read as "seated" while the camera moves. |
| HIOY0 14.2 s | true positive | **lost** | A real stand-up from a chair, filmed by a handheld camera that follows the person (59 % of the clip's frames have a moving camera). |

**HIOY0 outcome:** both pan-induced false positives are gone, and so is the clip's one correct detection. HIOY0 now yields no suggestion at all. Camera motion explains *when* posture cannot be trusted; it cannot tell a pan from a camera following a real stand-up. With the camera moving, posture there is reported as unobservable, which is the PRD's rule for missing evidence.

Net on the held-out half: false positives halve (6 → 3) for one lost stand-up (4 → 3). Precision rises from 0.40 to 0.50 and recall falls from 0.40 to 0.30. Both changes sit well inside the confidence intervals; this is directional evidence from 10 held-out stand-ups, not a conclusive improvement.

### Post-hoc sensitivity (holdout, *not* a selection)

| Translation /s | Rotation °/s | Scale /s | Pad | Holdout hits / detections | Frames masked |
|---|---|---|---|---|---|
| off | off | off | – | 4 / 10 | 0 % |
| **0.2** | **10** | **0.1** | **0** | **3 / 6** | **9.3 %** |
| 0.2 | 10 | 0.1 | 1 | 4 / 6 | 15.9 % |
| 0.2 | off | off | 0 | 4 / 7 | 6.4 % |
| 0.5 | off | off | 0 | 4 / 8 | 1.0 % |
| 0.3 | 10 | 0.1 | 0 | 3 / 6 | 7.3 % |

A one-frame pad would have kept the HIOY0 stand-up on the holdout, but it lost the tune split's only stand-up, so it was not selected. Choosing it now would be tuning on the held-out half (LEARNINGS M19). More annotated footage with moving cameras is needed to settle the pad.

### Limitations

- **A camera that follows a real stand-up hides it.** Masking cannot separate "camera moved, scene changed" from "camera tracked the person". Posture angles are fairly robust to small pans, so a gentler rule (for example, only masking when the cumulative shift between the seated and standing evidence is large) might keep such stand-ups, but HIOY0's true stand-up had a cumulative camera shift between the last seated and first standing frame (0.43 frame widths, summed over reliable estimates) as large as its false ones (0.33 and 0.64), and the tune split has no camera-induced false positive to select such a rule on.
- **Masking removes evidence only while the camera moves.** A still seated view, a short pan, then a still standing view of something else can still form a transition (`test_stand_up_seen_after_a_pan_still_counts`). Hallucinated poses on empty scenery (HIOY0) are a pose-quality problem that camera motion only catches when the camera is moving.
- **Large foreground objects or people** that are not the detected person can dominate the background fit; RANSAC and the inlier-ratio gate reduce, but do not remove, this. A second person is handled by the separate multi-person work.
- **Slow drift** below 0.2 frame/s is treated as static. Rotation and scale estimates from a few features are noisy (the 10 °/s rotation threshold sits well above the 95th percentile of 3–4 °/s on observed Charades posture frames).
- Samples remain small: 10 held-out stand-ups, 3 camera-related false positives.

## What we learned (and what changed)

1. **Thresholds tuned on synthetic data don't transfer.** Real seated people seen from elevated home cameras have hip-to-knee drops of 0.3–0.55 torso lengths (median 0.39). v0.1's "≤ 0.35" caught fewer than half of real seated frames. → v0.2 adds **hip angle** (≤ 140° seated, ≥ 150° standing) and **knee angle** (≤ 130° seated, when ankles are visible). The drop cue is kept for front-on camera setups.
2. **People leave the frame right after standing up.** → v0.2 uses dwell 0.8 s (was 1.0) and allows up to 5 s between the last seated frame and the first standing frame (was 3). Both were chosen on the tuning half.
3. **Most misses are unobservable, not misclassified.** Legs are hidden behind furniture or out of frame. In 7 of the 19 stand-up clips, posture is observable in fewer than 30% of the clip's frames; by design these produce no event where evidence is missing.
4. **Off-screen landmarks.** 2.45% of "visible" landmarks (visibility ≥ 0.5) lie outside the image. → v0.2 requires landmarks to be in frame. Separately, MediaPipe can **hallucinate a head and shoulders inside the frame** when only a torso is visible (the WalkingWithDog false positive). This remains a known failure mode.
5. **Class labels aren't operational definitions.** On visual review:
   - Both BodyWeightSquats "false positives" are correct under the PRD definition (hands clasped behind the head, or arms raised above shoulder height).
   - The JumpingJack misses are arm raises lasting about 0.4 s (2 frames at 5 fps), shorter than the PRD's 0.6 s minimum.
   - Under the operational definition, the only real `hand_raised` error is the hallucinated-head walking clip. This adjudication was done by the same agent that built the rules, so treat it as indicative.
6. **Annotations are incomplete.** One stand-up "false positive" (PKEZI, 24.6–27.0 s) is a real stand-up from a bed that the annotation places later (from 28.5 s). The other false positives come from handheld camera pans (2), partial-body close-ups or jitter on standing people (3), and **a second person** (1). Multi-person attribution is a documented MVP limitation.

## Limits of this validation

- **Small samples**: 19 stand-up windows (10 held out) and 40 clips. Confidence intervals are wide; 95% Wilson intervals: v0.2 held-out recall 4/10 → 0.17–0.69; overall 5/19 → 0.12–0.49; v0.1 held-out 2/10 → 0.06–0.51. The improvement is consistent across splits and per-frame metrics but **not statistically conclusive** at this sample size.
- Charades and UCF101 show adults in everyday or sports settings, not ABA sessions. Camera placement, furniture and people differ from a clinic room.
- Per-clip `hand_raised` metrics come from class labels; only disagreements were reviewed visually.
- Thresholds remain heuristics. **Not for clinical decision-making.**

## How to improve these numbers

- Record staged clips in the intended setting (front-on camera, full body visible) using the README procedure, and add them to `data/validation/` with a ground-truth CSV.
- ~~Try `pose_landmarker_full`/`heavy` and 10 fps.~~ Done (see *Pose model comparison*): neither larger model helps with the current thresholds, and 10 fps doesn't help `hand_raised`. Re-tuning the temporal parameters for 10 fps on the tune split is the remaining lead.
- ~~Add camera-motion detection.~~ Done (see *Camera motion*): it removes the pan false positives, at the cost of stand-ups filmed by a following camera. More annotated moving-camera footage is needed to tune the padding.
- Add person tracking (multi-person) to suppress the remaining second-person false positive.
