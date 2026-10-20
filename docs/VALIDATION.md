# Real-video validation

The synthetic demo only exercises the rules. This report covers **real MediaPipe inference on real videos of moving people**, measured against independent ground truth. Run date: 2026-10-18. Model `pose_landmarker_lite`, sampling at 5 fps (see *Pose model comparison* for `full`, `heavy` and 10 fps).

> These numbers come from small public research samples (19 annotated stand-ups and 40 action clips). They are engineering evidence about the heuristics, **not clinical validation**.

> **Current shipped version: rules-v0.3** (person tracking on; camera motion estimated and shown, masking off by default). Held-out stand-up recall 0.20 (2/10), precision 0.40 (2/5), 3 false positives, against rules-v0.2's 0.40, 0.40 and 6. See *rules-v0.3: tracking + camera motion combined*. The sections below are in chronological order.
>
> **Fixed-camera evidence (2026-10-20):** on 506 fixed-camera videos of adults (UTKinect-Action3D, PKU-MMD; 195 stand-ups), rules-v0.3 stand-up recall is 0.71 (138/195), precision 0.56 (138/248); held-out half 0.65 (72/110), 0.55 (72/130). `hand_raised` precision 0.88 against an independent Kinect-skeleton reference. Camera-motion masking is inert on fixed cameras. See *Fixed-camera datasets*.

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

## Person tracking

Run date: 2026-10-20. **Question:** does tracking people across frames, and computing events on one subject track, remove the identity-switch false positives of rules-v0.2 without losing real detections? rules-v0.2 thresholds are unchanged. `ALGORITHM_VERSION` is unchanged on this branch.

**Method.** MediaPipe returns up to 4 poses per frame (`num_poses = AnalysisSettings.max_people`). `tracking.track_people` links them across sampled frames using geometry only: the IoU and centre distance between the all-landmark bounding box and a constant-velocity prediction, a soft size term, and a staleness term. Assignment per frame is exact (most matches, then least cost). A track missing for more than 2 s gets a new ID. There is no appearance, face or identity model. The rules then run on one **subject** track, by default the one with the greatest coverage × mean box size. Frames without the subject are unobservable. The all-people landmarks have their own cache key (`cache_path(..., max_people=4)` → `*_multi4.npz`). Three attributions are scored on identical clips:

- **legacy**: rules-v0.2 as shipped (`num_poses = 2`, the largest person per frame). This reproduces the published v0.2 numbers exactly.
- **largest**: 4-person landmarks with v0.2's per-frame largest person. This isolates the effect of `num_poses`.
- **tracked**: 4-person landmarks with tracking and the subject track (the new default).

```bash
python scripts/validate_real_video.py --split holdout --attribution tracked   # also: largest, legacy
```

**Parameter selection (tune split only, M19).** The first run used a-priori defaults and fragmented single people badly (tune-split median subject pose coverage 0.64 against 0.94 detected). Diagnosis on tune clips found two causes. First, the box built from *visible* landmarks jumps whenever the legs flicker in and out of visibility (consecutive-frame size ratio p99 4.3×). The fix is to track on the box of all landmark positions (p99 2.9×). Second, a hard 3× size gate split people at those jumps. A grid over `max_size_ratio` ∈ {3, 6, 100}, `max_gap_s` ∈ {1, 2, 4} and `duplicate_iou` ∈ {0.5, 0.7}, ranked by tune stand-up recall, then precision, then `hand_raised`, then posture observability, selected `max_size_ratio = 100` (size is only a soft cost) and `duplicate_iou = 0.7`. The three `max_gap_s` values tied, so the a-priori 2 s was kept. The held-out half was then scored **once**.

### `standing_transition` (Charades), 95% Wilson intervals in brackets

| Attribution | Split | Recall | Precision | Confused with sitting down | Recall when legs observable | Median posture observability |
|---|---|---|---|---|---|---|
| legacy (v0.2) | tune | 0.11 (1/9) | 0.50 (1/2) | 0 | 0.25 (1/4) | 0.22 |
| largest, 4 poses | tune | 0.11 (1/9) | 0.25 (1/4) | 0 | 0.25 (1/4) | 0.21 |
| tracked | tune | 0.11 (1/9) | 1.00 (1/1) | 0 | 0.33 (1/3) | 0.13 |
| **legacy (v0.2)** | **holdout** | **0.40 (4/10)** [0.17–0.69] | **0.40 (4/10)** [0.17–0.69] | 0 | 0.50 (4/8) | 0.41 |
| largest, 4 poses | holdout | 0.30 (3/10) [0.11–0.60] | 0.27 (3/11) [0.10–0.57] | 0 | 0.38 (3/8) | 0.41 |
| **tracked** | **holdout** | **0.20 (2/10)** [0.06–0.51] | **0.40 (2/5)** [0.12–0.77] | 0 | 0.33 (2/6) | 0.29 |
| legacy (v0.2) | all | 0.26 (5/19) | 0.42 (5/12) | 0 | 0.42 (5/12) | 0.34 |
| largest, 4 poses | all | 0.21 (4/19) | 0.27 (4/15) | 0 | 0.33 (4/12) | 0.36 |
| tracked | all | 0.16 (3/19) | 0.50 (3/6) | 0 | 0.33 (3/9) | 0.20 |

`hand_raised` (UCF101) is **identical** under all three attributions on every split (all 40 clips: TP 10, FP 3, FN 6, TN 21). Only 3 of the 40 UCF clips have more than one substantial track. 4-person extraction cost about the same as 2-person extraction (166 clips: 177 s wall, 614 s CPU, against 169 s and 603 s).

### Held-out disagreements, reviewed visually (M18)

Frames were rendered with track boxes, viewed in scratch and not kept.

| Clip | v0.2 | Tracked | What the frames show |
|---|---|---|---|
| **K5CRM** | 1 false stand-up (12.6–13.8 s) | **none** | Two people. v0.2's per-frame largest person switched 13 times; its "seated" frames and "standing" frames came from different tracks (seated from track 1, standing from track 3 at 13.6 s). Tracking keeps subject #1 (99% coverage) and proposes nothing. The false positive is removed as intended. |
| NY5R2 | hit (3.8–4.2 s) | miss | The v0.2 "hit" was **a lucky identity switch**. Its seated evidence (2.0–3.8 s) came from a phantom pose that MediaPipe hallucinated on a backpack (track 1); its standing evidence came from the real person (track 2). Tracking follows the real person, whose seated phase is out of view, so no event is proposed. That is correct under the "never guess" rule. |
| HIOY0 | hit (14.2–16.4 s) plus 2 false positives | miss, no false positives | One person, but the handheld camera pans sharply at about 15 s. The box jumps and the tracker splits the person into tracks 1 and 2. This is a **real tracking loss caused by camera motion**. v0.2's two false positives were also built across track switches (tracks 3→4 and 7→9). |
| O1LOW | none | none | Under "largest" (4 poses) a false positive appears; tracking removes it. |
| 9HGNV, E64R4, PKEZI | 1 false positive each | the same 1 each | Not caused by attribution. PKEZI is the real but unannotated stand-up adjudicated earlier, so the tracked holdout precision after visual review is 3/5. |

**Attribution-correct comparison** (post-hoc adjudication by the authoring agent, so indicative only): v0.2 holdout recall is 3/10 once the NY5R2 phantom-switch hit is discounted. Tracked recall is 2/10. The remaining gap is HIOY0, caused by the camera pan.

### Verdict

- Tracking **removes identity-switch false positives**. Holdout false positives fell from 6 to 3, and every removed one (K5CRM and HIOY0 ×2) was a seated→standing pair stitched across two tracks. Precision is the same at 0.40 raw, and higher after visual review (0.60 against 0.50).
- It **loses recall** (holdout 2/10 against 4/10). One of the two lost hits was itself an artifact (NY5R2). The other is a camera pan breaking geometric continuity (HIOY0). `num_poses = 4` alone also costs one hit on holdout (largest: 3/10).
- Multiple substantial tracks are common in Charades (82 of 126 clips): real second people, phantom poses on furniture and clothes, and pan-induced splits. So the multi-person warning shows often on handheld home video.
- None of this is statistically conclusive at 10 held-out windows. The intervals overlap almost completely. It is engineering evidence about *why* detections happen, not a validated accuracy gain.

**Rejected alternative: stitching tracks that never co-occur.** This would recover HIOY0-type splits, but on the tune split it would also merge different people who appear one after another. 03OQS has a person at a table, then a different person behind them who is detected only once the first leaves. That is exactly the attribution error tracking exists to prevent. Combining tracking with the camera-motion detector from the parallel session is the better route to the pan-split case.


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

## rules-v0.3: tracking + camera motion combined

Run date: 2026-10-20. Person tracking and camera-motion detection were developed on separate branches, each measured against rules-v0.2 alone. This section measures them **together**, as they ship.

**Integration fix found while merging.** Real videos take the multi-person path (`extract_multi_pose` → tracking → subject track). Camera motion had only been wired into the single-person `extract_pose`, so after a plain merge **motion was silently never estimated in production**, even though every test passed. It is now estimated in the multi-person frame loop, with **every** detected person excluded from the background fit, and carried onto each track (`MultiPoseSequence.camera_motion`). Regression tests: `test_multi_person_extraction_records_camera_motion_and_tracks_carry_it`, `test_run_analysis_reports_camera_motion_on_the_tracking_path`, `test_every_detected_person_is_excluded_from_the_background_fit` (LEARNINGS M38).

### Pre-registered selection (written before any combined result was viewed)

> Candidates: **A** v0.2 attribution, masking off (the v0.2 baseline) · **B** v0.2 attribution + camera-motion masking · **C** tracking, masking off · **D** tracking + masking. Select on the **tune** split by the highest `standing_transition` F1; ties go to fewer tune false positives, then to the simpler configuration (A < B < C < D). Score the held-out half once for the selected candidate and for A.

All runs use `pose_landmarker_lite`, 5 fps, rules-v0.2 thresholds and `--tol 1.0`. B, C and D use the motion thresholds and tracking parameters their branches selected on the tune split.

| Candidate | Tune recall | Tune precision | Tune F1 | Tune FP |
|---|---|---|---|---|
| A v0.2 | 1/9 | 1/2 | 0.182 | 1 |
| B v0.2 + masking | 1/9 | 1/2 | 0.182 | 1 |
| **C tracking** | 1/9 | 1/1 | **0.200** | 0 |
| D tracking + masking | 1/9 | 1/1 | 0.200 | 0 |

C and D tie exactly, because masking changes nothing on the tune half once tracking is on. The tie-break selects **C**. **rules-v0.3 = tracking on, camera motion estimated and shown but `mask_camera_motion = False` by default.** Reviewers can switch masking on per analysis.

### Held-out half (scored once for C and A; B and D post-hoc)

`standing_transition` (Charades), 95% Wilson intervals in brackets:

| Configuration | Recall | Precision | False positives | Confused with sitting down | Per-frame posture (balanced acc.) |
|---|---|---|---|---|---|
| A rules-v0.2 | 0.40 (4/10) [0.17–0.69] | 0.40 (4/10) [0.17–0.69] | 6 | 0 | 0.783 |
| **C rules-v0.3 (selected)** | **0.20 (2/10)** [0.06–0.51] | **0.40 (2/5)** [0.12–0.77] | **3** | 0 | 0.778 |
| *B post-hoc* | 0.30 (3/10) | 0.50 (3/6) | 3 | 0 | 0.801 |
| *D post-hoc* | 0.20 (2/10) | 0.50 (2/4) | 2 | 0 | 0.802 |

All clips: A 5/19 recall, 5/12 precision · **C 3/19, 3/6** · B 4/19, 4/8 · D 3/19, 3/5.

`hand_raised` (UCF101) is **identical in all four configurations**: holdout TP/FP/FN/TN 5/3/1/9, all clips 10/3/6/21.

The shipped defaults reproduce C exactly with no `--params` overrides (`algorithm_version: rules-v0.3`, `params_override: null`).

### Verdict (honest)

- **rules-v0.3 halves held-out false positives (6 → 3) and halves raw held-out recall (4/10 → 2/10).** Raw precision is unchanged (0.40). None of these differences is statistically significant at 10 held-out stand-ups; the intervals overlap almost completely.
- **Why ship it anyway.** Frame review on the person-tracking branch showed that one of v0.2's four held-out "hits" (NY5R2) came from a phantom pose on a backpack, and that a v0.2 false positive (K5CRM) stitched seated and standing evidence from **two different people**. Attributing one person's movement to another is a worse failure in a review tool than a missed suggestion (draft risk file, H-03). Tracking is the control for that hazard, and it costs recall that partly came from artifacts.
- **Camera motion** is estimated, shown on the timeline and reported in warnings. Masking is off by default because the pre-registered rule could not distinguish it on the tune half. The post-hoc held-out rows suggest masking cuts false positives further (D: 2). That evidence is held-out and must not drive the default; it is a candidate for the next tune/holdout cycle, ideally on more data.
- **This is not clinical evidence.** Small public samples, non-ABA footage, and adjudication by the authoring agent. See `docs/regulatory/clinical_validation_plan.md` for what clinical evidence would require.

Reproduce:

```bash
python scripts/validate_real_video.py --split holdout                           # rules-v0.3 defaults (C)
python scripts/validate_real_video.py --split holdout --attribution legacy \
  --params '{"mask_camera_motion": false}'                                       # A: rules-v0.2
python scripts/validate_real_video.py --split holdout --params '{"mask_camera_motion": true}'   # D
```

## What we learned (and what changed)

1. **Thresholds tuned on synthetic data don't transfer.** Real seated people seen from elevated home cameras have hip-to-knee drops of 0.3–0.55 torso lengths (median 0.39). v0.1's "≤ 0.35" caught fewer than half of real seated frames. → v0.2 adds **hip angle** (≤ 140° seated, ≥ 150° standing) and **knee angle** (≤ 130° seated, when ankles are visible). The drop cue is kept for front-on camera setups.
2. **People leave the frame right after standing up.** → v0.2 uses dwell 0.8 s (was 1.0) and allows up to 5 s between the last seated frame and the first standing frame (was 3). Both were chosen on the tuning half.
3. **Most misses are unobservable, not misclassified.** Legs are hidden behind furniture or out of frame. In 7 of the 19 stand-up clips, posture is observable in fewer than 30% of the clip's frames; by design these produce no event where evidence is missing.
4. **Off-screen landmarks.** 2.45% of "visible" landmarks (visibility ≥ 0.5) lie outside the image. → v0.2 requires landmarks to be in frame. Separately, MediaPipe can **hallucinate a head and shoulders inside the frame** when only a torso is visible (the WalkingWithDog false positive). This remains a known failure mode.
5. **Class labels aren't operational definitions.** On visual review:
   - Both BodyWeightSquats "false positives" are correct under the PRD definition (hands clasped behind the head, or arms raised above shoulder height).
   - The JumpingJack misses are arm raises lasting about 0.4 s (2 frames at 5 fps), shorter than the PRD's 0.6 s minimum.
   - Under the operational definition, the only real `hand_raised` error is the hallucinated-head walking clip. This adjudication was done by the same agent that built the rules, so treat it as indicative.
6. **Annotations are incomplete.** One stand-up "false positive" (PKEZI, 24.6–27.0 s) is a real stand-up from a bed that the annotation places later (from 28.5 s). The other false positives come from handheld camera pans (2), partial-body close-ups or jitter on standing people (3), and **a second person** (1, K5CRM). That one is removed by person tracking (see *Person tracking*).

## Limits of this validation

- **Small samples**: 19 stand-up windows (10 held out) and 40 clips. Confidence intervals are wide; 95% Wilson intervals: v0.2 held-out recall 4/10 → 0.17–0.69; overall 5/19 → 0.12–0.49; v0.1 held-out 2/10 → 0.06–0.51. The improvement is consistent across splits and per-frame metrics but **not statistically conclusive** at this sample size.
- Charades and UCF101 show adults in everyday or sports settings, not ABA sessions. Camera placement, furniture and people differ from a clinic room.
- Per-clip `hand_raised` metrics come from class labels; only disagreements were reviewed visually.
- Thresholds remain heuristics. **Not for clinical decision-making.**

## How to improve these numbers

- ~~Find fixed-camera, full-body footage.~~ Done with UTKinect-Action3D and PKU-MMD (see *Fixed-camera datasets*). Record staged clips in the intended clinic-like setting with the README procedure and add them in the shared format (`scripts/datasets.py`).
- Next pre-registered cycle: seated thresholds for frontal views (`seated_max_drop` 0.45 looked strong on tune only) and track fragmentation for single-person videos (UTKinect s03_e01).
- ~~Try `pose_landmarker_full`/`heavy` and 10 fps.~~ Done (see *Pose model comparison*): neither larger model helps with the current thresholds, and 10 fps doesn't help `hand_raised`. Re-tuning the temporal parameters for 10 fps on the tune split is the remaining lead.
- ~~Add person tracking.~~ Done (see *Person tracking*).
- ~~Add camera-motion detection.~~ Done (see *Camera motion*).
- Integrated in rules-v0.3: see *rules-v0.3: tracking + camera motion combined*.

## Fixed-camera datasets (UTKinect-Action3D, PKU-MMD)

Run date: 2026-10-20. **Question:** the PRD's intended setting is a **fixed camera with the full body visible**, but all earlier evidence came from handheld home video (19 Charades stand-ups). How do rules-v0.3 perform on fixed-camera, temporally labelled footage of consenting adults, and does camera-motion masking matter there?

**Answer in one paragraph.** On 506 fixed-camera videos (195 stand-ups), shipped rules-v0.3 find **0.71 of stand-ups (138/195) [0.64–0.77] with precision 0.56 (138/248) [0.49–0.62]**; on the held-out half 0.65 (72/110) and 0.55 (72/130). That is far above Charades (0.16 recall), because legs are observable (median posture observability 0.91–0.94 against 0.20). Camera-motion masking is **exactly inert** on these cameras (no frame moving; B ≡ A and D ≡ C to the count). `hand_raised`, scored against an independent Kinect-skeleton reference, reaches **precision 0.88 and recall ≥ 0.79** (held-out pooled; recall is a lower bound, see below). The pre-registered Phase 2 grid found **no setting that improves the held-out half** (F1 0.580 → 0.585); the largest remaining stand-up lever is the **seated thresholds for frontal views**, which the grid deliberately did not include.

### Phase 2 pre-registration (written before any tune-split result on the new data was viewed)

> Disclosure: before writing this, one diagnostic run of the shipped rules (C) on the **UTKinect holdout** half was viewed (13 videos), and two held-out misses were inspected. No tune-split result on UTKinect or PKU-MMD had been computed.
>
> **Data.** Pooled tune split = tune halves of Charades (legacy split key), UTKinect, PKU-MMD Phase 1 and Phase 2 (front view), split by `datasets.split_of(f"{dataset}:{video_id}")`. Proceed only if it holds ≥ 25 `standing_transition` windows.
>
> **Fixed.** `pose_landmarker_lite`, 5 fps, tracked attribution (rules-v0.3, the H-03 control), tracking parameters, all posture/hand angle thresholds, `hand_raise_min_s` (the PRD's 0.6 s definition), `--tol 1.0`.
>
> **Grid (96 settings).** `dwell_s` ∈ {0.4, 0.6, 0.8, 1.0} × `max_transition_s` ∈ {3, 5, 7} × `visibility_threshold` ∈ {0.3, 0.4, 0.5, 0.6} × `mask_camera_motion` ∈ {false, true}. rules-v0.3 is (0.8, 5, 0.5, false).
>
> **Objective.** Highest pooled-tune `standing_transition` F1 (micro, from pooled counts). **Guard:** pooled-tune `hand_raised` F1 against labels must not fall more than 0.02 below rules-v0.3's. **Ties** (F1 within 0.005): fewer pooled-tune false positives, then fewest parameters changed from rules-v0.3, then the value closest to rules-v0.3.
>
> **Holdout.** Score the selected setting and rules-v0.3 on the pooled holdout **once**, per dataset and pooled, with Wilson intervals. The selected setting is reported as a *rules-v0.4 candidate*; `config.py` defaults are not changed here.

(This rule was committed in `1972765`, before any tune-split result on the new data was computed.)

### Data

| Dataset | Camera | Videos used | `standing_transition` | `sitting_down` | `hand_raised` | Source labels and mapping |
|---|---|---|---|---|---|---|
| **UTKinect-Action3D** (Xia, Chen & Aggarwal, CVPRW 2012) | fixed Kinect v1, 640×480 | 20 (10 adults × 2) | 20 | 20 | 20 | standUp, sitDown, waveHands → `hand_raised`; walk, pickUp, carry, throw, push, pull, clapHands → `other` |
| **PKU-MMD Phase 2** (Liu et al., 2017), middle (front) view | fixed Kinect v2, 1080p → 960×540 | 333 of 333 | 107 | 90 | 26 | 34 standing up, 33 sitting down, 13 hand waving → `hand_raised`; the other 48 actions → `other` (incl. 5 *cheer up*, see below) |
| **PKU-MMD Phase 1**, middle view, **subset** | fixed Kinect v2 | 153 of 359 | 68 | 69 | 63 | as Phase 2 |
| Charades sample (existing) | handheld | 126 | 19 | 20 | – | labels 154 / 151 |

All data is downloaded into gitignored `data/validation/<dataset>/` in the shared format of `scripts/datasets.py` (one `annotations.json` per dataset; schema validation, sha1 split on `dataset:video_id`; Charades keeps its bare-id legacy split so the v0.3 tables reproduce exactly). **Videos, frames, skeletons and landmark caches are never committed.** Both datasets are **academic / research use only** (PKU-MMD: CC BY-NC 4.0). Cite: Xia L., Chen C.-C., Aggarwal J.K., *View invariant human action recognition using histograms of 3D joints*, CVPRW 2012; Liu C., Hu Y., Li Y., Song S., Liu J., *PKU-MMD: A Large Scale Benchmark for Continuous Multi-Modal Human Action Understanding*, arXiv:1703.07475, 2017.

```bash
python scripts/fetch_fixed_camera_datasets.py utkinect pkummd2       # ~1.8 GB + ~7.5 GB downloaded, ~1 h with transcoding
python scripts/fetch_fixed_camera_datasets.py pkummd1 --limit 85 --stride 4   # the evaluated Phase 1 subset
python scripts/validate_real_video.py --split holdout --dataset charades,utkinect,pkummd1,pkummd2 --json out.json
python scripts/skeleton_adjudication.py out.json                     # needs the raw Kinect skeletons in data/raw/
python scripts/tune_grid.py --dataset charades,utkinect,pkummd1,pkummd2 --split tune --grid '{"dwell_s": [0.8, 1.0]}'
```

**How the data was obtained, and what failed.**
- *UTKinect timing.* The site says "30 f/s … final frame rate about 15 f/sec" and frame numbers have gaps. Frame numbers are always even, and the zip's per-file modification times are linear in frame number at 58–60 numbers/s in all 20 sequences (residual ≤ 2.5 s, the zip timestamp resolution). So capture is 30 fps with the counter advancing by 2: **t = (frame − first frame) / 60**, ~13 fps effective. Viewing frames confirms it: sit-down and stand-up label windows line up with the motion to within a frame or two. (A first physical estimate from the skeleton, peak hip velocity, had favoured 30 numbers/s; the timestamps settled it.) Each sequence became a constant 30 fps MP4 in which each recorded image is held until the next recorded frame number (ffmpeg concat with per-frame durations), so video time equals capture time; untracked gaps show a frozen image. One duplicate file (`colorImg2040 - Copy.jpg`) is skipped.
- *PKU-MMD access.* The Phase 1 Google Drive link on the project page returns 401 without a `resourcekey`; the key is published on the mirror page (`http://39.96.165.147/Projects/PKUMMD/PKU-MMD.html`). Files download directly from `drive.usercontent.google.com` (no `gdown` needed; Drive answered a few bursts with HTTP 500, handled by retries). Labels need `7z`.
- *PKU-MMD Phase 1 is a subset.* Front-view AVIs are ~98 MB each (~35 GB for 359), more than the session's free disk, so each is transcoded to 960×540 at 10 fps and deleted. Transcoding 1080p cost ~2.5 CPU-min per video; to stay within a sane time budget the run was stopped at a deterministic subset: the first 85 videos plus every 4th of the sorted list (153 videos, all subjects represented). Pose models resize to ≤ 256 px internally and validation samples at 5 fps, so neither change should affect landmarks materially.
- *Mappings checked by viewing frames* (scratch only, never committed): UTKinect *waveHands* is both arms swung in large overhead arcs, and the Kinect skeleton has a wrist above the shoulder continuously for ≥ 0.83 s in every one of the 20 windows (median fraction of the window 0.83) → `hand_raised`. PKU-MMD *hand waving* raised one hand above the shoulder for > 0.6 s in all 6 viewed instances (3 per phase) → `hand_raised`. PKU-MMD *cheer up* did **not** consistently (arms overhead ~2 s, ~0.5 s, or never above the shoulders in 3 viewed) → `other`. Stand-up/sit-down windows matched the motion in all viewed instances; PKU-MMD Phase 2 windows are tight (often ~0.5 s).
- *Split caveat.* The split is per video, as specified, so the same subject appears in both halves (UTKinect: the subject's two sequences; PKU-MMD: 13–66 subjects across many sequences). The rules have few parameters, so leakage risk is small, but it is not a subject-independent test.

### Candidates A–D on the new data (measurement, no threshold change)

Same pre-registered candidates as *rules-v0.3*: **A** v0.2 attribution, masking off · **B** v0.2 attribution + masking · **C** tracking, masking off (**shipped rules-v0.3**) · **D** tracking + masking. `pose_landmarker_lite`, 5 fps, `--tol 1.0`. `standing_transition`, 95 % Wilson intervals in brackets:

| Dataset | Split | A | C (rules-v0.3) | B / D |
|---|---|---|---|---|
| UTKinect | holdout (13 videos) | recall 0.85 (11/13) [0.58–0.96], precision 0.92 (11/12) | recall 0.77 (10/13) [0.50–0.92], precision 0.91 (10/11) [0.62–0.98] | B = A, D = C |
| UTKinect | all (20) | 0.75 (15/20), 0.94 (15/16) | 0.65 (13/20) [0.43–0.82], 0.93 (13/14) [0.69–0.99] | B = A, D = C |
| PKU-MMD P1 | holdout (79) | 0.57 (23/40), 0.62 (23/37) | 0.57 (23/40) [0.42–0.71], 0.62 (23/37) [0.46–0.76] | B = A, D = C |
| PKU-MMD P1 | all (153) | 0.66 (45/68), 0.60 (45/75) | 0.66 (45/68) [0.54–0.76], 0.61 (45/74) [0.49–0.71] | B = A, D = C |
| PKU-MMD P2 | holdout (168) | 0.68 (39/57), 0.48 (39/82) | 0.68 (39/57) [0.56–0.79], 0.48 (39/82) [0.37–0.58] | B = A, D = C |
| PKU-MMD P2 | all (333) | 0.75 (80/107), 0.50 (80/160) | 0.75 (80/107) [0.66–0.82], 0.50 (80/160) [0.42–0.58] | B = A, D = C |
| **Fixed cameras pooled** | holdout (260) | 0.69 (73/110), 0.56 (73/131) | **0.65 (72/110) [0.56–0.74], 0.55 (72/130) [0.47–0.64]** | B = A, D = C |
| **Fixed cameras pooled** | all (506) | 0.72 (140/195), 0.55 (140/251) | **0.71 (138/195) [0.64–0.77], 0.56 (138/248) [0.49–0.62]** | B = A, D = C |
| Charades | holdout | 0.40 (4/10), 0.40 (4/10) | 0.20 (2/10), 0.40 (2/5) | B 3/10, 3/6 · D 2/10, 2/4 |
| **All four pooled** | tune (311) | 0.72 (68/94), 0.56 (68/122), F1 0.630 | 0.71 (67/94), 0.56 (67/119), F1 0.629 | B = A, D = C |
| **All four pooled** | holdout (321) | 0.64 (77/120), 0.55 (77/141), F1 0.590 | 0.62 (74/120) [0.53–0.70], 0.55 (74/135) [0.46–0.63], F1 0.580 | B 0.591, D 0.583 |

The Charades rows reproduce the *rules-v0.3* table exactly (through the new `--dataset` path as well as the legacy one). Confusions with sitting down: 11 of 248 C detections (all splits), 0 on Charades.

**Observability.** Median share of frames where posture is observable: UTKinect 0.76, PKU-MMD 0.91–0.93, Charades 0.20 (C). On fixed, full-body footage most stand-ups are observable (`recall_when_legs_observable` ≈ recall), so the remaining misses are classification errors, not missing evidence.

**Camera motion.** The estimator reports **no moving frame in any of the 506 fixed-camera videos** (mean, median and max `camera_moving_ratio` = 0.0), and B/D are identical to A/C to the count on all three fixed-camera datasets in every split. Masking is inert when the camera is fixed, as it should be; it only acts on Charades.

**Tracking on single-person video.** Tracking (C) costs 2 of 20 UTKinect stand-ups against A, and ties on PKU-MMD. In UTKinect s03_e01 (one person, fixed camera) the person is split into four tracks across untracked gaps; the subject track chosen (22–66 s) excludes the stand-up at ~10 s, so those frames are unobservable. The largest-person sequence shows a clear seated run before the stand-up. Track fragmentation, not posture, causes this miss.

#### `hand_raised`

Against the action labels, `hand_raised` looks terrible on PKU-MMD (C, all: 103/109 label windows found, but 1,801 detections, F1 0.15). Visual review shows why: *drink water*, *taking a selfie*, *put on a hat*, *wear jacket*, *brushing hair*, *make a phone call* and *salute* all put a hand above the shoulder for ≥ 0.6 s, which is the PRD's definition. UTKinect labels only one segment per action (in s09_e02 the subject waves overhead from 42 s to 72 s, but only 70.8–73.5 s is labelled). **Labels are not operational definitions here (M18).**

Both datasets ship **Kinect skeletons** recorded by a separate depth sensor. `scripts/skeleton_adjudication.py` derives reference windows where the rule's definition holds in 3D (either wrist ≥ 0.5 head units above its shoulder for ≥ 0.6 s, gaps ≤ 0.4 s bridged; single-subject videos only) and scores detections against them:

| Dataset | Split | C recall (skeleton windows) | C precision | A recall | A precision |
|---|---|---|---|---|---|
| UTKinect | all (20) | 0.89 (48/54) [0.78–0.95] | 0.89 (50/56) [0.79–0.95] | 0.94 (51/54) | 0.90 (53/59) |
| PKU-MMD P1 | all (121 single-subject) | 0.83 (857/1028) [0.81–0.86] | 0.89 (885/998) [0.87–0.91] | 0.86 | 0.89 |
| PKU-MMD P2 | all (333) | 0.73 (580/800) [0.69–0.76] | 0.87 (575/663) [0.84–0.89] | 0.74 | 0.86 |
| **Pooled** | **holdout (248)** | **0.79 (821/1038) [0.77–0.82]** | **0.88 (835/946) [0.86–0.90]** | – | – |

**The reference is noisy, so read these as bounds.** Frame review of sampled disagreements:
- Detections the skeleton calls false (4 viewed): all borderline hand-at-head poses (drinking, phone at the ear, checking a watch with the arm raised), where either answer is defensible. True precision is likely ≥ 0.88.
- Skeleton windows without a detection (12 viewed, before and after a fix): the first version counted bending (sitting down, picking up) as "raised", because bending collapses the head-to-shoulder height; it now uses each video's median head unit. After the fix, most sampled misses were still **skeleton errors on seated or bent people** (hands at the chest, at the feet or at the mouth); one was a real MediaPipe miss (an arm overhead while taking off a jacket). **Recall is therefore a lower bound.**

### Held-out disagreements reviewed visually (`standing_transition`, C)

| Sample | Finding |
|---|---|
| PKU-MMD P2, 6 false positives | **2 annotation gaps** (real stand-ups at the start of a sequence or between scripted actions with no *standing up* label: A25N11, A23N11); **4 rule errors on seated people**, 2 of them turning sideways on a swivel chair (side view), 2 seated frontally and still. |
| PKU-MMD P1, 4 false positives | **2 rises from the floor after the scripted *falling*** (a real transition, labelled *falling*, not *standing up*: a definition gap); 2 rule errors on seated people. |
| PKU-MMD P2, 4 misses | All clear frontal, full-body stand-ups. Diagnosis (A11N12): seated frames have hip angle 146–149°, hip-to-knee drop 0.35–0.44 and knee angle 133–139°, each **just outside** the v0.2 seated thresholds (≤ 140°, ≤ 0.35, ≤ 130°) that were chosen on elevated Charades views; seated frames read as *ambiguous*, so no seated dwell precedes the stand-up. |
| UTKinect, 3 misses and the 1 false positive (holdout) | s03_e01: track fragmentation (above). s10_e01: the same seated ambiguity (hip angle 141–146°, and a hip-to-knee drop of ≈ 0.5 that reaches `seated_drop_ceiling`). s05_e01 not diagnosed. The only UTKinect "false positive" (s09_e01, 16.0–16.6 s) is a **real, unlabelled stand-up**: the subject sits and stands twice, and only the second is labelled (detected separately at 20 s). With it, UTKinect adjudicated precision is 11/11 (holdout) and 14/14 (all 20 videos). |

The authoring agent did this adjudication; treat adjudicated counts as indicative.

### Phase 2: selection on the pooled tune split (pre-registered above)

The pooled tune split has **94** `standing_transition` windows (≥ 25), so Phase 2 ran. All 96 settings passed the `hand_raised` guard (label F1 varies by < 0.002). Highest pooled-tune F1 = 0.667, reached by four settings; the tie-break (fewest tune false positives, then fewest changes) selects:

> **rules-v0.4 candidate: `dwell_s = 1.0`, `max_transition_s = 7`, `visibility_threshold = 0.4`, `mask_camera_motion = false`** (all else rules-v0.3). Tune: recall 0.72 (68/94), precision 0.62 (68/110), F1 0.667, against rules-v0.3's 0.71, 0.56, 0.629.

**Held-out half, scored once:**

| | Pooled recall | Pooled precision | FP | Sit-down conf. | F1 | Charades | UTKinect | PKU-MMD P1 | PKU-MMD P2 |
|---|---|---|---|---|---|---|---|---|---|
| rules-v0.3 | 0.62 (74/120) [0.53–0.70] | 0.55 (74/135) [0.46–0.63] | 61 | 6 | 0.580 | 2/10, 2/5 | 10/13, 10/11 | 23/40, 23/37 | 39/57, 39/82 |
| **v0.4 candidate** | 0.60 (72/120) [0.51–0.68] | 0.57 (72/126) [0.48–0.65] | 54 | 3 | 0.585 | 4/10, 4/9 | 10/13, 10/11 | 22/40, 22/34 | 36/57, 36/72 |

`hand_raised` is unchanged (skeleton reference, holdout: recall 0.793 vs 0.791, precision 0.881 vs 0.883).

**Verdict: the tune gain did not transfer.** Held-out F1 is flat (+0.005): 7 fewer false positives and 3 fewer sit-down confusions for 2 fewer stand-ups, all well inside the intervals. Charades gains 2 held-out hits but had lost its one tune hit under the same setting, which is noise at n = 10. **This does not justify changing the defaults**; it is recorded as the candidate the pre-registered rule produced. The temporal and visibility parameters are not where the remaining error is.

### Exploratory (tune split only, not a selection, holdout untouched)

Because the frame review pointed at the seated thresholds, an 8-setting sweep over `seated_max_drop` {0.35, 0.45} × `seated_max_hip_angle_deg` {140, 148} × `seated_max_knee_angle_deg` {130, 140} was run on the **pooled tune split only**: `seated_max_drop = 0.45` raises pooled-tune F1 from 0.629 to **0.725** (recall 75/94, precision 75/113), with gains on both PKU-MMD phases and UTKinect and no change on Charades tune (1/9). This is a **lead for the next pre-registered cycle**, not evidence: it was chosen after looking at errors, and the holdout has deliberately not been scored with it.

### Limitations

- Fixed-camera data is lab footage of adults performing scripted actions (Kinect labs in Austin and Beijing), not ABA sessions; people are small in PKU-MMD frames (about a third of the frame height).
- Splits are per video, not per subject (see *Data*).
- PKU-MMD Phase 1 is a 153-of-359 subset.
- Labels are incomplete for stand-ups (unlabelled rises at sequence starts and after *falling*), so raw precision underestimates; the skeleton reference for `hand_raised` is itself noisy for seated or bent people, so its recall is a lower bound.
- All adjudication was done by the authoring agent.
- **This is engineering evidence, not clinical validation.**

## Frontal posture thresholds and track merging (Phase 3, pre-registered)

Run date: 2026-10-20. **Question:** the fixed-camera frame review (*Held-out disagreements*, LEARNINGS M42, M44) pointed at two levers: (1) the seated thresholds, chosen on elevated Charades views, leave frontal seated frames *ambiguous*; (2) one person split into several tracks across untracked gaps loses frames outside the subject track. Does either lever, selected on the tune split under a rule written in advance, improve detection on held-out and never-seen footage without weakening the H-03 control?

**Code added for this cycle (behaviour unchanged by default).** `TrackingParams.merge_fragments` (default **off**) runs `tracking.merge_fragments` after tracking: tracks are taken in order of first appearance, and each joins the best-matching chain that **ended before it started** (never a chain present in any frame it is present in), within `merge_max_gap_s = 10`, `merge_max_center_dist = 1.0` track sizes and `merge_max_size_ratio = 2.0` (median box over 3 frames at each end). These gate values were set a priori from the existing tracking gate (same distance unit) and were not tuned. Tests: `tests/test_tracking.py` (fragmented single person merges and regains its stand-up; people taking turns at different positions or sizes stay apart; people present at the same time are never merged, even with gates that accept any geometry; K5CRM pattern and a size-mismatched handover produce no stand-up). `scripts/fetch_fixed_camera_datasets.py pkummd1_fresh --limit 85 --stride 4` fetches the complement of the evaluated Phase 1 subset; `scripts/tune_grid.py --tracking-params` scores a grid with tracking overrides.

### Pre-registration (written and committed before any tune result for this grid was computed)

> Disclosure: the levers come from viewing errors (fixed-camera holdout review, and an exploratory 8-setting tune-only seated sweep where `seated_max_drop = 0.45` raised pooled-tune F1 0.629 → 0.725). This is the **third use** of the existing pooled holdout (rules-v0.3 measurement, Phase 2, now). The fresh set below has never been viewed or scored by anyone.
>
> **Data.** Pooled **tune** split = tune halves of Charades (legacy key), UTKinect, PKU-MMD Phase 1 (the evaluated 153-video subset, `--limit 85 --stride 4`) and Phase 2 (front view), as in Phase 2. **Fresh confirmation set** = `pkummd1_fresh`: every labelled Phase 1 front-view video **not** in that subset (206 videos, all scored, no split), same source, labels and transcoding. It is fetched and its landmarks are extracted, but no rule output, frame or result from it is looked at until the single final confirmation scoring below. Proceed with confirmation only if ≥ 150 fresh videos were fetched (report the count).
>
> **Fixed.** `pose_landmarker_lite`, 5 fps, tracked attribution (rules-v0.3), all other RuleParams and TrackingParams at rules-v0.3 values, merge gates as above, `--tol 1.0`.
>
> **Grid (48 settings).** `seated_max_drop` ∈ {0.35, 0.40, 0.45} × `seated_max_hip_angle_deg` ∈ {140, 148} × `seated_max_knee_angle_deg` ∈ {130, 140} × `seated_drop_ceiling` ∈ {0.5, 0.6} × `merge_fragments` ∈ {false, true}. rules-v0.3 is (0.35, 140, 130, 0.5, false).
>
> **Objective.** Highest pooled-tune `standing_transition` F1 (micro, from pooled counts). **Guard:** pooled-tune `hand_raised` F1 against labels must not fall more than 0.02 below rules-v0.3's. **Ties** (F1 within 0.005 of the best): fewer pooled-tune false positives, then fewest parameters changed from rules-v0.3 (`merge_fragments` counts as one), then the smallest total distance from rules-v0.3 in grid steps.
>
> **Scoring, once each.** The selected setting **S** and rules-v0.3 (**C**) on the existing pooled holdout (per dataset and pooled) and on the fresh set; **A** (v0.2 attribution, masking off) on the fresh set for context. 95 % Wilson intervals. If S changes both levers, single-lever rows are reported afterwards as *post-hoc* and do not enter the decision.
>
> **Adoption rule (decided now).** Recommend adopting S as rules-v0.4 only if **all** hold: (1) fresh-set `standing_transition` F1(S) − F1(C) ≥ +0.01; (2) pooled-holdout F1(S) ≥ F1(C) − 0.02 (not worse beyond noise); (3) `hand_raised` label F1 on the fresh set and on the holdout not more than 0.02 below C; (4) H-03: if S merges tracks, every Charades detection (all splits) that S produces and C does not is viewed, and none stitches evidence from two different people; and Charades held-out false positives do not increase. If (1)–(3) hold for S but (4) fails, re-score nothing: recommend only the seated part, flagged as not separately confirmed. Otherwise recommend keeping rules-v0.3. `config.py` defaults and `ALGORITHM_VERSION` are not changed in this cycle either way; adoption is the maintainer's decision.
>
> **Also reported.** Whether S changes Charades (elevated, handheld) behaviour on either split, and whether one threshold set can plausibly serve both geometries. A sample of fresh-set and holdout disagreements is reviewed visually after scoring (scratch only, never committed).

> **Amendment 1 (committed before any tune result for this grid was computed).** Google Drive's per-file download quota blocked 61 of the 153 evaluated Phase 1 videos and 75 of the 206 fresh ones (HTTP 200 with a "Quota exceeded" page; the fetcher now rejects it, `DriveQuotaError`). Which files were blocked depends on download order, not on any result. So: (a) PKU-MMD Phase 1 in this cycle is the **92 videos available at this commit** (0002–0086, 0090, 0094, 0106, 0126, 0194, 0226, 0230: 46 tune, 46 holdout); the other three datasets are complete. Selection and the one holdout scoring use exactly these, and videos fetched later are not added to this cycle's tune or holdout. Phase 2 tune and holdout numbers are therefore not directly comparable to this cycle's; the comparison that matters, S against C, is paired on the same videos. (b) The fresh set rule stands: score it once, only when ≥ 150 fresh videos are present. If it stays below 150, the verdict is **"confirmation deferred (n < 150)" with no adoption recommendation**, and the fetch/extract commands for the remainder are documented for a later session.

### Phase 3 selection on the pooled tune split

Pooled tune split under Amendment 1: **283 videos, 80 `standing_transition` windows** (Charades 9, UTKinect 7, PKU-MMD P1 14, P2 50). All 48 settings pass the `hand_raised` guard (label F1 0.137–0.138 against C's 0.138). Best pooled-tune F1 0.709; six settings lie within 0.005 of it, and the tie-break (fewest tune false positives) selects:

> **S: `seated_max_drop = 0.45`**, all other RuleParams and TrackingParams at rules-v0.3 (`seated_max_hip_angle_deg = 140`, `seated_max_knee_angle_deg = 130`, `seated_drop_ceiling = 0.5`, `merge_fragments = false`).

| Tune (283 videos) | Recall | Precision | FP | F1 | Charades | UTKinect | PKU-MMD P1 | PKU-MMD P2 |
|---|---|---|---|---|---|---|---|---|
| C rules-v0.3 | 0.70 (56/80) | 0.55 (56/101) | 45 | 0.619 | 1/9, 1/1 | 3/7, 3/3 | 11/14, 11/19 | 41/50, 41/78 |
| **S** | 0.78 (62/80) | 0.65 (62/95) | 33 | **0.709** | 1/9, 1/1 | 4/7, 4/4 | 13/14, 13/18 | 44/50, 44/72 |
| *merge only (C + merge)* | 0.70 (56/80) | 0.55 (56/102) | 46 | 0.615 | 1/9, 1/2 | 3/7, 3/3 | 11/14, 11/19 | 41/50, 41/78 |
| *S + merge* | 0.78 (62/80) | 0.65 (62/96) | 34 | 0.705 | 1/9, 1/2 | 4/7, 4/4 | 13/14, 13/18 | 44/50, 44/72 |

(Each row is recall then precision. The italic rows are grid rows shown for context; they are not selections.) The other ties differ only in the hip or knee threshold, or in a ceiling of 0.6 that buys one stand-up for four false positives. **Track merging adds nothing on the tune half**: no tune stand-up depends on a fragmented track, and it adds one Charades false positive (reviewed below). **The seated change does not alter Charades tune** (1/9, 1/1 either way). This repeats the exploratory lead on the reduced tune set (0.629 → 0.725 on the full Phase 2 tune set), which is expected, since it is the same lever chosen on overlapping data.
