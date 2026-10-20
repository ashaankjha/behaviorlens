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

- Record staged clips in the intended setting (front-on camera, full body visible) using the README procedure, and add them to `data/validation/` with a ground-truth CSV.
- ~~Try `pose_landmarker_full`/`heavy` and 10 fps.~~ Done (see *Pose model comparison*): neither larger model helps with the current thresholds, and 10 fps doesn't help `hand_raised`. Re-tuning the temporal parameters for 10 fps on the tune split is the remaining lead.
- ~~Add person tracking.~~ Done (see *Person tracking*): it removes identity-switch false positives but is fragile under camera pans. Integrating camera-motion detection is the next step.
