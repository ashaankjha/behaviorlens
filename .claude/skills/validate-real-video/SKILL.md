---
name: validate-real-video
description: Measure BehaviorLens event rules on real moving-person video (Charades stand-ups, UCF101 arm movements) with real MediaPipe inference, using a fixed tune/holdout split. Use whenever a rule, threshold, feature or the pose model changes, before claiming detection accuracy, or when asked whether the rules work on real footage.
---

# Validate rules on real video

Synthetic tests prove the plumbing, not the detection claim (LEARNINGS M15). Run this for every rule change.

## Steps

1. Fetch the clips once (~220 MB, ~2 min, into the gitignored `data/validation/`; never commit them):
   ```bash
   .venv/bin/python scripts/fetch_validation_clips.py
   ```
2. Score the **current** rules and the **previous** version on the held-out half. Express previous thresholds through `--params`, for example the v0.1 equivalent:
   ```bash
   .venv/bin/python scripts/validate_real_video.py --split holdout --json /tmp/new.json
   .venv/bin/python scripts/validate_real_video.py --split holdout --json /tmp/old.json --params \
     '{"seated_max_hip_angle_deg":0,"seated_max_knee_angle_deg":0,"seated_drop_ceiling":99,"standing_min_drop":0.6,"standing_min_hip_angle_deg":0,"dwell_s":1.0,"max_transition_s":3.0}'
   ```
   The first run computes and caches landmarks (~2.5 min on 4 cores); later runs take seconds.
3. **Tuning discipline (M19).** Explore and select thresholds with `--split tune` only. Score `--split holdout` once with the chosen values. Present any holdout sweep as post-hoc sensitivity, never as the selection.
4. **Look before you change rules (M18).** For each disagreement, extract frames at the event or annotation times with OpenCV and view them with the Read tool. Decide whether it's a rule error, an annotation gap, or a definition mismatch. Keep viewed frames in scratch and never commit them.
5. Update `docs/VALIDATION.md` with numbers recomputed by the **production code path** (M20), counts in brackets, and Wilson intervals for small samples. Then update `docs/PRD_TRACEABILITY.md` (FR3 rows) and log any lesson with the `update-learnings` skill.
6. If thresholds changed, bump `ALGORITHM_VERSION` in `config.py`, comment each threshold's source, and add a synthetic regression test reproducing the real-data pattern (see `test_angled_seated_pose_is_seated_via_hip_angle`).

## Metric definitions

- **Charades `standing_transition`**: a detection is correct if it overlaps a label-154 window widened by `--tol` (default 1 s). Recall is over windows. `detections_during_sit_down` counts confusions with label 151. `recall_when_legs_observable` restricts to clips with posture observable in ≥30% of frames.
- **UCF101 `hand_raised`**: per clip, with any event counting as positive. Class labels are a proxy: JumpingJack raises are often shorter than the 0.6 s minimum.

## Known caveats

Small samples (19 stand-ups), home and sports footage rather than ABA settings, and incomplete Charades annotations (real stand-ups sometimes go unlabelled). Say so whenever you report the numbers.
