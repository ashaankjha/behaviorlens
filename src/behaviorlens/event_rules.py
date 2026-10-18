"""Interpretable temporal rules that turn per-frame states into candidate events.

Pipeline: per-frame features -> per-frame candidate states -> gap bridging
(debounce) -> temporal segments -> minimum-duration filter -> event records.

These are engineering heuristics, not validated clinical labels. Events carry
the rule name, parameters and an evidence summary so a reviewer can see why a
suggestion was made; ``confidence`` stays null because no score here is a
calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ALGORITHM_VERSION, RuleParams
from .features import (
    SEATED,
    STANDING,
    TRUE,
    UNKNOWN,
    hand_raise_features,
    posture_features,
)
from .models import Event, PoseSequence


@dataclass(frozen=True)
class Segment:
    """Inclusive frame-index range [first, last] of a bridged run."""

    first: int
    last: int


def find_segments(mask: np.ndarray, times: np.ndarray, max_gap_s: float) -> list[Segment]:
    """Runs of True frames, bridging interruptions no longer than ``max_gap_s``.

    The gap is measured between the last True frame of one run and the first
    True frame of the next, so a single noisy frame inside a long run does not
    split it.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    segments = []
    start = prev = idx[0]
    for i in idx[1:]:
        if times[i] - times[prev] > max_gap_s + 1e-9:
            segments.append(Segment(int(start), int(prev)))
            start = i
        prev = i
    segments.append(Segment(int(start), int(prev)))
    return segments


def segment_bounds(seg: Segment, seq: PoseSequence) -> tuple[float, float]:
    """Start/end seconds for a segment. Each sample covers one sampling period."""
    start = float(seq.timestamps_s[seg.first])
    end = float(seq.timestamps_s[seg.last]) + seq.frame_period_s
    return round(start, 3), round(min(end, max(seq.duration_s, start + 1e-3)), 3)


def _ratio(values: np.ndarray) -> float:
    return round(float(np.mean(values)), 3) if values.size else 0.0


def detect_hand_raised(seq: PoseSequence, params: RuleParams, session_id: str) -> list[Event]:
    """`hand_raised`: wrist above same-side shoulder for >= hand_raise_min_s.

    Both hands raised together form one contiguous event with side "both".
    """
    feats = hand_raise_features(seq, params)
    combined = feats.combined
    events = []
    for seg in find_segments(combined == TRUE, seq.timestamps_s, params.merge_gap_s):
        start, end = segment_bounds(seg, seq)
        if end - start < params.hand_raise_min_s - 1e-9:
            continue
        sl = slice(seg.first, seg.last + 1)
        left_up = bool(np.any(feats.left[sl] == TRUE))
        right_up = bool(np.any(feats.right[sl] == TRUE))
        side = "both" if left_up and right_up else ("left" if left_up else "right")
        # Visibility of the side(s) that were actually raised, not of the idle arm.
        raised_sides = [s for s, up in ((feats.left[sl], left_up), (feats.right[sl], right_up)) if up]
        side_visible = np.all([s != UNKNOWN for s in raised_sides], axis=0)
        margins = np.concatenate([feats.left_margin[sl][feats.left[sl] == TRUE],
                                  feats.right_margin[sl][feats.right[sl] == TRUE]])
        evidence = {
            "rule": "wrist_above_shoulder",
            "side": side,
            "frames_in_segment": seg.last - seg.first + 1,
            "raised_frame_ratio": _ratio(combined[sl] == TRUE),
            "visible_frame_ratio": _ratio(side_visible),
            "median_wrist_height_head_units": round(float(np.nanmedian(margins)), 3) if margins.size else None,
            "params": {
                "visibility_threshold": params.visibility_threshold,
                "hand_raise_min_height": params.hand_raise_min_height,
                "hand_raise_min_s": params.hand_raise_min_s,
                "merge_gap_s": params.merge_gap_s,
            },
        }
        events.append(_event(session_id, "hand_raised", start, end, evidence, seq))
    return events


def _stable_runs(state: np.ndarray, value: int, seq: PoseSequence, params: RuleParams) -> list[Segment]:
    runs = find_segments(state == value, seq.timestamps_s, params.posture_merge_gap_s)
    return [
        r for r in runs
        if segment_bounds(r, seq)[1] - segment_bounds(r, seq)[0] >= params.dwell_s - 1e-9
    ]


def detect_standing(seq: PoseSequence, params: RuleParams, session_id: str) -> list[Event]:
    """`standing_transition` and optional `sustained_standing` candidates.

    A transition needs a stable seated-like run immediately followed (within
    max_transition_s) by a stable standing-like run, each held for dwell_s.
    Frames where hips/knees are not visible are unobservable and can never
    supply either side of the evidence.
    """
    feats = posture_features(seq, params)
    seated = _stable_runs(feats.state, SEATED, seq, params)
    standing = _stable_runs(feats.state, STANDING, seq, params)
    times = seq.timestamps_s
    events = []

    runs = sorted([(r.first, "seated", r) for r in seated] + [(r.first, "standing", r) for r in standing])
    for (_, kind_a, a), (_, kind_b, b) in zip(runs, runs[1:]):
        if kind_a != "seated" or kind_b != "standing" or b.last <= a.last:
            continue
        after = np.flatnonzero(feats.state[a.last + 1: b.last + 1] == STANDING)
        if after.size == 0:
            continue
        first_up = a.last + 1 + int(after[0])
        gap = times[first_up] - times[a.last]
        if gap > params.max_transition_s + 1e-9:
            continue
        between = feats.state[a.last + 1: first_up]
        start = round(float(times[a.last]), 3)
        end = round(min(float(times[first_up]) + seq.frame_period_s, seq.duration_s), 3)
        if end <= start:
            continue
        evidence = {
            "rule": "seated_then_standing",
            "seated_before_s": round(float(times[a.last] - times[a.first]) + seq.frame_period_s, 3),
            "standing_after_s": round(float(times[b.last] - times[first_up]) + seq.frame_period_s, 3),
            "transition_gap_s": round(float(gap), 3),
            "unobservable_frames_in_transition": int(np.sum(between == UNKNOWN)),
            "median_hip_knee_drop_before": _nanmedian(feats.hip_knee_drop[a.first: a.last + 1]),
            "median_hip_knee_drop_after": _nanmedian(feats.hip_knee_drop[first_up: b.last + 1]),
            "median_hip_angle_before_deg": _nanmedian(feats.hip_angle_deg[a.first: a.last + 1]),
            "median_hip_angle_after_deg": _nanmedian(feats.hip_angle_deg[first_up: b.last + 1]),
            "params": {
                "visibility_threshold": params.visibility_threshold,
                "seated_max_drop": params.seated_max_drop,
                "seated_max_hip_angle_deg": params.seated_max_hip_angle_deg,
                "seated_max_knee_angle_deg": params.seated_max_knee_angle_deg,
                "standing_min_drop": params.standing_min_drop,
                "standing_min_hip_angle_deg": params.standing_min_hip_angle_deg,
                "dwell_s": params.dwell_s,
                "posture_merge_gap_s": params.posture_merge_gap_s,
                "max_transition_s": params.max_transition_s,
            },
        }
        events.append(_event(session_id, "standing_transition", start, end, evidence, seq))

    if params.enable_sustained_standing:
        for r in standing:
            start, end = segment_bounds(r, seq)
            if end - start < params.sustained_standing_min_s - 1e-9:
                continue
            sl = slice(r.first, r.last + 1)
            evidence = {
                "rule": "upright_pose_sustained",
                "standing_frame_ratio": _ratio(feats.state[sl] == STANDING),
                "visible_frame_ratio": _ratio(feats.state[sl] != UNKNOWN),
                "median_hip_knee_drop": _nanmedian(feats.hip_knee_drop[sl]),
                "params": {
                    "sustained_standing_min_s": params.sustained_standing_min_s,
                    "standing_min_drop": params.standing_min_drop,
                    "max_torso_tilt_deg": params.max_torso_tilt_deg,
                },
            }
            events.append(_event(session_id, "sustained_standing", start, end, evidence, seq))
    return events


def observability_stats(seq: PoseSequence, params: RuleParams) -> dict:
    """Fraction of frames in which each rule had enough landmark evidence."""
    n = max(len(seq), 1)
    hands = hand_raise_features(seq, params).combined
    posture = posture_features(seq, params).state
    return {
        "frames": len(seq),
        "pose_detected_ratio": round(float(np.sum(seq.has_pose)) / n, 3),
        "hand_rule_observable_ratio": round(float(np.sum(hands != UNKNOWN)) / n, 3),
        "posture_rule_observable_ratio": round(float(np.sum(posture != UNKNOWN)) / n, 3),
        "multi_person_frame_ratio": round(float(np.sum(np.asarray(seq.num_people) > 1)) / n, 3),
    }


def detect_all(seq: PoseSequence, params: RuleParams, session_id: str) -> list[Event]:
    events = detect_hand_raised(seq, params, session_id) + detect_standing(seq, params, session_id)
    return sorted(events, key=lambda e: (e.start_s, e.event_type))


def _event(session_id: str, event_type: str, start: float, end: float, evidence: dict,
           seq: PoseSequence) -> Event:
    return Event(
        session_id=session_id,
        event_type=event_type,
        start_s=start,
        end_s=end,
        source="algorithm",
        review_status="pending",
        evidence=evidence,
        algorithm_version=ALGORITHM_VERSION,
        data_mode=seq.data_mode,
    )


def _nanmedian(values: np.ndarray) -> float | None:
    values = values[~np.isnan(values)]
    return round(float(np.median(values)), 3) if values.size else None
