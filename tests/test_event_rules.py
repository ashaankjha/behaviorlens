import numpy as np
import pytest

from behaviorlens.config import RuleParams
from behaviorlens.event_rules import detect_hand_raised, detect_standing, find_segments
from behaviorlens.features import FALSE, SEATED, STANDING, TRUE, UNKNOWN, hand_raise_features, posture_features

from conftest import LEG_IDS, make_seq, window

P = RuleParams()


def of_type(events, t):
    return [e for e in events if e.event_type == t]


# ------------------------------------------------------------ segments
def test_find_segments_bridges_short_gaps_only():
    times = np.arange(10) * 0.2
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], bool)
    segs = find_segments(mask, times, max_gap_s=0.4)
    assert [(s.first, s.last) for s in segs] == [(0, 4), (8, 9)]


def test_find_segments_empty():
    assert find_segments(np.zeros(5, bool), np.arange(5) * 0.2, 0.4) == []


# ------------------------------------------------------------ hand_raised
def test_hand_raised_basic_right():
    seq = make_seq(10, right_up=window(3, 5))
    events = detect_hand_raised(seq, P, "s")
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "hand_raised" and e.evidence["side"] == "right"
    assert e.start_s == pytest.approx(3.0, abs=0.21) and e.end_s == pytest.approx(5.0, abs=0.21)
    assert e.confidence is None and e.review_status == "pending" and e.source == "algorithm"
    assert e.evidence["rule"] == "wrist_above_shoulder" and "params" in e.evidence


def test_wrist_above_shoulder_uses_image_coordinates():
    seq = make_seq(2, right_up=window(0, 2))
    f = hand_raise_features(seq, P)
    assert np.all(f.right == TRUE) and np.all(f.left == FALSE)
    # positive margin means the wrist has a smaller y (higher in the image) than the shoulder
    assert np.all(f.right_margin > 0)


def test_short_spike_below_minimum_is_ignored():
    seq = make_seq(6, left_up=window(2.0, 2.4))  # 0.4 s < 0.6 s
    assert detect_hand_raised(seq, P, "s") == []


def test_exactly_minimum_duration_counts():
    seq = make_seq(6, left_up=window(2.0, 2.6))
    assert len(detect_hand_raised(seq, P, "s")) == 1


def test_both_hands_form_one_contiguous_event():
    seq = make_seq(8, left_up=window(2, 4), right_up=window(2.4, 4.6))
    events = detect_hand_raised(seq, P, "s")
    assert len(events) == 1
    assert events[0].evidence["side"] == "both"
    assert events[0].start_s == pytest.approx(2.0, abs=0.01)


def test_occluded_wrist_produces_no_event():
    seq = make_seq(8, right_up=window(2, 5), hidden=lambda t: {12, 16})
    f = hand_raise_features(seq, P)
    assert np.all(f.right == UNKNOWN)
    assert detect_hand_raised(seq, P, "s") == []


def test_single_dropped_frame_is_bridged():
    seq = make_seq(8, right_up=window(2, 5), hidden=lambda t: {16} if 3.0 <= t < 3.2 else set())
    events = detect_hand_raised(seq, P, "s")
    assert len(events) == 1
    assert events[0].evidence["visible_frame_ratio"] < 1.0


def test_long_gap_splits_events():
    seq = make_seq(10, right_up=lambda t: 1.0 if (1 <= t < 3 or 4 <= t < 6) else 0.0)
    assert len(detect_hand_raised(seq, P, "s")) == 2


def test_missing_pose_frames_never_create_events():
    seq = make_seq(6, absent=lambda t: True)
    assert detect_hand_raised(seq, P, "s") == []
    assert detect_standing(seq, P, "s") == []


def test_noise_robustness():
    seq = make_seq(10, right_up=window(3, 6), noise=0.006, seed=3)
    events = detect_hand_raised(seq, P, "s")
    assert len(events) == 1
    assert events[0].start_s == pytest.approx(3.0, abs=0.41)


def test_threshold_is_configurable():
    seq = make_seq(6, left_up=window(2.0, 2.4))
    assert len(detect_hand_raised(seq, RuleParams(hand_raise_min_s=0.4), "s")) == 1


# ------------------------------------------------------------ standing
def stand_up_at(t0: float, ramp: float = 1.0):
    return lambda t: float(np.clip((t - t0) / ramp, 0, 1))


def test_posture_states_seated_and_standing():
    seated = posture_features(make_seq(2, stand=lambda t: 0.0), P)
    standing = posture_features(make_seq(2, stand=lambda t: 1.0), P)
    assert np.all(seated.state == SEATED)
    assert np.all(standing.state == STANDING)


def test_standing_transition_basic():
    seq = make_seq(10, stand=stand_up_at(4.0, 1.2))
    events = of_type(detect_standing(seq, P, "s"), "standing_transition")
    assert len(events) == 1
    e = events[0]
    assert 3.5 <= e.start_s <= 5.2 and e.end_s <= 5.6 and e.start_s < e.end_s
    assert e.evidence["seated_before_s"] >= P.dwell_s and e.evidence["standing_after_s"] >= P.dwell_s


def test_sitting_down_is_not_a_standing_transition():
    seq = make_seq(10, stand=lambda t: float(np.clip(1 - (t - 4.0) / 1.2, 0, 1)))
    assert of_type(detect_standing(seq, P, "s"), "standing_transition") == []


def test_upper_body_only_never_infers_sitting():
    seq = make_seq(10, stand=stand_up_at(4.0), hidden=lambda t: LEG_IDS)
    f = posture_features(seq, P)
    assert np.all(f.state == UNKNOWN)
    assert detect_standing(seq, P, "s") == []


def test_transition_while_legs_hidden_is_not_guessed():
    # seated & visible, legs hidden during the stand-up, then standing & visible
    seq = make_seq(14, stand=stand_up_at(5.0), hidden=lambda t: LEG_IDS if 3.0 <= t < 9.0 else set())
    assert of_type(detect_standing(seq, P, "s"), "standing_transition") == []


def test_brief_standing_spike_is_not_a_transition():
    seq = make_seq(10, stand=lambda t: 1.0 if 4.0 <= t < 4.4 else 0.0)
    assert of_type(detect_standing(seq, P, "s"), "standing_transition") == []


def test_insufficient_dwell_before_transition():
    seq = make_seq(8, stand=stand_up_at(0.4, 0.4))  # seated for only ~0.4 s
    assert of_type(detect_standing(seq, P, "s"), "standing_transition") == []


def test_sustained_standing_optional():
    seq = make_seq(10, stand=stand_up_at(3.0))
    sustained = of_type(detect_standing(seq, P, "s"), "sustained_standing")
    assert len(sustained) == 1 and sustained[0].end_s == pytest.approx(10.0, abs=0.01)
    off = RuleParams(enable_sustained_standing=False)
    assert of_type(detect_standing(seq, off, "s"), "sustained_standing") == []


def test_events_stay_within_duration():
    seq = make_seq(5, right_up=window(4, 5), stand=lambda t: 1.0)
    for e in detect_hand_raised(seq, P, "s") + detect_standing(seq, P, "s"):
        assert 0 <= e.start_s < e.end_s <= seq.duration_s


# ------------------------------------------------------------ regressions found with real inference
def test_horizontal_arms_are_not_hand_raised():
    """Arms held out sideways, wrists marginally above the shoulder line (real-image regression)."""
    seq = make_seq(4)
    for wr, sh in ((15, 11), (16, 12)):
        seq.landmarks[:, wr, 1] = seq.landmarks[:, sh, 1] - 0.02  # ~0.2 head units above
    assert detect_hand_raised(seq, P, "s") == []


def test_hand_raise_threshold_is_scale_invariant():
    big = make_seq(4, right_up=window(1, 3))
    small = make_seq(4, right_up=window(1, 3))
    small.landmarks[:, :, :2] = 0.5 + (small.landmarks[:, :, :2] - 0.5) * 0.3  # person 30% the size
    assert len(detect_hand_raised(big, P, "s")) == len(detect_hand_raised(small, P, "s")) == 1


def test_lunge_is_neither_seated_nor_standing():
    seq = make_seq(3, stand=lambda t: 1.0)
    seq.landmarks[:, 25, 1] = seq.landmarks[:, 23, 1] + 0.03  # left thigh horizontal, right leg straight
    assert set(posture_features(seq, P).state.tolist()) == {0}  # AMBIGUOUS


# ------------------------------------------------------------ rules-v0.2 cues learned from real video
def _angled_thighs(seq, dx=0.15, dy=0.10):
    """Three-quarter/elevated-camera seated pose: thighs angled forward-down (drop ~0.42 torso)."""
    for hip, knee, sign in ((23, 25, 1), (24, 26, -1)):
        seq.landmarks[:, knee, 0] = seq.landmarks[:, hip, 0] + sign * dx
        seq.landmarks[:, knee, 1] = seq.landmarks[:, hip, 1] + dy
    return seq


def test_angled_seated_pose_is_seated_via_hip_angle():
    """Real Charades seated frames have drops 0.3-0.55; v0.1's drop-only rule called them ambiguous."""
    seq = _angled_thighs(make_seq(2, stand=lambda t: 1.0))
    f = posture_features(seq, P)
    assert np.all(f.state == SEATED)
    assert np.nanmedian(f.hip_angle_deg) < P.seated_max_hip_angle_deg
    v01 = RuleParams(seated_max_hip_angle_deg=0, seated_max_knee_angle_deg=0, seated_drop_ceiling=99)
    assert not np.any(posture_features(seq, v01).state == SEATED)


def test_out_of_frame_landmarks_are_unobservable():
    """MediaPipe extrapolates off-screen joints with high visibility (2.45% of 'visible' real landmarks)."""
    seq = make_seq(3, right_up=window(0, 3))
    seq.landmarks[:, 16, 1] = -0.2  # right wrist above the top edge of the image
    f = hand_raise_features(seq, P)
    assert np.all(f.right == UNKNOWN)
    assert detect_hand_raised(seq, P, "s") == []


def test_posture_gap_bridging_is_independent_of_hand_gap():
    seq = make_seq(10, stand=stand_up_at(4.0, 1.2), right_up=lambda t: 1.0 if (2 <= t < 2.4 or 3.0 <= t < 3.4) else 0.0)
    wide_posture = RuleParams(posture_merge_gap_s=1.5)
    # widening posture bridging must not merge two brief hand flicks into one >=0.6 s raise
    assert detect_hand_raised(seq, wide_posture, "s") == []


# ------------------------------------------------------------ multi-person (AUDIT D2)
def test_multi_person_frames_raise_a_warning():
    from behaviorlens.pipeline import analyze_pose

    single = analyze_pose(make_seq(4), P, "s")
    multi = analyze_pose(make_seq(4, people=2), P, "s")
    assert not any("More than one person" in w for w in single.warnings)
    assert any("More than one person" in w and "not reliable" in w for w in multi.warnings)
    assert multi.stats["multi_person_frame_ratio"] == 1.0


def test_primary_person_is_the_largest_pose():
    from behaviorlens.pose import select_primary

    big = make_seq(1).landmarks[0]
    small = big.copy()
    small[:, :2] = 0.8 + (small[:, :2] - 0.5) * 0.2  # a distant second person in the corner
    assert select_primary([small, big]) is big
    assert select_primary([big, small]) is big
