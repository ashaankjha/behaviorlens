"""Person tracking on synthetic multi-person sequences (no video, no inference)."""

import numpy as np
import pytest

from behaviorlens.config import RuleParams, TrackingParams
from behaviorlens.event_rules import detect_all
from behaviorlens.models import X
from behaviorlens.pipeline import analyze_multi_pose
from behaviorlens.pose import select_primary
from behaviorlens.tracking import MultiPoseSequence, assign, landmark_bbox, track_people

from conftest import make_multi, make_seq, person

P = RuleParams()


def track_x(result, tid):
    """Nose x per frame of one track (NaN where absent)."""
    return result.track_landmarks(tid)[:, 0, X]


def lerp(a, b, t0, t1):
    return lambda t: a + (b - a) * min(max((t - t0) / (t1 - t0), 0.0), 1.0)


# ------------------------------------------------------------ crossing paths
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_crossing_paths_keep_their_ids(seed):
    a, b = lerp(0.2, 0.8, 0, 6), lerp(0.8, 0.2, 0, 6)
    multi = make_multi(6, people=lambda t: [person(a(t), 0.6, cy=0.45), person(b(t), 0.6, cy=0.55)],
                       shuffle_seed=seed)
    res = track_people(multi)
    assert res.track_ids == [1, 2]
    for tid in res.track_ids:
        x = track_x(res, tid)
        assert np.all(np.isfinite(x))  # present in every frame
        going_right = x[-1] > x[0]
        assert np.all(np.diff(x) > 0) if going_right else np.all(np.diff(x) < 0)  # never swaps mid-clip


def test_crossing_with_occlusion_reidentifies_after_the_gap():
    a, b = lerp(0.2, 0.8, 0, 6), lerp(0.8, 0.2, 0, 6)

    def people(t):
        ps = [person(a(t), 0.6, cy=0.45)]
        if not 2.6 <= t < 3.6:  # B hidden behind A while they cross (1 s)
            ps.append(person(b(t), 0.6, cy=0.55))
        return ps

    res = track_people(make_multi(6, people=people, shuffle_seed=3))
    assert res.track_ids == [1, 2]
    xb = track_x(res, 2) if track_x(res, 2)[0] > 0.5 else track_x(res, 1)
    seen = xb[np.isfinite(xb)]
    assert np.all(np.diff(seen) < 0)  # B continues leftwards with the same ID after reappearing
    assert np.isnan(xb[14:18]).all()  # and is unobservable (not filled in) while hidden


# ------------------------------------------------------------ leaving and re-entering
def test_short_absence_keeps_the_id_long_absence_gets_a_new_one():
    def people(t):
        ps = [person(0.25, 0.7)]
        if t < 2.0 or 3.0 <= t < 4.0 or t >= 7.0:  # out 2-3 s (1 s), then 4-7 s (3 s)
            ps.append(person(0.75, 0.7))
        return ps

    res = track_people(make_multi(10, people=people), TrackingParams(max_gap_s=2.0))
    right = [tid for tid in res.track_ids if np.nanmean(track_x(res, tid)) > 0.5]
    assert len(right) == 2  # re-identified after 1 s, new ID after 3 s (no guessing)
    first, second = sorted(right)
    assert res.present(first).sum() == 15 and res.present(second).sum() == 15
    stats = res.stats(res.auto_subject(), "auto")
    assert stats["tracks"] == 3 and stats["substantial_tracks"] == 3


def test_reentry_far_away_after_a_gap_is_a_new_track():
    people = lambda t: [person(0.2, 0.6)] if t < 2 else ([] if t < 3 else [person(0.85, 0.6)])
    res = track_people(make_multi(6, people=people))
    assert res.track_ids == [1, 2]


# ------------------------------------------------------------ different sizes / subject choice
def test_two_people_of_different_sizes_and_subject_choice():
    """A large person in view half the time and a small person always in view."""
    def people(t):
        ps = [person(0.78, 0.4)]
        if int(t) % 2 == 0:
            ps.append(person(0.3, 1.0))
        return ps

    multi = make_multi(8, people=people, shuffle_seed=0)
    res = track_people(multi)
    assert len(res.track_ids) == 2
    big = next(s for s in res.summaries() if s.mean_bbox_area > 0.05)
    small = next(s for s in res.summaries() if s.mean_bbox_area < 0.05)
    assert big.coverage == pytest.approx(0.5) and small.coverage == 1.0
    # coverage x size: 0.5 x big area beats 1.0 x (0.4^2 = 0.16 of it)
    assert res.auto_subject() == big.track_id
    # per-frame "largest person" (rules-v0.2) attribution would switch identity every second
    stats = res.stats(big.track_id, "auto")
    assert stats["largest_person_switches"] == 7
    assert stats["subject_is_largest_ratio"] == pytest.approx(0.5)  # largest whenever in view


def test_k5crm_pattern_no_false_stand_up_from_identity_switch():
    """Charades K5CRM mechanism: one person seated or low, a second standing person nearby.
    Per-frame largest-person attribution stitches seated frames of one person to standing
    frames of the other and proposes a standing_transition that nobody performed."""
    def people(t):
        ps = [person(0.75, 0.5, stand=1.0)]  # small standing person, always visible
        if t < 4.0:
            ps.append(person(0.3, 1.0, stand=0.0))  # large seated person, then occluded
        return ps

    multi = make_multi(8, people=people, shuffle_seed=1)
    legacy = detect_all(multi.largest_person_sequence(), P, "s")
    assert any(e.event_type == "standing_transition" for e in legacy)  # the v0.2 false positive

    result, tracked, seq = analyze_multi_pose(multi, P, "s")
    assert not any(e.event_type == "standing_transition" for e in result.events)
    t = result.stats["tracking"]
    assert t["substantial_tracks"] == 2 and t["largest_person_switches"] == 1
    assert any("2 people were tracked" in w and f"#{t['subject_track_id']}" in w for w in result.warnings)
    # frames without the subject are unobservable, never filled in from the other person
    assert not seq.has_pose[multi.timestamps_s >= 4.0].any()


def test_reviewer_can_choose_the_other_subject():
    def people(t):
        return [person(0.3, 1.0, stand=0.0), person(0.75, 0.5, stand=1.0 if t >= 3 else 0.0)]

    multi = make_multi(8, people=people)
    auto, tracked, _ = analyze_multi_pose(multi, P, "s")
    assert not any(e.event_type == "standing_transition" for e in auto.events)
    other = next(tid for tid in tracked.track_ids if tid != auto.stats["tracking"]["subject_track_id"])
    chosen, _, _ = analyze_multi_pose(multi, P, "s", subject_track_id=other)
    (st,) = [e for e in chosen.events if e.event_type == "standing_transition"]
    assert st.start_s == pytest.approx(2.8, abs=0.3)
    assert chosen.stats["tracking"]["subject_selection"] == "reviewer"
    ev = st.evidence["tracking"]
    assert ev == {"subject_track_id": other, "tracked_people": 2, "other_person_present_ratio": 1.0}

    missing, _, _ = analyze_multi_pose(multi, P, "s", subject_track_id=99)
    assert missing.stats["tracking"]["subject_selection"] == "auto"
    assert "does not exist" in missing.warnings[0]


def test_single_person_matches_largest_person_attribution():
    """With one person in view, tracking changes nothing (same landmarks, events, warnings)."""
    seq = make_seq(10, stand=lambda t: 1.0 if t >= 4 else 0.0, right_up=lambda t: 1.0 if 6 <= t < 8 else 0.0,
                   absent=lambda t: 1.0 <= t < 1.4)
    multi = MultiPoseSequence.from_pose_sequence(seq)
    result, tracked, sub = analyze_multi_pose(multi, P, "s")
    assert tracked.track_ids == [1]
    assert np.array_equal(np.isnan(sub.landmarks), np.isnan(seq.landmarks))
    legacy = detect_all(seq, P, "s")
    assert [(e.event_type, e.start_s, e.end_s) for e in result.events] == \
        [(e.event_type, e.start_s, e.end_s) for e in legacy]
    assert not any("tracked" in w for w in result.warnings)


def test_no_person_at_all():
    result, tracked, seq = analyze_multi_pose(make_multi(3, people=lambda t: []), P, "s")
    assert tracked.track_ids == [] and result.events == []
    assert result.stats["tracking"]["subject_track_id"] is None
    assert any("No person" in w for w in result.warnings)


# ------------------------------------------------------------ mechanics
def test_duplicate_detections_of_one_person_are_dropped():
    multi = make_multi(2, people=lambda t: [person(0.5, 0.8), person(0.505, 0.8)])
    res = track_people(multi)
    assert res.track_ids == [1] and res.duplicates_dropped == len(multi)


def test_slot_order_does_not_change_tracks():
    a, b = lerp(0.2, 0.7, 0, 5), lerp(0.8, 0.4, 0, 5)
    people = lambda t: [person(a(t), 0.7), person(b(t), 0.5, cy=0.6)]
    r1 = track_people(make_multi(5, people=people, shuffle_seed=10))
    r2 = track_people(make_multi(5, people=people, shuffle_seed=11))
    for tid in r1.track_ids:
        assert np.allclose(r1.track_landmarks(tid), r2.track_landmarks(tid), equal_nan=True)


def test_assignment_is_globally_optimal_not_greedy():
    # Greedy takes the cheapest pair (d0-t0, 0.1) and leaves d1 unmatched;
    # the optimum matches both detections.
    assert assign([[0.1, 0.2], [0.3, None]]) == [1, 0]
    assert assign([[None], [0.5]]) == [None, 0]
    assert assign([]) == []


def test_bbox_all_landmarks_or_visible_only():
    lm = person(0.5, 1.0)
    lm[25:33, 3] = 0.05  # legs not visible
    assert landmark_bbox(lm)[3] == pytest.approx(np.nanmax(lm[:, 1]))  # tracking box: all positions
    box = landmark_bbox(lm, 0.5)
    assert box[3] < lm[25, 1]  # visible-only box stops above the (hidden) knees
    lm[:, 3] = 0.0
    assert landmark_bbox(lm, 0.5) is not None  # falls back to all finite landmarks
    assert landmark_bbox(np.full((33, 5), np.nan)) is None


def test_multi_pose_sequence_roundtrip_and_largest_person(tmp_path):
    small, big = person(0.8, 0.3), person(0.4, 0.9)
    multi = make_multi(2, people=lambda t: [small, big] if t < 1 else [small])
    path = tmp_path / "m.npz"
    multi.save_npz(path)
    loaded = MultiPoseSequence.load_npz(path)
    assert np.allclose(loaded.poses, multi.poses, equal_nan=True, atol=1e-6)
    assert list(loaded.counts) == [2] * 5 + [1] * 5
    seq = loaded.largest_person_sequence()
    assert np.allclose(seq.landmarks[0], select_primary([small, big]), atol=1e-6)
    assert np.allclose(seq.landmarks[-1], small, atol=1e-6)
    assert list(seq.num_people) == list(loaded.counts)
    with pytest.raises(ValueError):
        MultiPoseSequence(timestamps_s=[0, 1], frame_indices=[0, 1], poses=np.zeros((2, 1, 33, 5)),
                          counts=[2, 0], sample_fps=1, duration_s=2)
