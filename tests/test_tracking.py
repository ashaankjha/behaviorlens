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


@pytest.mark.parametrize("tracking", [None, TrackingParams(merge_fragments=True)])
def test_k5crm_pattern_no_false_stand_up_from_identity_switch(tracking):
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

    result, tracked, seq = analyze_multi_pose(multi, P, "s", tracking)
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


# ------------------------------------------------------------ fragment merging (opt-in, LEARNINGS M44)
MERGE = TrackingParams(merge_fragments=True)


def test_fragment_merging_is_off_by_default():
    assert TrackingParams().merge_fragments is False


def fragmented_stand_up(t):
    """One person: seated, stands at ~2 s, then untracked 4-7 s and 11.4-14 s (longer than
    max_gap_s), each time reappearing a little shifted. The longest fragment has no stand-up."""
    if 4.0 <= t < 7.0 or 11.4 <= t < 14.0:
        return []
    cx = 0.40 if t < 4.0 else (0.43 if t < 11.4 else 0.41)
    return [person(cx, 0.7, stand=0.0 if t < 2.0 else 1.0)]


def test_fragmented_single_person_merges_into_one_track_and_keeps_the_stand_up():
    multi = make_multi(20, people=fragmented_stand_up)
    plain = track_people(multi)
    assert len(plain.track_ids) == 3 and plain.fragments_merged == 0
    merged = track_people(multi, MERGE)
    assert merged.track_ids == [1] and merged.fragments_merged == 2
    assert np.array_equal(merged.present(1), multi.counts > 0)  # every observed frame, gaps stay empty
    assert merged.stats(1, "auto")["fragments_merged"] == 2

    def stand_ups(tracking):
        result, _, _ = analyze_multi_pose(multi, P, "s", tracking)
        return [e for e in result.events if e.event_type == "standing_transition"]

    assert stand_ups(TrackingParams()) == []  # the subject (longest fragment) excludes the stand-up
    assert len(stand_ups(MERGE)) == 1


@pytest.mark.parametrize("second", [
    person(0.80, 0.7),  # same size, far away
    person(0.42, 0.35),  # same place, a quarter of the area
    person(0.80, 0.35),  # both
])
def test_people_taking_turns_at_different_positions_or_sizes_are_not_merged(second):
    first = person(0.40, 0.7)
    people = lambda t: [first] if t < 3 else ([] if t < 5.4 else [second] if t < 8 else [] if t < 10.4 else [first])
    res = track_people(make_multi(13, people=people), MERGE)
    ids = {tid: set(np.flatnonzero(res.present(tid))) for tid in res.track_ids}
    second_frames = set(range(27, 40))
    assert second_frames in ids.values()  # the second person keeps a track of their own
    assert res.fragments_merged == 1  # only the first person's two appearances join
    assert len(res.track_ids) == 2


@pytest.mark.parametrize("seed", range(5))
def test_people_present_at_the_same_time_are_never_merged(seed):
    """Even with gates that accept any geometry, a merged track is never in one frame twice and
    every original track maps whole onto one merged track."""
    rng = np.random.default_rng(seed)
    spans = [(float(a), float(a + d), float(x)) for a, d, x in
             zip(rng.uniform(0, 8, 6), rng.uniform(0.6, 4, 6), rng.uniform(0.2, 0.8, 6))]
    people = lambda t: [person(x, 0.6) for a, b, x in spans if a <= t < b]
    multi = make_multi(12, people=people)
    loose = TrackingParams(merge_fragments=True, merge_max_gap_s=100, merge_max_center_dist=100,
                           merge_max_size_ratio=100)
    plain, merged = track_people(multi), track_people(multi, loose)
    for k in range(len(multi)):
        ids = merged.assignment[k][merged.assignment[k] > 0]
        assert len(ids) == len(set(ids)) == int(np.sum(plain.assignment[k] > 0))
    for tid in plain.track_ids:
        assert len(np.unique(merged.assignment[plain.assignment == tid])) == 1
    for a, b in ((a, b) for a in plain.track_ids for b in plain.track_ids if a < b):
        if np.any(plain.present(a) & plain.present(b)):
            assert merged.assignment[plain.assignment == a][0] != merged.assignment[plain.assignment == b][0]


def test_two_people_side_by_side_with_the_same_size_stay_two_tracks():
    people = lambda t: [person(0.30, 0.6)] + ([person(0.36, 0.6)] if 2 <= t < 4 or 6 <= t < 8 else [])
    res = track_people(make_multi(10, people=people), TrackingParams(
        merge_fragments=True, merge_max_center_dist=100, merge_max_size_ratio=100))
    assert np.all(res.people_per_frame() == make_multi(10, people=people).counts)
    left = res.assignment[0][res.assignment[0] > 0][0]
    assert all(left not in res.assignment[k][res.assignment[k] > 0][1:] for k in range(len(res.multi)))
    assert len(res.track_ids) == 2  # the second person's two appearances join each other only


def test_handover_to_a_different_person_after_a_gap_is_not_stitched_into_a_stand_up():
    """A large seated person leaves; 2.5 s later (beyond max_gap_s, within max_transition_s) a
    smaller person stands where they were. Merging must not join them (size gate), so no
    standing_transition is attributed to anyone (H-03)."""
    people = lambda t: [person(0.40, 0.9, stand=0.0)] if t < 4 else [] if t < 6.5 else [person(0.45, 0.45)]
    multi = make_multi(10, people=people)
    assert any(e.event_type == "standing_transition" for e in detect_all(multi.largest_person_sequence(), P, "s"))
    result, tracked, _ = analyze_multi_pose(multi, P, "s", MERGE)
    assert tracked.fragments_merged == 0
    assert not any(e.event_type == "standing_transition" for e in result.events)
