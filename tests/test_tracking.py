"""Person tracking on synthetic multi-person sequences (no video, no inference)."""

import numpy as np
import pytest

from behaviorlens.config import RuleParams, TrackingParams
from behaviorlens.event_rules import detect_all
from behaviorlens.models import X
from behaviorlens.pipeline import analyze_multi_pose
from behaviorlens.pose import select_primary
from behaviorlens.tracking import MultiPoseSequence, assign, identity_jumps, landmark_bbox, track_people

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


# ------------------------------------------------------------ identity-jump guard (opt-in, H-03 / G-12)
MASK = TrackingParams(identity_jump_guard="mask")
SPLIT = TrackingParams(identity_jump_guard="split")


def test_identity_jump_guard_is_off_by_default_and_rejects_unknown_modes():
    assert TrackingParams().identity_jump_guard == "off"
    assert "identity_jump_guard" not in track_people(make_multi(2, people=lambda t: [person()])).stats(1, "auto")
    with pytest.raises(ValueError, match="identity_jump_guard"):
        TrackingParams(identity_jump_guard="on")


def alternating_people(t):
    """A seated person and a standing person side by side; MediaPipe returns only one of them per
    frame (the 6HR54 / HOI88 mechanism, LEARNINGS M49): A alone until 2 s, then B for 1 s and A
    for 0.6 s in turn. The tracker joins them into one track."""
    a, b = person(0.40, 0.6, stand=0.0), person(0.63, 0.6, stand=1.0)
    return [a] if t < 2.0 or (t - 2.0) % 1.6 >= 1.0 else [b]


def hand_handover(t):
    """A raises a hand for 0.4 s, then B (whose hand is up) is detected instead for 1 s: each
    raise alone is shorter than hand_raise_min_s, but stitched together they last 0.8 s."""
    if 2.4 <= t < 3.4:
        return [person(0.63, 0.6, stand=1.0, right_up=1.0 if t < 2.8 else 0.0)]
    return [person(0.40, 0.6, stand=0.0, right_up=1.0 if 2.0 <= t < 2.4 else 0.0)]


@pytest.mark.parametrize("scene", [alternating_people, hand_handover])
def test_people_alternating_at_nearby_positions_yield_no_event_stitched_across_them(scene):
    multi = make_multi(10, people=scene)
    # Without the guard the tracker alternates one track between the two people, and the rules
    # see seated evidence from one person and standing (or hand) evidence from the other.
    off, tracked, _ = analyze_multi_pose(multi, P, "s")
    jumps = identity_jumps(tracked.track_landmarks(tracked.auto_subject()))
    assert np.nanmax(jumps) > 1.0
    stitched = {e.event_type for e in off.events} & {"standing_transition", "hand_raised"}
    assert stitched
    for guard in (MASK, SPLIT):
        result, _, _ = analyze_multi_pose(multi, P, "s", guard)
        assert not {e.event_type for e in result.events} & stitched
        assert any("Identity-jump guard" in w for w in result.warnings)


def test_mask_guard_makes_frames_after_a_jump_unobservable_and_split_cuts_the_track():
    multi = make_multi(6, people=hand_handover)
    masked = track_people(multi, MASK)
    assert masked.track_ids == [1]  # the mask never changes track membership
    jump_frames = np.flatnonzero(masked.identity_jump_frames(1))
    assert list(multi.timestamps_s[jump_frames]) == [2.4, 3.4]  # A -> B and B -> A
    seq = masked.track_sequence(1)
    gone = ~np.isfinite(seq.landmarks[:, 0, X])
    assert list(np.flatnonzero(gone)) == list(range(12, 22))  # 2.4 s up to 3.4 s + 1 s
    stats = masked.stats(1, "auto")["identity_jump_guard"]
    assert stats == {"mode": "mask", "threshold": 1.0, "mask_s": 1.0, "tracks_split": 0,
                     "subject_jumps": 2, "subject_masked_frames": 10}

    split = track_people(multi, SPLIT)
    assert split.identity_splits == 2 and len(split.track_ids) == 3
    assert np.array_equal(split.people_per_frame(), multi.counts)  # frames are moved, never dropped
    b = split.assignment[12, 0]
    assert np.flatnonzero(split.present(b)).tolist() == list(range(12, 17))  # B alone in its own track
    assert split.stats(split.auto_subject(), "auto")["identity_jump_guard"]["tracks_split"] == 2


def test_a_single_switch_outlasting_the_mask_window_is_still_stitched_but_split_is_not():
    """The mask is a window, not a cut: if the second person stays in view longer than
    identity_jump_mask_s after a single switch, seated evidence before the switch and standing
    evidence after the window still form a transition. Splitting the track prevents that."""
    people = lambda t: [person(0.40, 0.6, stand=0.0)] if t < 3.0 else [person(0.63, 0.6, stand=1.0)]
    multi = make_multi(8, people=people)
    stand_ups = lambda tp: [e for e in analyze_multi_pose(multi, P, "s", tp)[0].events
                            if e.event_type == "standing_transition"]
    assert stand_ups(TrackingParams()) and stand_ups(MASK)
    assert not stand_ups(SPLIT)
    assert stand_ups(TrackingParams(identity_jump_guard="mask", identity_jump_mask_s=6.0)) == []


@pytest.mark.parametrize("guard", [MASK, SPLIT, TrackingParams(identity_jump_guard="split",
                                                                identity_jump_threshold=0.8)])
@pytest.mark.parametrize("rise_s", [0.2, 0.4])
def test_one_person_moving_fast_keeps_their_events_below_the_threshold(guard, rise_s):
    """A stand-up within one or two sampled frames and a quick arm raise move the body core by
    well under one body size per frame, so no jump is flagged and the events stay."""
    def people(t):
        stand = min(max((t - 3.0) / rise_s, 0.0), 1.0)
        return [person(0.40 + 0.02 * stand, 0.6, stand=stand, right_up=1.0 if 5.0 <= t < 5.8 else 0.0)]

    multi = make_multi(8, people=people)
    off, tracked, _ = analyze_multi_pose(multi, P, "s")
    assert np.nanmax(identity_jumps(tracked.track_landmarks(1))) < 0.5
    on, guarded, _ = analyze_multi_pose(multi, P, "s", guard)
    types = lambda r: [(e.event_type, e.start_s, e.end_s) for e in r.events]
    assert {"standing_transition", "hand_raised"} <= {e.event_type for e in on.events}
    assert types(on) == types(off)
    assert guarded.track_ids == [1]
    assert not any("Identity-jump guard" in w for w in on.warnings)


def test_identity_jumps_ignore_gaps_and_need_enough_core_landmarks():
    lm = np.stack([person(0.40, 0.6), np.full((33, 5), np.nan), person(0.40, 0.6), person(0.70, 0.6)])
    j = identity_jumps(lm)
    assert np.isnan(j[0]) and np.isnan(j[1])  # first frame, absent frame
    assert j[2] == 0.0  # compared with the last frame in which the track was present
    assert j[3] > 1.0
    lm[3, [0, 11, 12]] = np.nan  # only the hips left: too few core landmarks to judge
    assert np.isnan(identity_jumps(lm)[3])
