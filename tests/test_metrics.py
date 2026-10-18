"""Ground-truth parsing and agreement metrics (AUDIT D2)."""

import pytest

from behaviorlens.metrics import match_events, parse_ground_truth_csv


def test_parse_ground_truth_csv_valid_and_extra_columns():
    rows = parse_ground_truth_csv("event_type,start_s,end_s,note\nhand_raised,1.0,2.5,x\n standing_transition ,4,5,\n")
    assert rows == [{"event_type": "hand_raised", "start_s": 1.0, "end_s": 2.5},
                    {"event_type": "standing_transition", "start_s": 4.0, "end_s": 5.0}]


@pytest.mark.parametrize("text, message", [
    ("type,start_s,end_s\nhand_raised,1,2\n", "missing columns: event_type"),
    ("event_type,start_s\nhand_raised,1\n", "missing columns: end_s"),
    ("event_type,start_s,end_s\nhand_raised,one,2\n", "row 2: start_s/end_s must be numbers"),
    ("event_type,start_s,end_s\nhand_raised,3,2\n", "row 2: need 0 <= start_s < end_s"),
    ("event_type,start_s,end_s\nhand_raised,-1,2\n", "row 2"),
    ("", "missing columns"),
])
def test_parse_ground_truth_csv_rejects_bad_input(text, message):
    with pytest.raises(ValueError, match=message):
        parse_ground_truth_csv(text)


def ev(t, s, e):
    return {"event_type": t, "start_s": s, "end_s": e}


def test_match_requires_both_start_and_end_within_tolerance():
    truth = [ev("hand_raised", 10, 12)]
    assert match_events([ev("hand_raised", 10.9, 12.9)], truth, 1.0)["overall"]["tp"] == 1
    assert match_events([ev("hand_raised", 10.0, 13.5)], truth, 1.0)["overall"]["tp"] == 0  # end too far
    assert match_events([ev("standing_transition", 10, 12)], truth, 1.0)["overall"]["tp"] == 0  # wrong type


def test_matching_is_one_to_one():
    truth = [ev("hand_raised", 10, 12)]
    m = match_events([ev("hand_raised", 10, 12), ev("hand_raised", 10.2, 12.2)], truth, 1.0)
    assert (m["overall"]["tp"], m["overall"]["fp"], m["overall"]["fn"]) == (1, 1, 0)
    assert m["overall"]["precision"] == 0.5 and m["overall"]["recall"] == 1.0


def test_metrics_are_none_without_data():
    m = match_events([], [], 1.0)
    assert m["overall"]["precision"] is None and m["overall"]["recall"] is None and m["overall"]["f1"] is None
