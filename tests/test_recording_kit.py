"""Staged-recording kit: scripts/ingest_staged_recordings.py (docs/recording_kit/)."""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import write_video

from behaviorlens import config
from behaviorlens.logging_setup import configure_logging
from behaviorlens.video_io import ffmpeg_exe, probe_video

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ingest_staged_recordings",
                                               ROOT / "scripts" / "ingest_staged_recordings.py")
assert _spec and _spec.loader
kit = importlib.util.module_from_spec(_spec)
sys.modules["ingest_staged_recordings"] = kit
_spec.loader.exec_module(kit)

HEADER = "event_type,start_s,end_s,annotator,notes\n"


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("BEHAVIORLENS_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data" / "validation" / "staged"


@pytest.fixture
def src(tmp_path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    return d


def video(directory: Path, name: str, seconds: float = 4.0) -> Path:
    return write_video(directory / name, seconds, fps=10.0, size=(96, 64))


def csv_file(path: Path, rows: list[str]) -> Path:
    path.write_text(HEADER + "".join(r + "\n" for r in rows))
    return path


def run(source: Path, **kw) -> tuple[dict, str]:
    out = io.StringIO()
    summary = kit.ingest(source, out=out, **kw)
    return summary, out.getvalue()


def load(data: Path) -> list[dict]:
    return json.loads((data / "annotations.json").read_text())


def test_ingest_writes_the_shared_format(data, src):
    video(src, "take01.mp4")
    video(src, "take02.mov", seconds=3.0)
    csv_file(src / "take01.csv", ["hand_raised,1.0,2.0,A,right hand", "standing_transition,2.2,3.1,A,"])
    csv_file(src / "take02.A.csv", ["hand flick,0.5,0.8,A,distractor", "Sit-Down,1.0,2.0,A,"])
    (src / "manifest.csv").write_text("video,participant,camera,subjects,session,camera_angle_deg\n"
                                      "take01.mp4,P01,fixed,1,S1,0\ntake02.mov,P02,handheld,2,S1,35\n")
    summary, _ = run(src)
    assert summary == {"videos": 2, "events": 4, "disagreements": 0, "check_only": False, "dataset_videos": 2}
    entries = load(data)
    assert len(entries) == 2 and [e["video_id"] for e in entries] == sorted(e["video_id"] for e in entries)
    for e in entries:
        kit.validate_dataset_entry(e)
        assert list(e) == ["video_id", "video", "fps", "duration_s", "camera", "subjects", "events"]
        meta = probe_video(data / e["video"])
        assert meta.codec in ("avc1", "h264") and e["fps"] == pytest.approx(10.0)
        assert e["duration_s"] == pytest.approx(meta.duration_s)
    by_camera = {e["camera"]: e for e in entries}
    assert by_camera["fixed"]["subjects"] == 1 and by_camera["handheld"]["subjects"] == 2
    assert by_camera["fixed"]["events"] == [
        {"type": "hand_raised", "start_s": 1.0, "end_s": 2.0, "source_label": "hand_raised"},
        {"type": "standing_transition", "start_s": 2.2, "end_s": 3.1, "source_label": "standing_transition"}]
    assert by_camera["handheld"]["events"] == [
        {"type": "other", "start_s": 0.5, "end_s": 0.8, "source_label": "hand_flick"},
        {"type": "sitting_down", "start_s": 1.0, "end_s": 2.0, "source_label": "sitting_down"}]
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest[by_camera["handheld"]["video_id"]] == {"participant": "P02", "session": "S1",
                                                          "camera_angle_deg": 35.0}


def test_original_filenames_and_container_metadata_never_reach_the_dataset(data, src, tmp_path):
    exe = ffmpeg_exe()
    if exe is None:
        pytest.skip("ffmpeg unavailable")
    name = "jane_doe_home_visit"
    subprocess.run([exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=10:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
                    "-metadata", f"title={name}", "-metadata", "location=+40.7128-074.0060/",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src / f"{name}.mp4")], check=True)
    csv_file(src / f"{name}.csv", ["hand_raised,1.0,2.0,A,"])
    stream = io.StringIO()
    configure_logging(stream=stream)
    logging.getLogger("behaviorlens").setLevel(logging.INFO)
    _, console = run(src, default_camera="fixed")
    (entry,) = load(data)
    assert kit.VIDEO_ID_RE.fullmatch(entry["video_id"])
    assert name in console  # the operator sees the mapping once...
    for f in data.rglob("*"):  # ...but no stored file, file name or video byte carries it
        assert "jane" not in f.name
        if f.is_file():
            blob = f.read_bytes()
            # MP4 stores the location as a binary 'loci' atom (or Apple '\xa9xyz'), so the
            # coordinates never appear as text: check for the atoms themselves (LEARNINGS M47).
            assert b"jane" not in blob and b"loci" not in blob and b"\xa9xyz" not in blob, f.name
    logs = stream.getvalue()
    assert "staged_video_ingested" in logs and "jane" not in logs and str(src) not in logs
    probe = subprocess.run([exe, "-hide_banner", "-i", str(data / entry["video"])], capture_output=True, text=True)
    assert "Video: h264" in probe.stderr and "Audio:" not in probe.stderr
    assert "location" not in probe.stderr and "title" not in probe.stderr


def test_ids_are_stable_on_reingest_and_unlinkable_across_datasets(data, src):
    video(src, "a.mp4")
    csv_file(src / "a.csv", ["hand_raised,1,2,A,"])
    run(src)
    first = load(data)[0]["video_id"]
    csv_file(src / "a.csv", ["hand_raised,1,2.5,A,corrected"])
    run(src)  # re-ingest after fixing an annotation: same id, entry replaced
    entries = load(data)
    assert [e["video_id"] for e in entries] == [first] and entries[0]["events"][0]["end_s"] == 2.5
    kit.delete_dataset(assume_yes=True, out=io.StringIO())
    run(src)
    assert load(data)[0]["video_id"] != first  # new salt: ids can't be linked to the old dataset


def test_check_only_writes_nothing(data, src):
    video(src, "a.mp4")
    csv_file(src / "a.csv", ["hand_raised,1,2,A,"])
    summary, console = run(src, check_only=True)
    assert summary["check_only"] and "nothing was written" in console
    assert not data.exists()


@pytest.mark.parametrize("files, message", [
    ({"a.csv": ["hand_rasied,1,2,A,"]}, "unknown event_type 'hand_rasied'. Did you mean 'hand_raised'"),
    ({"a.csv": ["hand_raised,1,9,A,"]}, "end_s 9 is beyond the end of a.mp4"),
    ({"a.csv": ["hand_raised,1,2,A,", "hand_raised,1.5,2.5,A,"]}, "overlaps a.csv row 2"),
    ({"a.csv": ["hand_raised,2,1,A,"]}, "need 0 <= start_s < end_s"),
    ({"a.csv": ["hand_raised,one,2,A,"]}, "start_s must be a number of seconds"),
    ({"a.csv": ["hand_raised,1,2,Jane Doe,"]}, "annotator 'Jane Doe' must be a short code"),
    ({"a.A.csv": ["hand_raised,1,2,A,"], "a.B.csv": ["hand_raised,1,2,B,"], "a.C.csv": ["hand_raised,1,2,C,"]},
     "3 annotators"),
    ({"b.csv": ["hand_raised,1,2,A,"]}, "without a matching video: b.csv"),
    ({}, "No annotations for: a.mp4"),
])
def test_bad_annotations_are_rejected_with_actionable_errors(data, src, files, message):
    video(src, "a.mp4")
    for fname, rows in files.items():
        csv_file(src / fname, rows)
    with pytest.raises(kit.IngestError, match=message.replace("(", r"\(").replace(".", r"\.")):
        run(src)
    assert not (data / "annotations.json").exists()


def test_missing_columns_bad_manifest_and_broken_video_are_rejected(data, src):
    video(src, "a.mp4")
    (src / "a.csv").write_text("type,start,end\nhand_raised,1,2\n")
    with pytest.raises(kit.IngestError, match="missing column"):
        run(src)
    csv_file(src / "a.csv", ["hand_raised,1,2,A,"])
    (src / "manifest.csv").write_text("video,camera\na.mp4,tripod\n")
    with pytest.raises(kit.IngestError, match="camera must be one of fixed, handheld, unknown"):
        run(src)
    (src / "manifest.csv").write_text("video,participant\na.mp4,Jane Doe\n")
    with pytest.raises(kit.IngestError, match="participant must be a code"):
        run(src)
    (src / "manifest.csv").unlink()
    (src / "b.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)
    csv_file(src / "b.csv", ["hand_raised,1,2,A,"])
    with pytest.raises(kit.IngestError, match="b.mp4: .*could not be decoded|b.mp4: No frames"):
        run(src)
    assert not data.exists()


def test_source_inside_the_repo_is_refused(data):
    with pytest.raises(kit.IngestError, match="inside the repository"):
        kit.discover(ROOT / "sample_data")


def test_two_annotators_consensus_and_disagreements(data, src):
    video(src, "a.mp4", seconds=6.0)
    csv_file(src / "a.A.csv", ["hand_raised,1.0,2.0,A,", "standing_transition,3.0,4.0,A,"])
    csv_file(src / "a.B.csv", ["hand_raised,1.2,2.3,B,"])
    summary, console = run(src)
    assert summary["events"] == 1 and summary["disagreements"] == 1
    assert "1 disagreement(s) left out" in console
    (entry,) = load(data)
    assert entry["events"] == [{"type": "hand_raised", "start_s": 1.0, "end_s": 2.3, "source_label": "hand_raised"}]
    report = json.loads((data / "ingest_report.json").read_text())["videos"][entry["video_id"]]
    assert report["reference"] == "consensus"
    assert report["disagreements"] == [{"annotator": "A", "label": "standing_transition", "start_s": 3.0,
                                        "end_s": 4.0, "reason": "no matching event from the other annotator"}]
    hand = report["agreement"]["per_label"]["hand_raised"]
    assert hand["matched"] == 1 and hand["mean_onset_diff_s"] == 0.2
    assert hand["mean_iou"] == pytest.approx(0.8 / 1.3, abs=1e-3)
    with pytest.raises(kit.IngestError, match="--strict"):
        run(src, strict=True)


def test_header_only_file_counts_as_an_annotator_who_saw_nothing(data, src):
    video(src, "a.mp4")
    csv_file(src / "a.A.csv", ["hand_raised,1,2,A,"])
    (src / "a.B.csv").write_text(HEADER)
    summary, _ = run(src)
    assert summary["events"] == 0 and summary["disagreements"] == 1


def test_adjudicator_replaces_the_consensus(data, src):
    video(src, "a.mp4", seconds=6.0)
    csv_file(src / "a.A.csv", ["hand_raised,1.0,2.0,A,", "standing_transition,3.0,4.0,A,"])
    csv_file(src / "a.B.csv", ["hand_raised,1.2,2.3,B,"])
    csv_file(src / "a.adjudicator.csv", ["hand_raised,1.1,2.2,adjudicator,", "standing_transition,3,4,adjudicator,"])
    summary, _ = run(src, strict=True)
    assert summary["disagreements"] == 0
    (entry,) = load(data)
    assert [(e["type"], e["start_s"]) for e in entry["events"]] == [("hand_raised", 1.1), ("standing_transition", 3.0)]
    report = json.loads((data / "ingest_report.json").read_text())["videos"][entry["video_id"]]
    assert report["reference"] == "adjudicated" and report["agreement"]["annotators"] == ["A", "B"]


def test_agreement_statistics():
    assert kit.iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    pairs, ua, ub = kit.match_intervals([(0, 2), (5, 6)], [(0.1, 2.1), (9, 10)], 0.5)
    assert [p[:2] for p in pairs] == [(0, 0)] and ua == [1] and ub == [1]
    assert kit.cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert kit.cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert kit.cohens_kappa([False] * 5, [False] * 5) is None
    x = [True, True, False, False, False]
    y = [True, False, False, False, False]
    assert kit.cohens_kappa(x, y) == pytest.approx((0.8 - (0.4 * 0.2 + 0.6 * 0.8)) / (1 - (0.4 * 0.2 + 0.6 * 0.8)))
    assert kit._bins([(0.5, 1.5)], 3.0, 1.0) == [True, True, False]
    ioa = kit.interval_ioa([(1, 2), (3, 4), (25, 26)], [(1, 2), (12, 13)], 30.0)
    # counts per 10 s interval: A [2, 0, 1], B [1, 1, 0]
    assert ioa == {"interval_ioa": round(1 / 3, 3), "exact_count_ioa": 0.0,
                   "mean_count_per_interval_ioa": round(0.5 / 3, 3)}


def test_pooled_agreement_sums_over_videos():
    a = kit.Annotation("A", "hand_raised", "hand_raised", 1, 2, "x")
    b = kit.Annotation("B", "hand_raised", "hand_raised", 1.1, 2.1, "y")
    pooled = kit.pooled_agreement([([a], [b], 5.0), ([a], [], 5.0)], 0.5)
    assert pooled["hand_raised"] == {"matched": 1, "only_a": 1, "only_b": 0, "event_agreement": 0.5,
                                     "kappa": 0.375}  # po 0.8, pe 0.68 over 10 one-second bins


EAF = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" FORMAT="3.0" VERSION="3.0">
  <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
  <TIME_ORDER>
    <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="1000"/><TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="2000"/>
    <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="1100"/><TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="2100"/>
  </TIME_ORDER>
  <TIER TIER_ID="events-A" ANNOTATOR="A" LINGUISTIC_TYPE_REF="default">
    <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
      <ANNOTATION_VALUE>hand_raised</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
  </TIER>
  <TIER TIER_ID="B" LINGUISTIC_TYPE_REF="default">
    <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
      <ANNOTATION_VALUE>hand raise</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
  </TIER>
  <TIER TIER_ID="notes" ANNOTATOR="C" LINGUISTIC_TYPE_REF="default"/>
</ANNOTATION_DOCUMENT>
"""


def test_elan_eaf_is_parsed_per_tier_annotator(src):
    path = src / "a.eaf"
    path.write_text(EAF.replace('  <TIER TIER_ID="notes" ANNOTATOR="C" LINGUISTIC_TYPE_REF="default"/>\n', ""))
    anns, who = kit.parse_eaf(path)
    assert who == {"A", "B"}
    assert [(a.annotator, a.type, a.start_s, a.end_s) for a in anns] == [
        ("A", "hand_raised", 1.0, 2.0), ("B", "hand_raised", 1.1, 2.1)]
    path.write_text(EAF)
    assert kit.parse_eaf(path)[1] == {"A", "B", "C"}  # an empty tier still counts as an annotator


def test_elan_eaf_ingests_and_unsafe_xml_is_refused(data, src):
    video(src, "a.mp4")
    (src / "a.eaf").write_text(EAF.replace('  <TIER TIER_ID="notes" ANNOTATOR="C" LINGUISTIC_TYPE_REF="default"/>\n',
                                           ""))
    summary, _ = run(src)
    assert summary == {"videos": 1, "events": 1, "disagreements": 0, "check_only": False, "dataset_videos": 1}
    (src / "a.eaf").write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><ANNOTATION_DOCUMENT/>')
    with pytest.raises(kit.IngestError, match="DOCTYPE/ENTITY"):
        run(src)
    (src / "a.eaf").write_text(EAF.replace('TIME_VALUE="1000"', 'TIME_VALUE="1.5s"'))
    with pytest.raises(kit.IngestError, match="not whole milliseconds"):
        run(src)
    (src / "a.eaf").write_text("<ANNOTATION_DOCUMENT><unclosed></ANNOTATION_DOCUMENT>")
    with pytest.raises(kit.IngestError, match="not valid XML"):
        run(src)


def test_delete_dataset_removes_only_known_files(data, src):
    video(src, "a.mp4")
    csv_file(src / "a.csv", ["hand_raised,1,2,A,"])
    run(src)
    vid = load(data)[0]["video_id"]
    cache = data.parent / "cache"
    cache.mkdir()
    (cache / f"{vid}_lite_5fps.npz").write_bytes(b"x")
    (cache / "OTHER_clip.npz").write_bytes(b"x")
    (data / "my_notes.txt").write_text("keep me")
    out = io.StringIO()
    assert not kit.delete_dataset(assume_yes=False, out=out)  # no tty, no --yes: refuses
    assert (data / "annotations.json").exists()
    assert kit.delete_dataset(assume_yes=True, out=out)
    assert sorted(p.name for p in data.rglob("*")) == ["my_notes.txt"]
    assert sorted(p.name for p in cache.iterdir()) == ["OTHER_clip.npz"]
    assert "my_notes.txt" in out.getvalue() and "Not deleted: your raw recordings" in out.getvalue()
    (data / "my_notes.txt").unlink()
    kit.delete_dataset(assume_yes=True, out=out)
    assert not data.exists() or not any(data.iterdir())


def test_delete_video_and_participant_withdrawal(data, src):
    video(src, "a.mp4")
    video(src, "b.mp4", seconds=3.0)
    video(src, "c.mp4", seconds=5.0)
    for n in "abc":
        csv_file(src / f"{n}.csv", ["hand_raised,1,2,A,"])
    (src / "manifest.csv").write_text("video,participant\na.mp4,P01\nb.mp4,P01\nc.mp4,P02\n")
    run(src)
    manifest = json.loads((data / "manifest.json").read_text())
    p02 = next(k for k, v in manifest.items() if v["participant"] == "P02")
    assert kit.delete_participant("P01", assume_yes=True, out=io.StringIO()) == 2
    assert [e["video_id"] for e in load(data)] == [p02]
    assert sorted(p.stem for p in (data / "videos").iterdir()) == [p02]
    assert list(json.loads((data / "manifest.json").read_text())) == [p02]
    assert list(json.loads((data / "ingest_report.json").read_text())["videos"]) == [p02]
    assert kit.delete_videos([p02], assume_yes=True, out=io.StringIO()) == 1
    assert load(data) == []
    with pytest.raises(kit.IngestError, match="not a staged video_id"):
        kit.delete_videos(["../annotations"], assume_yes=True)


def test_dataset_dir_symlink_is_refused(data, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    data.parent.mkdir(parents=True)
    data.symlink_to(elsewhere)
    with pytest.raises(kit.IngestError, match="symlink"):
        kit.delete_dataset(assume_yes=True)


def test_synthetic_example_round_trip(data, tmp_path, capsys):
    example = tmp_path / "synthetic"
    assert kit.main(["--synthetic-example", str(example)]) == 0
    assert kit.main([str(example)]) == 0
    entries = load(data)
    assert len(entries) == 2 and all(e["camera"] == "fixed" and e["duration_s"] == pytest.approx(20.0)
                                     for e in entries)
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "standing_transition" in out
    report = json.loads((data / "ingest_report.json").read_text())
    assert sum(len(v["disagreements"]) for v in report["videos"].values()) == 2
    assert kit.main(["--delete", "--yes"]) == 0 and not (data / "annotations.json").exists()


def test_cli_reports_errors_with_exit_code_2(data, src, capsys):
    video(src, "a.mp4")
    csv_file(src / "a.csv", ["jump,1,2,A,"])
    assert kit.main([str(src)]) == 2
    assert "unknown event_type 'jump'" in capsys.readouterr().err
    assert kit.main([str(src), "--allow-other-labels"]) == 0
    assert load(data)[0]["events"] == [{"type": "other", "start_s": 1.0, "end_s": 2.0, "source_label": "jump"}]
    with pytest.raises(SystemExit):
        kit.main([])


def test_dataset_lives_under_the_gitignored_data_dir(data):
    assert kit.dataset_dir().is_relative_to(config.data_dir())
    assert "data/" in (ROOT / ".gitignore").read_text().splitlines()
