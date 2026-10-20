# Staged-recording kit

> **Adults only.** This kit is for recording **consenting adult volunteers** (18 or older) acting out scripted movements. Never record children or real patients, and never record a real therapy session. The public datasets used so far (Charades home videos, UCF101 sports clips) don't look like the intended setting. This kit lets you build footage that does: a room set up like an ABA table-work area, filmed by a fixed camera.

The kit gives you everything needed to turn a few recording sessions into a tune/holdout validation dataset for BehaviorLens:

| File | What it is for |
|---|---|
| [protocol.md](protocol.md) | Room and camera setup, the numbered script of takes, and target counts |
| [consent_template.md](consent_template.md) | Plain-language consent and release **template**. Your IRB or legal counsel must review it before use. |
| [annotation_guide.md](annotation_guide.md) | Operational definitions, how to mark start and end, double annotation, agreement and adjudication |
| [`scripts/ingest_staged_recordings.py`](../../scripts/ingest_staged_recordings.py) | Validates videos and annotations, strips audio and metadata, assigns opaque IDs, writes `data/validation/staged/` |

Everything stays on your machine. Nothing in this kit uploads video, and `data/` is gitignored. Never commit recordings, frames, annotation files that contain names, or signed consent forms.

## Workflow

```
consent ─► record ─► annotate (2 people, blind) ─► ingest ─► validate (tune, then holdout once) ─► delete
```

### 0. Before the first session (one-off, about 1–2 h plus review time)

1. Have the consent template reviewed and adapted by your IRB, ethics committee or legal counsel ([consent_template.md](consent_template.md)). Don't record until that's done.
2. Pick a participant-code scheme (`P01`, `P02`…) and keep the key that links codes to names **on paper or in a separate encrypted file**, never next to the videos.
3. Set up an encrypted folder **outside the repository** for raw recordings (for example an encrypted external drive, or a FileVault, BitLocker or LUKS volume). The ingest script refuses source folders inside the repo, except under the gitignored `data/`.
4. Print the take sheet from [protocol.md](protocol.md#take-sheet).

### 1. Record (about 60–75 min per participant)

| Step | Time |
|---|---|
| Consent conversation and signature | 10 min |
| Room and camera setup, test shot ([protocol.md](protocol.md#camera-setup)) | 15 min (5 min when the room is reused) |
| Script at camera angle 1 (frontal), 12 takes | 25–30 min |
| Move the camera to angle 2 (oblique), repeat takes T01–T05 and T12 | 15–20 min |
| Copy clips to the encrypted folder and fill in `manifest.csv` | 5 min |

Name each clip after the take (`P01_S1_A0_T04.mp4`). The names never leave your folder: the ingest step replaces them with opaque IDs.

### 2. Annotate (about 3–4× the footage length, per annotator)

Two people annotate every clip **independently**, using ELAN or a video player plus a spreadsheet, **not** BehaviorLens, and without seeing each other's work or any algorithm output. A participant's ~15 minutes of footage takes about 45–60 minutes per annotator. Follow [annotation_guide.md](annotation_guide.md). Each annotator saves one file per clip:

```
raw/                                   # your encrypted folder, outside the repo
  manifest.csv                         # video,participant,camera,subjects,session,camera_angle_deg,distance_m,notes
  P01_S1_A0_T04.mp4
  P01_S1_A0_T04.A.csv                  # annotator A: event_type,start_s,end_s,annotator,notes
  P01_S1_A0_T04.B.csv                  # annotator B
  P01_S1_A0_T06.mp4
  P01_S1_A0_T06.A.eaf                  # ELAN works too (one tier per annotator)
  P01_S1_A0_T06.B.eaf
  P01_S1_A0_T06.adjudicator.csv        # later, only when A and B disagree
```

### 3. Ingest (about 1 min per clip)

```bash
.venv/bin/python scripts/ingest_staged_recordings.py /path/to/raw --check   # validate only, writes nothing
.venv/bin/python scripts/ingest_staged_recordings.py /path/to/raw           # add or update the dataset
```

The script:

- validates every video with OpenCV (it must decode, at 1–240 fps, 1 s to 5 min, at most 4K) and every annotation file (known labels, numeric times inside the video, no overlapping events of the same label by the same annotator, at most two annotators plus an `adjudicator`). Every problem is listed with the file and row so you can fix it, and nothing is written until all files pass;
- converts each video to H.264 MP4 at a constant frame rate, at most 1080p, **with no audio track and with all container metadata stripped**: phones embed GPS location, device model and creation time;
- names each video `stg_<16 hex>`, an HMAC of the file's content hash under a secret salt kept in `data/validation/staged/.id_salt`. Re-ingesting the same file keeps its ID, so fixing an annotation never reshuffles the split. IDs can't be traced back to the file name, and without the salt they can't be linked to the original file;
- merges two annotators into one reference set ([annotation_guide.md](annotation_guide.md#consensus-and-adjudication)) and prints their agreement;
- writes `data/validation/staged/annotations.json` in the shared dataset format, plus local-only `manifest.json` (participant code, session, angle) and `ingest_report.json` (agreement and the disagreements to adjudicate).

The console shows the `source file → video_id` mapping **once**. It isn't stored anywhere. If you need it later, copy it into your private session log. The JSON log lines contain IDs and counts only.

`annotations.json` contains one entry per video:

```json
{"video_id": "stg_b74269496096e6fc", "video": "videos/stg_b74269496096e6fc.mp4", "fps": 30.0, "duration_s": 74.2,
 "camera": "fixed", "subjects": 1,
 "events": [{"type": "standing_transition", "start_s": 12.4, "end_s": 13.9, "source_label": "standing_transition"},
            {"type": "other", "start_s": 20.1, "end_s": 20.4, "source_label": "hand_flick"}]}
```

`type` is one of `standing_transition`, `sitting_down`, `hand_raised` and `other`. Distractors and conditions (`hand_flick`, `occlusion`, `camera_pan`…) become `other`, and the exact label is kept in `source_label`. `camera` is `fixed` when the camera never moved during the clip, `handheld` when it moved at any point (including the deliberate pan take, even on a tripod), and `unknown` when not given. Set it in `manifest.csv` or with `--camera`. `subjects` is the number of people visible in the clip, from the manifest (`null` when not given).

### 4. Validate (5–10 min)

The evaluator for this format is being added on a parallel branch (`scripts/datasets.py`, plus a `--dataset` option on `scripts/validate_real_video.py`). Check `python scripts/validate_real_video.py --help` on your checkout for the exact option. Follow the [`validate-real-video` skill](../../.claude/skills/validate-real-video/SKILL.md) and the same discipline as [VALIDATION.md](../VALIDATION.md):

1. Choose any threshold or rule change on the **tune** half only.
2. Score the **holdout** half **once**, against the previous rules version.
3. Report raw numbers with Wilson intervals, then look at the frames of every disagreement before calling anything an algorithm error (LEARNINGS M18).

**Splits and participants.** The existing evaluator's tune/holdout split (`validate_real_video.split_of`) hashes the clip ID, so one participant's clips usually land in both halves. With few participants, holdout numbers are then optimistic, because the rules were tuned on the same bodies, clothes and room. `manifest.json` keeps the participant code of each video so that a participant-level split can be applied once the evaluator supports it ([clinical_validation_plan.md](../regulatory/clinical_validation_plan.md) §12 asks for splitting by participant). Until then, report how many participants appear in both halves.

### 5. Delete (1 min)

```bash
.venv/bin/python scripts/ingest_staged_recordings.py --delete-participant P03   # a volunteer withdraws
.venv/bin/python scripts/ingest_staged_recordings.py --delete-video stg_0123456789abcdef
.venv/bin/python scripts/ingest_staged_recordings.py --delete                     # the whole staged dataset
```

The delete commands ask you to type `delete` (or take `--yes`). They remove only files the script created: converted videos, `annotations.json`, `manifest.json`, `ingest_report.json`, the ID salt, and evaluator landmark caches in `data/validation/cache/` whose names contain a deleted video ID. Anything else in the folder is listed and left alone. They **can't** delete your raw recordings, consent forms, BehaviorLens sessions you created by uploading clips into the app (use **Delete session** there), or OS backups and snapshots. Delete those yourself, by the retention date in the consent form. Use a secure-erase tool where the storage supports it, and remember that SSDs and snapshots may keep copies (see [cybersecurity.md](../regulatory/cybersecurity.md)).

## Synthetic end-to-end example (no real footage)

> 🧪 **SYNTHETIC.** These clips are stick figures rendered from the demo timeline in `sample_data/`, not real people. MediaPipe finds no pose in a stick figure, so any evaluator run on them reports zero detections. The example tests the plumbing only (files, validation, agreement, IDs, deletion), not detection quality.

```bash
SCR=$(mktemp -d)                                   # scratch folder outside the repo
export BEHAVIORLENS_DATA_DIR="$SCR/data"           # keep the example out of your real data/
.venv/bin/python scripts/ingest_staged_recordings.py --synthetic-example "$SCR/raw"
ls "$SCR/raw"     # synthetic_take01.mp4, synthetic_take01.A.csv, synthetic_take01.B.csv, ..., manifest.csv
.venv/bin/python scripts/ingest_staged_recordings.py "$SCR/raw"
```

Expected output (the video IDs differ on every run because the salt is new):

```
source file           video_id              events  disagreements  reference
synthetic_take01.mp4  stg_b74269496096e6fc       4              1  consensus
synthetic_take02.mp4  stg_dc91daba89498b57       4              1  consensus

Inter-rater agreement this run (pooled; event agreement = matched / all events, IoU >= 0.5; kappa on 1 s bins):
  hand_flick           matched   1  only A   0  only B   0  agreement 1.0  kappa 1.0
  hand_raised          matched   2  only A   1  only B   0  agreement 0.667  kappa 0.541
  ...
  standing_transition  matched   2  only A   1  only B   0  agreement 0.667  kappa 0.773

2 disagreement(s) left out of the reference set; see ingest_report.json and annotation_guide.md 'Adjudication'.
```

Annotator B was generated 0.2–0.3 s late and missing one event per clip, so each clip has one disagreement to adjudicate. To finish the example, write an adjudicated file and re-ingest with `--strict`, then delete:

```bash
cp "$SCR/raw/synthetic_take01.A.csv" "$SCR/raw/synthetic_take01.adjudicator.csv"
cp "$SCR/raw/synthetic_take02.A.csv" "$SCR/raw/synthetic_take02.adjudicator.csv"
sed -i 's/,A,/,adjudicator,/' "$SCR/raw"/*.adjudicator.csv
.venv/bin/python scripts/ingest_staged_recordings.py "$SCR/raw" --strict      # 0 disagreements now
.venv/bin/python scripts/ingest_staged_recordings.py --delete --yes
rm -rf "${SCR:?}"
```

`tests/test_recording_kit.py::test_synthetic_example_round_trip` runs the same round trip in CI.

## What this kit does not do

- It doesn't make the data clinical. Phase A recordings are **staged engineering data** ([clinical_validation_plan.md](../regulatory/clinical_validation_plan.md) §4). Results from them aren't clinical validation and support no performance claim.
- It doesn't replace ethics review. The consent template and protocol are drafts for your reviewer.
- It doesn't annotate for you or score detections. Annotation is human work by design, and scoring is the evaluator's job.
