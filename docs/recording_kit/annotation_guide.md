# Annotation guide

Annotations are the **reference standard** that BehaviorLens is scored against, so they must be independent of it. Annotate in **ELAN** or in a video player with frame stepping (VLC, mpv), plus a spreadsheet. **Never annotate in BehaviorLens** and never look at its output first: seeing the suggestions anchors you to them.

Describe **only what is visible**: body positions and movements. Don't label why someone moved, what they felt or meant, or whether a movement was "appropriate". This is the same rule BehaviorLens follows (PRD non-negotiables).

## Labels

### Target events (scored)

| Label | Definition | Start | End |
|---|---|---|---|
| `hand_raised` | At least one **wrist is clearly above the same-side shoulder**, roughly at or above chin height (the rules require the wrist above the shoulder line by at least half the vertical distance from shoulders to nose), held for **at least 0.6 s**. | First frame where the wrist reaches that height | Last frame before it drops below that height |
| `standing_transition` | A change from **stable seated** (weight on the seat, for about a second or more) to **stable standing** (upright, weight on the feet, hips and knees extended, held for about a second or more). | Onset of the rise: the last frame where the buttocks are on the seat | First frame of upright stance: hips and knees extended, trunk roughly vertical |
| `sitting_down` | The reverse: stable standing to stable seated. A hard negative: the rules must **not** call it a stand-up. | First frame the hips start to flex and lower from upright stance | First frame the buttocks rest on the seat |

Rules for the target events:

- **Both hands at once is one event.** If the right hand goes up and the left joins while the right is still up, mark one `hand_raised` from the first wrist up to the last wrist down. The rules merge them the same way.
- **Short dips.** If the wrist drops below the line for **under 0.4 s** and goes back up, keep it as one event. A longer drop ends the event.
- **Shorter than 0.6 s** above the line is a `hand_flick`, not a `hand_raised`, even if it was meant as a raise. Judge by frames, not intent. A stretch with the wrists overhead for ≥ 0.6 s **is** a `hand_raised`. Also mark it `stretch` (see below).
- **Partial rises** (rise halfway and sit back) are `partial_stand`, not `standing_transition`. Stepping from one standing position to another isn't a transition.
- **Seated → standing → seated in quick succession:** mark both the `standing_transition` and the `sitting_down`, even if the standing phase was short. Note "brief standing" when it lasted under about 1 s.
- **Only the target participant's movements are events.** In take T11, the second adult's own hand raise or stand-up is **not** marked as `hand_raised` or `standing_transition`; their presence is marked as `second_person`.
- **Can't see it, still mark it.** If a stand-up happens behind the table (take T10), mark it from what you can see (head and trunk rising, hips leaving the seat), and add `legs hidden` in `notes`. Missing evidence is the algorithm's problem, not the annotator's: an annotated event that the rules can't observe is a documented miss.

### Distractors and conditions (kept as type `other`, not scored as positives)

| Label | Mark when |
|---|---|
| `hand_flick` | A wrist goes above the shoulder for **under 0.6 s** |
| `reach_point` | A reach or point at or below shoulder height (arm extended, wrist not above the shoulder line) |
| `stretch` | Any stretch. Overlaps a `hand_raised` when the wrists are overhead for ≥ 0.6 s |
| `partial_stand` | Rising part-way and sitting back, or leaning far forward without standing |
| `lunge` | One leg bent, one straight, while standing (posture is neither seated nor standing) |
| `wave` | Hand moves side to side; mark `hand_raised` as well if the wrist stays above the shoulder for ≥ 0.6 s |
| `occlusion` | The participant's legs or arms are hidden (by the table, an object, a person) |
| `second_person` | Another person is visible, from first to last frame |
| `camera_pan` | The camera moves (take T12), from the first to the last frame of movement |
| `out_of_frame` | The participant is partly or wholly out of frame |
| `other` | Anything else worth flagging. Explain in `notes` |

Labels are case-insensitive. Spaces and hyphens are read as underscores (`Hand raised` = `hand_raised`), and a few aliases are accepted (`stand_up`, `sit_down`, `hand_raise`). Anything else is rejected with a "did you mean" hint, so typos can't silently create new labels. `--allow-other-labels` keeps unlisted labels as type `other`.

Labels may overlap each other (a `hand_raised` during `second_person`), but the same annotator must not mark two overlapping events with the **same** label. Merge them, or fix the times.

## How to mark times

- Step frame by frame around each boundary (ELAN: Shift+arrow; VLC: E; mpv: `.` and `,`). Record times **in seconds from the start of the clip**, to 0.1 s or better. Use the clip's own timeline, not a wall clock.
- Mark the **frames where the definition becomes true and stops being true**. Don't widen the interval "to be safe".
- Mark every occurrence, including clumsy or ambiguous ones. If you're unsure whether something meets a definition, mark your best judgement and write `unsure` in `notes`. Unsure items are good adjudication material.

## File formats

### (a) CSV, one file per clip and annotator (recommended)

File name `<video stem>.<annotator>.csv`, for example `P01_S1_A0_T04.A.csv`. Columns:

```csv
event_type,start_s,end_s,annotator,notes
hand_raised,4.3,6.4,A,right hand
hand_raised,9.8,11.9,A,left hand
hand_raised,15.2,17.3,A,both hands
hand_flick,20.0,20.3,A,
```

- `annotator` is a short code (`A`, `B`, initials), never a full name. If the column is empty or missing, the code is taken from the file name (`….A.csv` → `A`).
- **Takes without events:** save a file with the header row only. It records that this annotator watched the clip and saw no events, which matters for agreement: an empty file from B means B disagrees with every event A marked.
- `notes` is free text for your team. The ingest step **drops** it: notes never reach `annotations.json`. Still, don't write names in them.
- Any spreadsheet works: build these columns and export as **CSV UTF-8**. Commas, semicolons and tabs are all accepted as separators. Extra columns are ignored.

### (b) ELAN `.eaf`

File name `<video stem>.eaf` or `<video stem>.<anything>.eaf`. Create **one tier per annotator**, and set the tier's *Annotator* field in ELAN (Tier → Change Tier Attributes). If it's empty, the participant field or the tier name is used as the annotator code. Each time-aligned annotation is one event, and its value is the label. Reference (symbolic) tiers are ignored. An empty tier counts as "watched, saw nothing". If two annotators work in separate ELAN files, name them `<stem>.A.eaf` and `<stem>.B.eaf`.

## Double annotation and agreement

Two annotators (**A** and **B**) annotate **every** clip independently. Ideally they are trained on this guide with a few practice clips (not from the dataset) and discuss those practice clips before starting. They must not see each other's files until both are finished.

The ingest script computes agreement per label, for each clip and pooled over each run. Results are printed and written to `ingest_report.json`:

| Measure | How it is computed | Target (plan §7) |
|---|---|---|
| **Event agreement** | Events of the same label from A and B are matched one-to-one, greedily by highest temporal IoU (intersection over union), when IoU ≥ 0.5 (`--iou`). Agreement = matched / (matched + only A + only B). | ≥ 0.80 |
| **Cohen's kappa** | The clip is cut into 1 s bins; a bin is "on" for a label when any event of that label touches it. κ = (p_o − p_e) / (1 − p_e) over all bins (pooled over clips for the run summary). It is undefined (`null`) when neither annotator marked that label. | ≥ 0.80 |
| **Interval-by-interval IOA** | 10 s intervals; an event belongs to the interval containing its start. Share of intervals where A and B agree on occurrence versus non-occurrence. Also exact-count IOA, and mean count-per-interval IOA (min/max per interval). | ≥ 0.80 |
| Mean onset and offset difference, mean IoU | Over matched pairs | Report |

Example with κ worked out: over 10 one-second bins, A marks bins 2 and 7, B marks bins 2 and 3. Observed agreement p_o = 8/10 = 0.8. Each annotator marks 2/10 bins, so chance agreement p_e = 0.2 × 0.2 + 0.8 × 0.8 = 0.68, and κ = (0.8 − 0.68) / (1 − 0.68) = 0.375. Kappa punishes disagreement on rare labels hard, which is why it's reported alongside event agreement.

**Check agreement early.** After the first participant, ingest with `--check` and look at the pooled agreement. If any target label is below 0.80, refine the wording here, discuss the disagreeing clips (A and B together, before any further annotation), and re-annotate the pilot clips. Don't use pilot clips to tune the algorithm and judge agreement on them at the same time. Always report the **raw** (pre-adjudication) agreement.

## Consensus and adjudication

How the reference events in `annotations.json` are built for each clip:

| Annotators on the clip | Reference set |
|---|---|
| One | That annotator's events, as marked (`reference: single`). Fine for pilots, but the plan requires two for anything reported. |
| A and B | **Consensus.** Events matched one-to-one (same label, IoU ≥ 0.5) become **one event spanning the union** of the pair (earliest start to latest end). The union is used because annotators differ mainly in where a movement begins and ends, and both of them saw the movement somewhere within it. Mean onset and offset differences in the report show how wide that disagreement is. Events that only one annotator marked are **left out** of the reference set and listed in `ingest_report.json` as `disagreements` for adjudication (`reference: consensus`). |
| A, B and `adjudicator` | The **adjudicator's file replaces the consensus** for that clip (`reference: adjudicated`). A and B agreement is still reported. |

**Adjudication.** A third person, ideally a senior annotator who hasn't annotated the clip and hasn't seen BehaviorLens output on it, reviews each listed disagreement on the video. They then write `<stem>.adjudicator.csv` with annotator code `adjudicator`, containing the **complete final event list** for that clip: every event, not just the disputed ones. A practical shortcut is to copy the consensus events from the report, then add or drop the disputed ones. Re-run the ingest with `--strict`. It fails while any clip still has unadjudicated disagreements, so the dataset you validate on is fully resolved.

Leaving unmatched events out until adjudication (rather than including them) keeps the reference set to what two people independently saw. Until adjudication is done, a detection on a disputed event counts as a false positive, and a missed disputed event isn't counted. Report how many disagreements remained when you scored.

## Checklist per clip

1. Watch the whole clip once at normal speed.
2. Mark target events, then distractors and conditions, frame-stepping each boundary.
3. Check the times are inside the clip and same-label events don't overlap.
4. Save as `<stem>.<annotator>.csv` (header only if nothing happened).
5. Don't open the other annotator's file.
