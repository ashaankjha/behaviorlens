# Recording protocol (staged adult volunteers)

> **Adults only (18+), with signed consent ([consent_template.md](consent_template.md)), after ethics or legal review.** No children, no patients, no real sessions. Participants act out movements. Don't ask about or record health information.

This protocol produces footage that resembles a table-work area in an ABA session room, filmed by a fixed camera. It is Phase A of [clinical_validation_plan.md](../regulatory/clinical_validation_plan.md). The script deliberately mixes **target events** (`standing_transition`, `hand_raised`, plus `sitting_down` as a hard negative) with **distractors** that the rules must not flag, and **challenge conditions** (occlusion, a second adult, a camera pan) that test the known failure modes in [VALIDATION.md](../VALIDATION.md).

## Room setup

Arrange a table-work corner:

- **Table:** an adult table, or a child-size one (about 50–60 cm high). Participants are adults either way. A low table changes how far people lean and rise, so note which you used.
- **Chairs:** one for the participant, plus one beside or across the table for an "instructor" (empty, except in take T11). Use a stable chair without wheels. Vary chair type across participants if you can (chair with arms, stool, child-size chair), and note it in `manifest.csv` `notes`.
- **Placement (important):** by default, seat the participant **at the end or side of the table nearest the camera**, so the table does **not** hide their legs. The rules need visible hips, knees and ankles to see a stand-up. A table between the camera and the person (realistic for ABA rooms) is tested deliberately in take T10.
- **Props:** a few table-work items (picture cards, blocks, a cup). Use a shuffled cue-card deck for T13.
- **Background:** a plain wall if possible. Remove posters with people or faces, mirrors, and coat stands with hanging clothes: MediaPipe has hallucinated poses on furniture and clutter (LEARNINGS M31).
- **Floor:** clear the path for the second adult in T11.
- **No identifying material in view:** no name badges, mail or screens.

## Camera setup

Use one phone or camera on a **tripod**, in landscape. Set it up once per angle and don't touch it during takes (except take T12, the pan).

| Setting | Value | Why |
|---|---|---|
| Resolution, frame rate | 1080p, 30 fps (720p minimum; never above 4K) | Ingest limit is 4K, 1–240 fps; it downsizes to ≤1080p |
| Stabilisation, auto-framing | **Off** (no "Center Stage", "auto-framing", "action mode", digital zoom) | Digital reframing moves the image like a pan |
| Focus and exposure | Locked (tap and hold on the participant) | Exposure pumping when someone stands changes the image |
| HDR, beauty filters, portrait blur | Off | Blur and filters alter body edges |
| Audio | Off if the app allows. Ingest strips audio either way. | Audio is identifying and not needed |
| Location tagging | Off in the camera app. Ingest strips metadata either way. | Phone videos embed GPS position |

**Framing.** The **whole body**, including feet, must stay in frame **when standing with both arms raised overhead**. Leave about 10% margin above the raised hands and below the feet. Check this in a test shot before every angle: ask the participant to stand and raise both arms.

**Distance and height.** About 2.5–4 m from the participant. Tripod height 1.0–1.2 m, about the seated participant's chest height (default), or 1.5 m (adult standing eye level), as an optional variation. Record both distance and height in the manifest (`distance_m`, `notes`).

**Angles.** Vary these deliberately, one angle per block of takes:

| Angle | `camera_angle_deg` | When |
|---|---|---|
| Frontal | 0 (camera faces the participant's chest) | Angle 1: every take |
| Oblique | 30–45 (move the tripod sideways along an arc, same distance) | Angle 2: takes T01–T05 and T13 |

A side view (90°) is a known weak spot of the posture rule (LEARNINGS §6). Record it only as an extra, clearly labelled block if you want to measure that.

**Lighting.** Even room light. No window or lamp behind the participant (backlight), no hard shadows across the legs. If the lighting changes between sessions, write it in `notes` (for example "daylight, overcast" or "ceiling LEDs").

## Script

Before each take, the operator reads the take number aloud and shows it on a card **to the camera** for 2 seconds, then waits about 3 seconds before cueing the first action. Use a stopwatch or call-outs ("…and up", "…and down") to get the durations. Clips should be 30–120 s long (the hard limit is 5 min). Keep about 3 s of stillness between actions so the events are separable. Natural variation is welcome; the annotators record what actually happened, not the script.

Legend: **ST** standing_transition, **SD** sitting_down, **HR** hand_raised (counted positives). Distractors and conditions are annotated with the labels in [annotation_guide.md](annotation_guide.md#labels).

| Take | Name | Actions (in order) | ST | SD | HR | Distractor / condition labels |
|---|---|---|---|---|---|---|
| T01 | Baseline | Sit 10 s → stand up at a normal pace, hands pushing on the table → stand 8 s → sit down → sit 5 s | 1 | 1 | 0 | none |
| T02 | Stand-up speeds | Three cycles, each with ≥5 s seated and ≥5 s standing: **slow** stand (about 3 s), **normal** (about 1.5 s), **fast** (under 1 s); sit down after each | 3 | 3 | 0 | none |
| T03 | Stand-ups without hands | Arms crossed: stand, hold 5 s, sit. Hands on thighs: stand, hold 5 s, sit | 2 | 2 | 0 | none |
| T04 | Seated hand raises | ≥3 s hands down between raises. Right 2 s; left 2 s; both 2 s; right 1 s; left 3 s | 0 | 0 | 5 | none |
| T05 | Standing hand raises | Seated 5 s → stand → right 2 s; left 1 s; both 3 s → sit | 1 | 1 | 3 | none |
| T06 | Hand flicks (distractor) | Seated. Four quick flicks up past the shoulder and straight down, **under half a second** each (R, L, R, L); then twice touch the top of the head briefly | 0 | 0 | 0 | `hand_flick` |
| T07 | Reach and point (distractor) | Seated. Point at the camera with the right arm at shoulder height 2 s, then the left; reach across the table for a card twice; both arms straight out sideways at shoulder height 2 s | 0 | 0 | 0 | `reach_point` |
| T08 | Stretching | Seated. Overhead stretch with both arms up 3 s (**this is a hand_raised**: the wrists go above the shoulders); arms out sideways at shoulder height 3 s; torso twist left and right | 0 | 0 | 1 | `stretch` (every stretch, including the overhead one) |
| T09 | Near-miss standing (distractor) | Seated. Rise halfway and sit back (twice); lean forward to pick something off the floor; shuffle the chair closer to the table | 0 | 0 | 0 | `partial_stand` |
| T10 | Occlusion by the table | Move the table (or a box) between the camera and the participant so the legs are hidden from about the hips down. Sit 5 s → stand → hold 5 s → sit; repeat once; raise right hand 2 s; raise both hands 2 s | 2 | 2 | 2 | `occlusion` (whole time the legs are hidden) |
| T11 | Second adult | Needs a second consenting adult (B). B walks behind the participant; while B crosses **in front**, the participant stands, holds 5 s, sits. B sits in the instructor chair and raises their own hand 2 s (not the participant's event). The participant raises their right hand 2 s while B sits there | 1 | 1 | 1 | `second_person` (whole time B is in view). Manifest `subjects = 2` |
| T12 | Deliberate pan (**once per session**, angle 1 only) | Seated 5 s → the operator pans the camera about 30° away and back over about 2 s → seated 5 s → stand → hold 5 s → sit | 1 | 1 | 0 | `camera_pan` (pan interval). Manifest `camera = handheld` |
| T13 | Table-work sequence | 2–3 min. The participant sorts picture cards and does what each shuffled cue card says, at a natural pace: 2× "stand up, then sit down", 3× "raise a hand (any, 1–3 s)", 2× "quick flick", 2× "reach across" | 2 | 2 | 3 | `hand_flick`, `reach_point` |

All takes are with the frontal camera (angle 1) unless marked. Everything except T12 uses `camera = fixed`.

### Target counts

Positives per participant, if every take goes as scripted:

| Block | ST | HR | SD (hard negatives) | Footage |
|---|---|---|---|---|
| Angle 1, T01–T13 | 13 | 15 | 13 | about 13 min |
| Angle 2, T01–T05 and T13 | 9 | 11 | 9 | about 7 min |
| **Per participant (both angles)** | **22** | **26** | **22** | **about 20 min** |

The clinical validation plan (§10) asks for about **60–140 reference-positive events per type**, and assumes about 5 events per participant. Many events from the same person are correlated, so they count for less. With the design effect DE = 1 + (m − 1) × ICC and an assumed ICC of 0.10:

| Plan | Participants | ST | HR | Events per participant (m) | DE | Effective ST |
|---|---|---|---|---|---|---|
| Full script, both angles | 3 | 66 | 78 | 22 | 3.1 | ~21 |
| Full script, both angles | 6 | 132 | 156 | 22 | 3.1 | ~43 |
| Angle 1 only | 6 | 78 | 90 | 13 | 2.2 | ~35 |
| Angle 1 only | 10 | 130 | 150 | 13 | 2.2 | ~59 |

**So more volunteers beat more takes per volunteer.** Three participants on the full script reach the raw 60-event floor within a few sessions, enough for engineering iteration, but the effective sample stays small. For anything you report, aim for at least 8–10 distinct participants. The tune/holdout split halves each number again. Estimate the real ICC from the first sessions and re-check (plan §10).

Expect some loss: about 10–20% of takes need a redo or come out unusable (camera bumped, participant left the frame). Log those as deviations rather than deleting them silently.

### Variety to aim for across participants

Clothing (loose versus fitted trousers, dark versus light, skirts), body size, handedness, chair type, table height, and lighting. Write these in `notes` as plain descriptions such as "dark loose trousers, stool, daylight". **Don't** record age, health, diagnosis or ethnicity.

## Take sheet

Print one sheet per participant and session. Times come from the camera clock or a stopwatch shown at the start. They help annotators, but the annotation comes from the video, not the sheet.

| Take | Angle | Clip file name | Done | Redo? | Deviations (camera bumped, left the frame, wrong order…) |
|---|---|---|---|---|---|
| T01 | 0 | P__\_S_\_A0_T01 | ☐ | ☐ | |
| T02 | 0 | | ☐ | ☐ | |
| T03 | 0 | | ☐ | ☐ | |
| T04 | 0 | | ☐ | ☐ | |
| T05 | 0 | | ☐ | ☐ | |
| T06 | 0 | | ☐ | ☐ | |
| T07 | 0 | | ☐ | ☐ | |
| T08 | 0 | | ☐ | ☐ | |
| T09 | 0 | | ☐ | ☐ | |
| T10 | 0 | | ☐ | ☐ | |
| T11 | 0 | | ☐ | ☐ | second adult code: |
| T12 | 0 | | ☐ | ☐ | pan direction: |
| T13 | 0 | | ☐ | ☐ | |
| T01–T05, T13 | 30–45 | | ☐ | ☐ | angle used: |

## Manifest

One `manifest.csv` per source folder, one row per clip (columns are optional except `video`):

```csv
video,participant,camera,subjects,session,camera_angle_deg,distance_m,notes
P01_S1_A0_T01.mp4,P01,fixed,1,S1,0,3.0,"tripod 1.1 m, dark loose trousers, stool, ceiling LEDs"
P01_S1_A0_T11.mp4,P01,fixed,2,S1,0,3.0,"second adult P02"
P01_S1_A0_T12.mp4,P01,handheld,1,S1,0,3.0,"pan right ~30 deg"
P01_S1_A35_T01.mp4,P01,fixed,1,S1,35,3.0,
```

Use codes, not names, for `participant` and `session`. The ingest script rejects anything that looks like a name (spaces, more than 32 characters). Notes are capped at 200 characters and stay local in `manifest.json`. They never reach `annotations.json`.

## Participant comfort and safety

Repeated standing and sitting is mild exercise. Offer breaks after every few takes and let people skip any take. Use a stable chair, and keep the floor clear. Stop if anyone feels unwell. A skipped take is a deviation, not a problem.
