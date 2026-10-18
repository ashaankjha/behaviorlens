# BehaviorLens — Product Requirements Document
Version: 1.0 | Date: 2026-10-18 | Status: MVP build specification

## 1. Agent mission and operating instructions
Build a functional local-first MVP of BehaviorLens, a computer-vision-assisted review tool for videos of staged behavioral sessions. Deliver working source code, reproducible setup, demo video/data, tests, and a clear README. Do not deliver only wireframes or pseudocode.

Proceed autonomously. Do not ask questions unless truly blocked by an unrecoverable external credential or permission. Prefer reasonable defaults, document assumptions, and finish a smaller working vertical slice rather than incomplete advanced features. Never claim that an untested function works. At completion, report working functionality, test commands/results, known limitations, and exact launch steps.

**Non-negotiable constraints:** local processing; use only staged videos of consenting adults or synthetic video/pose data; no real patient data; no cloud video upload, telemetry, or third-party LLM calls; no diagnoses, inferred intent/emotion, behavioral function classification, or treatment recommendations. Detected labels describe visually observable movement patterns only and must be flagged as algorithm-generated suggestions requiring review.

## 2. Product purpose and users
**Problem:** Human reviewers of video may spend time scrubbing footage and manually marking repeated observable events. A timeline of candidate movement events with timestamps and supporting evidence may reduce review effort, but must not be treated as validated behavioral assessment.

**Primary user:** BCBA or behavior-analytics researcher reviewing staged demonstration footage. **Secondary user:** ML engineer who wants reproducible labeled data for evaluating event detection.

**Core user story:** As a reviewer, I upload a local MP4, inspect system-suggested events, click each timestamp to review footage, correct or reject events, optionally annotate an additional event, and export the corrected event data.

**Non-goals for MVP:** multiple-person tracking, child/patient recordings, diagnosis, autism severity detection, facial emotion recognition, language understanding, real-time production surveillance, HIPAA-compliance certification, live wearable integration, claims submission, automatic treatment decisions, cloud deployment, or model fine-tuning.

## 3. Success criteria
- A new developer can start app with the documented setup on a supported Python version.
- User can load supplied staged demo video and a local MP4 (target <= 100 MB, <= 5 minutes); errors are handled cleanly.
- Uploaded files remain local, in a gitignored application storage directory.
- Pose analysis returns time-indexed landmarks or explicit no-person/low-confidence frames.
- At least two candidate observable event types work end-to-end on a staged video: `standing_transition` and `hand_raised` (see definitions below). `sustained_standing` is optional.
- Candidate event list includes type, start/end time, status, provenance, and linked video review.
- Reviewer can confirm, reject, or edit an event; reviewer can add a manual event.
- Review state persists across app restarts via SQLite.
- CSV and JSON exports include event data and review provenance; re-running detection does not silently overwrite reviewer edits.
- Automated unit tests cover event logic, storage, export, and malformed input, plus one smoke/integration test of a full demo processing run.
- README shows a verified start command, demo workflow, test command, and limitations.

## 4. Scope and functional requirements

### FR1 — Upload and session management (P0)
- Upload `.mp4`, `.mov`, or `.avi` where OpenCV/FFmpeg can decode the content; MVP guarantee MP4/H.264 if installed codecs support it.
- Enforce <=100 MB upload and <=5 minutes duration; show helpful failure messages. Allow a built-in sample session with one click.
- Create a unique session ID, content checksum, source filename, creation timestamp, fps, dimensions, duration, and processing status.
- Never expose absolute filesystem paths in exported artifacts. Sanitize filenames and do not trust client MIME type alone.

### FR2 — Pose extraction (P0)
- Use MediaPipe Tasks Pose Landmarker VIDEO mode with a locally downloaded, documented model asset and OpenCV for frame decoding.
- Sample at a configurable target of 5 fps by default (optionally 10 fps); derive monotonically increasing timestamps from source frame indices and source FPS, rather than processing wall-clock time.
- Normalize keypoints and store per-sampled-frame landmark coordinates, visibility/presence when available, and timestamp.
- One primary person only; if zero people are detected mark frame `no_pose`. If multiple people appear, warn that tracking attribution is not reliable.
- Show progress and handle decode/model failures without crashing the UI. Cache processing results by content checksum + model version + settings.
- Offer a toggled pose-overlay preview, preferably via pre-rendered low-resolution local overlay video. If this threatens P0 completion, offer landmark chart and sampled annotated images instead; do not block core timeline.

### FR3 — Candidate event detection (P0)
Use interpretable rules with configurable thresholds; these are engineering heuristics, not validated clinical labels.

1. `standing_transition`: candidate transition from a stable seated-like configuration to an upright/standing-like configuration using hip, knee, torso and hip-height features with visibility gating. Require evidence for both before and after states for a configurable dwell duration; do not infer sitting merely from upper-body visibility. If knee/hip landmarks are insufficient, mark as unobservable instead of guessing.
2. `hand_raised`: wrist above same-side shoulder in image coordinates (accounting for coordinate origin) for >=0.6 seconds, provided shoulder/wrist landmarks exceed confidence threshold. Two raised hands may count as one contiguous event, with side metadata if available.
3. Optional P1 `sustained_standing`: an upright-like pose sustained >=2 seconds, using conservative angle and visibility checks.

Event pipeline: per-frame feature extraction -> per-frame candidate states -> smoothing/debouncing -> temporal event segments -> merged intervals (configurable short gaps) -> saved suggested events. Every rule must include a test with synthetic landmark sequences and edge cases (occlusion, noise, short spikes, missing pose). Store `algorithm_version`, rule parameters, and `evidence` summary for explainability. Do not present arbitrary heuristic scores as calibrated probabilities. `confidence` may be null; show rule-based evidence/quality instead.

### FR4 — Review UI (P0)
Build a Streamlit application with these logical sections:
- Top: BehaviorLens title, local-only/demo notice, session selector/upload, status indicators.
- Overview: video player, analysis controls, session metadata, event counts.
- Timeline: chronological table or simple Plotly event-lane chart, color-separated by event type/status, with click/select via a supported Streamlit interaction. Selecting an event provides a replay action at or just before its timestamp. If precise seeking is not supported, rerender a player with `start_time` or show a generated short clip; disclose seek precision limitation.
- Review pane: event label, time interval, supporting evidence, accept/reject/edit controls, manual event insertion, reviewer note. Streamlit reruns must not duplicate review actions.
- Export pane: CSV/JSON download of reviewed records; keep rejected events with status rather than silently omitting them unless filter explicitly selected.

### FR5 — Review persistence and reproducibility (P0)
- SQLite with sessions, frames/landmarks (or compressed per-session JSON plus DB reference), events, and review history.
- Each event has immutable stable `event_id`, `session_id`, `source` (`algorithm` or `manual`), type, start/end timestamps, review status (`pending`, `confirmed`, `rejected`, `edited`), note, `created_at`, `updated_at`, and model/rule provenance when algorithmic.
- Preserve original algorithm proposal for edited events (e.g., original label/times in separate fields or an audit table).
- Rerunning detection should create a new analysis version or replace only unreviewed suggestions after explicit confirmation. Preserve reviewed versions.

### FR6 — Export (P0)
- CSV fields at minimum: session_id, event_id, event_type, start_s, end_s, duration_s, source, review_status, review_note, algorithm_version, created_at, updated_at.
- JSON includes session metadata, analysis settings, event records, and descriptive warning that suggestions have not been clinically validated.
- No raw video bytes, full local paths, identities, or facial crops in export.

### FR7 — Optional polish (P1)
- Frame sampling rate and per-rule threshold controls.
- Before/after thumbnails attached to events.
- Overlay video or pose animation.
- Summary counts and duration by event type, separated for proposed vs confirmed events.
- Reviewer agreement comparison using manually supplied ground truth and time-tolerance matching; precision, recall, F1 are only displayed if ground truth exists.

## 5. Architecture and stack
**Language:** Python 3.11 or compatible verified version. **UI:** Streamlit. **Vision:** MediaPipe Tasks Pose Landmarker. **Decode:** OpenCV (with FFmpeg where available). **Data:** SQLite via standard `sqlite3` or SQLModel if justified. **Plots:** Plotly optional. **Validation:** Pydantic optional. **Testing:** pytest. **Packaging:** pip/requirements.txt or pyproject.toml.

```
Local video -> validation/metadata -> OpenCV sampled frames
            -> MediaPipe pose -> feature extraction
            -> temporal rules -> suggested event records
            -> SQLite -> Streamlit review/timeline -> CSV/JSON
```

Keep core processing separate from UI and make the event rules callable on synthetic pose sequences without video inference. Prefer synchronous local processing with progress indicators; background queues, Redis, cloud storage and microservices are out of scope.

Recommended repository:
```
behaviorlens/
  app.py
  src/behaviorlens/
    __init__.py
    config.py
    models.py
    video_io.py
    pose.py
    features.py
    event_rules.py
    pipeline.py
    storage.py
    export.py
    demo_generator.py
  tests/
    test_event_rules.py
    test_storage.py
    test_export.py
    test_pipeline.py
  assets/models/          # downloaded weights, gitignored if required by license/size
  data/                   # local temporary processing, gitignored
  sample_data/            # only synthetic or staged consenting-adult footage
  scripts/download_model.py
  README.md
  requirements.txt
  .gitignore
```

## 6. Data model and example
Event JSON example:
```json
{
  "event_id": "uuid",
  "session_id": "uuid",
  "analysis_id": "uuid",
  "event_type": "hand_raised",
  "start_s": 12.4,
  "end_s": 14.0,
  "source": "algorithm",
  "review_status": "pending",
  "evidence": {"side": "right", "visible_frame_ratio": 0.94, "rule": "wrist_above_shoulder"},
  "algorithm_version": "rules-v0.1",
  "review_note": "",
  "original_proposal": null
}
```
Use typed data validation and ensure `0 <= start_s < end_s <= video_duration + tolerance`.

## 7. Sample data strategy
Generate a demonstrable offline sample without accessing private clinical videos. Preferred: record/ship a short video of a consenting adult acting out standing and hand-raising in a single-person uncluttered scene with consent to include the clip in the repo. If no such footage is available to the agent, provide a *clearly labeled synthetic pose-sequence fixture* that exercises the full event-rule/review UI and a synthetic stick-figure MP4 or animation for timeline demonstration. A rendered stick figure may not produce reliable MediaPipe landmarks: therefore distinguish synthetic fixtures that bypass inference from actual video-inference tests; never claim synthetic fixture processing validates real pose extraction. Include a tiny optional user-directed local-video test procedure.

## 8. UX design
Professional, simple, dark/neutral dashboard. Avoid medical-sounding determinations. Vocabulary: `Candidate observations`, `Review status`, `Observable movement`, `Analysis limitations`. Explicit banner: `Research/demo prototype. Not for clinical decision-making. Use only synthetic or staged footage with informed consent.` Show states: ready -> processing -> review-ready -> reviewed/exported. Empty and error states must explain next action.

## 9. Quality, privacy, and ethics
- Video processed locally; disable Streamlit telemetry and remote endpoints where possible. Do not write frames or identifying information to logs. Do not put uploaded footage under web-public directories.
- Store temporary local files in ignored paths, provide a `Delete session` action that deletes DB records and associated video/derived artifacts. Clarify backups beyond app control.
- No child images or real patient PHI in sample data or automated tests. Do not market or claim HIPAA compliance, FDA clearance, clinical validity, diagnostic capability, or insurance readiness.
- Missing landmark evidence should yield `unknown`/no event, not forced classification. Expose failure modes and request human review.
- No emotion/intent inference, stereotypy classification, aggression classification, or treatment prescriptions in this MVP.

## 10. Implementation sequence and priorities
**P0 — vertical slice first:** initialize repo/dependencies -> schema/data store -> synthetic fixture and event-rule tests -> video decode + pose extractor -> analysis pipeline -> Streamlit upload/player/event review -> persistence/export -> full smoke test -> README.

**P1 — only after P0 passes:** polished interactive timeline, overlay visualization, adjustable thresholds, metrics with actual ground truth, packaging refinements.

If time or environment is constrained, preserve a working synthetic-data review experience with clear mode labeling while making best effort to complete video inference. Never conceal missing functionality. Do not implement auth, hosting, billing, LLM summarization, or mobile applications.

## 11. Acceptance tests and definition of done
1. `pip install -r requirements.txt` installs dependencies on documented platform; model is obtainable with a documented command and correct license notice.
2. `streamlit run app.py` starts with no uncaught exception.
3. Demo fixture loads and visibly displays >=1 `hand_raised` and >=1 `standing_transition` example, all labeled **synthetic**.
4. New local MP4 upload processes through decode and pose inference or returns explicit actionable failure message.
5. Clicking/choosing an event displays its metadata and allows video seek/replay near the event.
6. Confirm, reject, edit and manual-add operations save to DB and persist after app restart.
7. CSV and JSON exports have expected fields and accurately represent the reviewed event statuses.
8. Invalid format, too-large upload, missing model, missing pose, short clip, bad fps and corrupted video are handled gracefully.
9. `pytest -q` passes rule/storage/export tests and smoke test; test output recorded in final handoff.
10. No patient footage, cloud API keys, private assets, or local data are committed to git.

## 12. Handoff requirements
In final response provide: (a) repo tree; (b) exact environment/install/start/test commands; (c) what you actually implemented vs omitted; (d) screenshots if feasible; (e) passing/failing test results; (f) privacy/clinical limitations; (g) top 3 next improvements. Include runnable sample/demo scenario and use a single clear main entrypoint. Do not claim application success without actually executing tests.

## 13. Reference docs
- MediaPipe Pose Landmarker for Python: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- Streamlit video API: https://docs.streamlit.io/develop/api-reference/media/st.video
