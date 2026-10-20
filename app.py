"""BehaviorLens Streamlit app. Run with: streamlit run app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from behaviorlens import config, export, metrics, pipeline  # noqa: E402
from behaviorlens.logging_setup import configure_logging  # noqa: E402
from behaviorlens.config import AnalysisSettings, RuleParams  # noqa: E402
from behaviorlens.models import Event, new_id  # noqa: E402
from behaviorlens.pose import ModelNotFoundError, PoseInferenceError, download_model  # noqa: E402
from behaviorlens.storage import NotFoundError, SchemaVersionError, Store  # noqa: E402
from behaviorlens.video_io import VideoValidationError  # noqa: E402

st.set_page_config(page_title="BehaviorLens", page_icon="🔎", layout="wide")

LABELS = {
    "hand_raised": "Hand raised",
    "standing_transition": "Standing transition",
    "sustained_standing": "Sustained standing",
    "other_observable": "Other observable",
}
STATUS_COLORS = {"pending": "#c9a227", "confirmed": "#3fa66b", "rejected": "#d0574f", "edited": "#4f8fd0"}
STEPS = [("ready", "Ready"), ("processing", "Processing"), ("review_ready", "Review-ready"),
         ("reviewed", "Reviewed"), ("exported", "Exported")]


@st.cache_resource
def get_store(root: str) -> Store:
    configure_logging()
    return Store(Path(root))


try:
    store = get_store(str(config.data_dir()))
except SchemaVersionError as exc:
    st.error(str(exc))
    st.stop()
ss = st.session_state
ss.setdefault("flash", [])
ss.setdefault("seek", {})
ss.setdefault("selected_event", {})


def flash(kind: str, msg: str) -> None:
    ss["flash"].append((kind, msg))


# ------------------------------------------------------------------ actions (callbacks)
def action_load_demo() -> None:
    try:
        sid = pipeline.create_demo_session(store)
        pipeline.run_analysis(store, sid)
        ss["session_select"] = sid
        flash("success", "Loaded the SYNTHETIC demo session (programmatic pose data, no inference).")
    except Exception as exc:  # surface any failure without crashing the UI
        flash("error", f"Could not create the demo session: {exc}")


def action_create_upload() -> None:
    up = ss.get("uploader")
    if up is None:
        flash("warning", "Choose a video file first.")
        return
    try:
        with st.spinner("Validating, fingerprinting and storing the video locally (non-H.264 files are "
                        "transcoded for browser playback)…"):
            sid = pipeline.create_upload_session(store, up.name, up, up.size)
        ss["session_select"] = sid
        flash("success", "Video stored locally. Run analysis to detect candidate observations.")
    except VideoValidationError as exc:
        flash("error", str(exc))
    except Exception as exc:
        flash("error", f"Upload failed: {exc}")


def action_delete(session_id: str) -> None:
    try:
        store.delete_session(session_id)
        ss.pop("session_select", None)
        flash("success", "Session, events, review history and local video/derived files deleted. "
                         "Copies in external backups (if any) are outside the app's control.")
    except NotFoundError:
        flash("warning", "Session already deleted.")


def action_download_model() -> None:
    try:
        download_model()
        flash("success", "Pose model downloaded.")
    except Exception as exc:
        flash("error", f"Model download failed: {exc}. Run `python scripts/download_model.py` manually.")


def action_review(event_id: str, status: str, request_id: str) -> None:
    note = ss.get(f"note_{event_id}")
    store.set_status(event_id, status, note=note, request_id=request_id)


def action_select(session_id: str, event_id: str | None) -> None:
    ss["selected_event"][session_id] = event_id
    if event_id:
        e = store.get_event(event_id)
        seek(session_id, max(0.0, e.start_s - 1.0))


def seek(session_id: str, t: float) -> None:
    # A tiny nonce forces the player to re-render even when seeking to the same time.
    prev = ss["seek"].get(session_id, 0.0)
    ss["seek"][session_id] = round(t, 1) + (0.001 if abs(prev - round(t, 1)) < 1e-6 else 0.0)


def on_table_select(session_id: str, ids: list[str]) -> None:
    rows = ss[f"table_{session_id}"].selection.rows
    if rows:
        action_select(session_id, ids[rows[0]])


def on_chart_select(session_id: str) -> None:
    pts = ss[f"chart_{session_id}"].selection.points
    if pts:
        cd = pts[0].get("customdata")
        if cd:
            action_select(session_id, cd[0] if isinstance(cd, list) else cd)


# ------------------------------------------------------------------ layout helpers
def session_label(s: dict) -> str:
    mode = "SYNTHETIC" if s["data_mode"] == "synthetic" else "video"
    # The short ID keeps labels unique (the selectbox resolves choices by label).
    return (f"{s['source_filename']} · {mode} · {s['status']} · "
            f"{s['created_at'][:16].replace('T', ' ')} · {s['session_id'][:6]}")


def status_stepper(status: str) -> None:
    if status == "failed":
        st.markdown("**Status:** :red[failed]")
        return
    order = [k for k, _ in STEPS]
    cur = order.index(status) if status in order else 0
    parts = []
    for i, (_key, label) in enumerate(STEPS):
        if i < cur:
            parts.append(f":green[✓ {label}]")
        elif i == cur:
            parts.append(f"**:blue[● {label}]**")
        else:
            parts.append(f":gray[○ {label}]")
    st.markdown(" → ".join(parts))


def timeline_chart(session_id: str, events: list[Event], duration: float) -> None:
    fig = go.Figure()
    for status, color in STATUS_COLORS.items():
        evs = [e for e in events if e.review_status == status]
        if not evs:
            continue
        fig.add_trace(go.Bar(
            orientation="h", base=[e.start_s for e in evs], x=[max(e.duration_s, 0.15) for e in evs],
            y=[LABELS.get(e.event_type, e.event_type) for e in evs], name=status,
            marker=dict(color=color, line=dict(width=0)),
            customdata=[[e.event_id] for e in evs],
            hovertemplate="%{y}<br>start %{base:.1f}s, duration %{x:.1f}s<extra>" + status + "</extra>",
        ))
    sel = ss["selected_event"].get(session_id)
    for e in events:
        if e.event_id == sel:
            fig.add_vrect(x0=e.start_s, x1=e.end_s, fillcolor="#ffffff", opacity=0.12, line_width=0)
    fig.update_layout(
        barmode="overlay", height=230, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title="seconds", range=[0, max(duration, 1)]), yaxis=dict(title=None),
        legend=dict(orientation="h", y=1.15), clickmode="event+select", dragmode=False,
    )
    st.plotly_chart(fig, key=f"chart_{session_id}", on_select=lambda: on_chart_select(session_id),
                    selection_mode="points", width="stretch")


def events_table(session_id: str, events: list[Event]) -> None:
    df = pd.DataFrame([{
        "type": LABELS.get(e.event_type, e.event_type),
        "start_s": e.start_s, "end_s": e.end_s, "duration_s": e.duration_s,
        "status": e.review_status, "source": e.source, "note": e.review_note,
    } for e in events])
    ids = [e.event_id for e in events]
    st.dataframe(df, key=f"table_{session_id}", on_select=lambda: on_table_select(session_id, ids),
                 selection_mode="single-row", hide_index=True, width="stretch", height=260)


def summary_counts(events: list[Event]) -> pd.DataFrame:
    rows = []
    for t in config.MANUAL_EVENT_TYPES:
        evs = [e for e in events if e.event_type == t]
        if not evs:
            continue
        accepted = [e for e in evs if e.review_status in ("confirmed", "edited")]
        rows.append({
            "event type": LABELS[t],
            "proposed (algorithm)": sum(e.source == "algorithm" for e in evs),
            "pending": sum(e.review_status == "pending" for e in evs),
            "confirmed/edited": len(accepted),
            "rejected": sum(e.review_status == "rejected" for e in evs),
            "manual": sum(e.source == "manual" for e in evs),
            "confirmed duration (s)": round(sum(e.duration_s for e in accepted), 1),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Sessions")
    sessions = store.list_sessions()
    ids = [s["session_id"] for s in sessions]
    if ss.get("session_select") not in ids:
        ss["session_select"] = ids[0] if ids else None
    labels = {s["session_id"]: session_label(s) for s in sessions}
    st.selectbox("Session", ids, key="session_select", format_func=lambda i: labels.get(i, i),
                 placeholder="No sessions yet")
    st.button("Load synthetic demo session", on_click=action_load_demo, width="stretch",
              help="Programmatically generated pose data and a stick-figure video. No real footage.")
    st.divider()
    st.subheader("Upload video")
    st.file_uploader("Staged footage of consenting adults only (MP4/MOV/AVI, ≤100 MB, ≤5 min)",
                     type=[e.lstrip(".") for e in config.ALLOWED_EXTENSIONS], key="uploader")
    st.button("Create session from upload", on_click=action_create_upload, width="stretch")
    st.caption("Files stay on this machine under the gitignored `data/` directory.")
    st.divider()
    if not config.model_path().is_file():
        st.error("Pose model not downloaded. Real video analysis needs it.")
        st.button("Download pose model (~6 MB)", on_click=action_download_model)
        st.caption("or run `python scripts/download_model.py`")
    else:
        st.caption(f"Pose model: `{config.MODEL_NAME}` · local inference · "
                   f"[model card / license]({config.MODEL_CARD_URL})")

# ------------------------------------------------------------------ header
st.title("BehaviorLens")
st.warning(config.DISCLAIMER, icon="⚠️")
for kind, msg in ss["flash"]:
    getattr(st, kind)(msg)
ss["flash"] = []

session_id = ss.get("session_select")
if not session_id:
    st.info("No session yet. Use **Load synthetic demo session** in the sidebar to explore the review "
            "workflow, or upload a short staged video of a consenting adult.")
    st.stop()

try:
    session = store.get_session(session_id)
except NotFoundError:
    st.error("This session no longer exists. Pick another one in the sidebar.")
    st.stop()

synthetic = session["data_mode"] == "synthetic"
st.subheader(session["source_filename"])
if synthetic:
    st.info("**SYNTHETIC FIXTURE.** Pose landmarks were generated programmatically and MediaPipe "
            "inference was *not* run. Results demonstrate the rules and review workflow only and do not "
            "validate real pose extraction.", icon="🧪")
else:
    st.caption("Candidate observations from local MediaPipe pose inference + interpretable rules. "
               "Algorithm-generated suggestions require human review.")
status_stepper(session["status"])
if session.get("error"):
    st.error(f"Last analysis failed: {session['error']}")

events = store.list_events(session_id)
analysis = store.latest_analysis(session_id)
duration = session["duration_s"] or 0.0

tab_review, tab_export, tab_details = st.tabs(["Review", "Export", "Analysis details"])

# ------------------------------------------------------------------ review tab
with tab_review:
    left, right = st.columns([3, 2], gap="large")
    with left:
        preview = store.path(session.get("preview_relpath"))
        overlay = store.path(session.get("overlay_relpath"))
        show_overlay = False
        if overlay and overlay.is_file():
            show_overlay = st.toggle("Show pose overlay (sampled frames, low resolution)", key=f"ov_{session_id}")
        video_path = overlay if show_overlay else preview
        if video_path and video_path.is_file():
            st.video(str(video_path), start_time=float(ss["seek"].get(session_id, 0.0)))
            st.caption("Selecting an event re-renders the player ~1 s before the event start. Seek "
                       "precision depends on the browser and keyframes (typically within ~1 s).")
        else:
            st.warning("No playable video for this session.")
        if not synthetic and analysis and not (overlay and overlay.is_file()):
            if st.button("Render pose overlay preview"):
                with st.spinner("Rendering overlay locally…"):
                    try:
                        pipeline.render_overlay(store, session_id)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Overlay rendering failed: {exc}")

        st.markdown("#### Candidate observations timeline")
        if events:
            timeline_chart(session_id, events, duration)
            events_table(session_id, events)
            st.caption("Click a bar or a table row to review that event.")
        elif analysis:
            st.info("Analysis ran but produced no candidate observations. Check *Analysis details* for "
                    "observability warnings, or add a manual event.")
        else:
            st.info("Not analyzed yet. Use **Run analysis** on the right.")

    with right:
        with st.expander("Analysis controls", expanded=analysis is None):
            fps = st.radio("Sampling rate", config.SAMPLE_FPS_CHOICES, horizontal=True,
                           format_func=lambda v: f"{v:g} fps", key=f"fps_{session_id}", disabled=synthetic)
            d = RuleParams()
            c1, c2 = st.columns(2)
            vis = c1.slider("Landmark visibility threshold", 0.1, 0.9, d.visibility_threshold, 0.05)
            hmin = c2.slider("Hand raised: min duration (s)", 0.2, 3.0, d.hand_raise_min_s, 0.1)
            dwell = c1.slider("Standing: dwell before/after (s)", 0.4, 3.0, d.dwell_s, 0.1)
            gap = c2.slider("Bridge gaps up to (s)", 0.0, 1.5, d.merge_gap_s, 0.1)
            sustained = st.checkbox("Also suggest sustained standing (≥2 s)", d.enable_sustained_standing)
            params = RuleParams(visibility_threshold=vis, hand_raise_min_s=hmin, dwell_s=dwell,
                                merge_gap_s=gap, enable_sustained_standing=sustained)
            pending = store.pending_algorithm_count(session_id)
            confirm_ok = True
            if analysis and pending:
                confirm_ok = st.checkbox(f"Replace {pending} unreviewed suggestion(s). Reviewed and manual "
                                         "events are kept.", key=f"confirm_rerun_{session_id}")
            need_model = not synthetic and not config.model_path().is_file()
            if need_model:
                st.error("Download the pose model first (sidebar).")
            if st.button("Run analysis" if analysis is None else "Re-run analysis", type="primary",
                         disabled=not confirm_ok or need_model, width="stretch"):
                bar = st.progress(0.0, text="Starting…")
                try:
                    res = pipeline.run_analysis(
                        store, session_id, AnalysisSettings(sample_fps=fps, rules=params),
                        progress=lambda p, m: bar.progress(min(max(p, 0.0), 1.0), text=m))
                    flash("success", f"Analysis complete: {res['added']} new suggestion(s); "
                                     f"{res['replaced_pending']} unreviewed replaced; "
                                     f"{res['skipped_already_reviewed']} skipped (already reviewed).")
                except (VideoValidationError, ModelNotFoundError, PoseInferenceError) as exc:
                    flash("error", str(exc))
                except Exception as exc:
                    flash("error", f"Analysis failed: {exc}")
                st.rerun()

        if events:
            c = st.columns(4)
            for col, status in zip(c, ("pending", "confirmed", "edited", "rejected")):
                col.metric(status.capitalize(), sum(e.review_status == status for e in events))

        st.markdown("#### Review")
        sel_id = ss["selected_event"].get(session_id)
        by_id = {e.event_id: e for e in events}
        if events:
            order = list(by_id)
            idx = order.index(sel_id) if sel_id in by_id else -1
            nav = st.columns([1, 1, 2])
            prev_id = order[idx - 1] if idx > 0 else None
            next_id = order[idx + 1] if idx + 1 < len(order) else None
            nav[0].button("◀ Prev", disabled=prev_id is None, on_click=action_select, args=(session_id, prev_id))
            nav[1].button("Next ▶", disabled=next_id is None, on_click=action_select, args=(session_id, next_id))
            nxt_pending = next((e.event_id for e in events
                                if e.review_status == "pending" and e.event_id != sel_id), None)
            nav[2].button("Next pending", disabled=nxt_pending is None, on_click=action_select,
                          args=(session_id, nxt_pending))
        ev = by_id.get(sel_id)
        if ev is None:
            st.caption("Select an event from the timeline or table.")
        else:
            st.markdown(f"**{LABELS.get(ev.event_type, ev.event_type)}** · {ev.start_s:.2f}s – {ev.end_s:.2f}s "
                        f"({ev.duration_s:.2f}s) · status `{ev.review_status}` · source `{ev.source}`")
            if ev.source == "algorithm":
                st.caption("Algorithm-generated suggestion describing observable movement only. "
                           f"Rule `{ev.evidence.get('rule')}`, {ev.algorithm_version}. No calibrated confidence.")
            if ev.original_proposal:
                o = ev.original_proposal
                st.caption(f"Original proposal: {o['event_type']} {o['start_s']:.2f}s – {o['end_s']:.2f}s")
            thumbs = pipeline.event_thumbnails(store, session_id, ev.start_s, ev.end_s)
            if thumbs:
                tc = st.columns(len(thumbs))
                for col, (t, path), label in zip(tc, thumbs, ("Before", "After")):
                    col.image(str(path), caption=f"{label} · {t:.1f}s", width="stretch")
            with st.expander("Supporting evidence"):
                st.json(ev.evidence)
            st.text_input("Reviewer note", value=ev.review_note, key=f"note_{ev.event_id}")
            b = st.columns(3)
            b[0].button("✓ Confirm", on_click=action_review, args=(ev.event_id, "confirmed", new_id()),
                        width="stretch", type="primary")
            b[1].button("✗ Reject", on_click=action_review, args=(ev.event_id, "rejected", new_id()),
                        width="stretch")
            b[2].button("↻ Replay", on_click=seek, args=(session_id, max(0.0, ev.start_s - 1.0)),
                        width="stretch")
            with st.form(f"edit_{ev.event_id}"):
                st.markdown("Edit event")
                types = list(config.MANUAL_EVENT_TYPES)
                et = st.selectbox("Type", types, index=types.index(ev.event_type) if ev.event_type in types else 0,
                                  format_func=LABELS.get)
                c1, c2 = st.columns(2)
                s0 = c1.number_input("Start (s)", 0.0, float(duration + 1), float(ev.start_s), 0.1, format="%.2f")
                s1 = c2.number_input("End (s)", 0.0, float(duration + 1), float(ev.end_s), 0.1, format="%.2f")
                if st.form_submit_button("Save edit"):
                    try:
                        store.edit_event(ev.event_id, event_type=et, start_s=s0, end_s=s1,
                                         note=ss.get(f"note_{ev.event_id}"),
                                         request_id=ss.setdefault(f"edit_req_{ev.event_id}", new_id()))
                        ss.pop(f"edit_req_{ev.event_id}", None)
                        flash("success", "Edit saved; original proposal preserved.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        with st.expander("Add manual event"):
            with st.form(f"manual_{session_id}", clear_on_submit=True):
                types = list(config.MANUAL_EVENT_TYPES)
                mt = st.selectbox("Type", types, format_func=LABELS.get)
                c1, c2 = st.columns(2)
                m0 = c1.number_input("Start (s)", 0.0, float(duration + 1), 0.0, 0.1, format="%.2f")
                m1 = c2.number_input("End (s)", 0.0, float(duration + 1), min(1.0, float(duration)), 0.1, format="%.2f")
                mnote = st.text_input("Note")
                if st.form_submit_button("Add event"):
                    try:
                        e = store.add_manual_event(session_id, mt, m0, m1, mnote,
                                                   request_id=ss.setdefault(f"manual_req_{session_id}", new_id()))
                        ss.pop(f"manual_req_{session_id}", None)
                        ss["selected_event"][session_id] = e.event_id
                        flash("success", "Manual event added.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

# ------------------------------------------------------------------ export tab
with tab_export:
    st.markdown("Exports contain event records, review status and provenance. They never include video, "
                "file paths, filenames or images. Rejected events are kept (with status) unless filtered out.")
    statuses = st.multiselect("Include review statuses", list(config.REVIEW_STATUSES),
                              default=list(config.REVIEW_STATUSES), key=f"exp_{session_id}")
    filt = None if set(statuses) == set(config.REVIEW_STATUSES) else statuses
    n = len(store.list_events(session_id, filt)) if statuses else 0
    st.caption(f"{n} event(s) will be exported.")
    mark = lambda: store.mark_exported(session_id)  # noqa: E731
    c1, c2 = st.columns(2)
    stem = f"behaviorlens_{session_id[:8]}"
    c1.download_button("Download CSV", export.to_csv(store, session_id, filt) if statuses else "",
                       f"{stem}.csv", "text/csv", on_click=mark, disabled=not statuses, width="stretch")
    c2.download_button("Download JSON", export.to_json(store, session_id, filt) if statuses else "",
                       f"{stem}.json", "application/json", on_click=mark, disabled=not statuses, width="stretch")
    st.divider()
    with st.expander("Delete this session"):
        st.markdown("Deletes database records, review history, the stored video and all derived files.")
        sure = st.checkbox("I understand this cannot be undone", key=f"del_{session_id}")
        st.button("Delete session", type="primary", disabled=not sure, on_click=action_delete, args=(session_id,))

# ------------------------------------------------------------------ details tab
with tab_details:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Session")
        st.json({k: session[k] for k in ("session_id", "data_mode", "status", "fps", "width", "height",
                                          "frame_count", "duration_s", "created_at")})
    with c2:
        st.markdown("#### Latest analysis")
        if analysis:
            for w in analysis["warnings"]:
                st.warning(w)
            st.json({k: analysis[k] for k in ("analysis_id", "created_at", "algorithm_version",
                                              "model_version", "settings", "stats")}, expanded=False)
        else:
            st.caption("No analysis yet.")
    if events:
        st.markdown("#### Summary by event type")
        st.dataframe(summary_counts(events), hide_index=True, width="stretch")

    st.markdown("#### Agreement with ground truth")
    gt_file = st.file_uploader("Ground-truth CSV (columns: event_type,start_s,end_s)", type=["csv"],
                               key=f"gt_{session_id}")
    if gt_file is not None and st.button("Use this ground truth"):
        try:
            store.set_ground_truth(session_id, metrics.parse_ground_truth_csv(gt_file.getvalue().decode()))
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    truth = session.get("ground_truth")
    if not truth:
        st.caption("No ground truth supplied, so precision/recall are not shown.")
    else:
        if synthetic:
            st.caption("Ground truth is the scripted timeline of the SYNTHETIC fixture, not human annotation "
                       "of real footage.")
        tol = st.slider("Time tolerance (s) for start and end", 0.2, 3.0, 1.0, 0.1)
        proposals = [e.original_proposal or e.model_dump() for e in events if e.source == "algorithm"]
        reviewed = [e.model_dump() for e in events if e.review_status in ("confirmed", "edited")]
        for title, preds in (("Algorithm proposals (as originally proposed)", proposals),
                             ("Reviewer-accepted events (confirmed, edited, manual)", reviewed)):
            m = metrics.match_events(preds, truth, tol)
            st.markdown(f"**{title}** — overall P={m['overall']['precision']} R={m['overall']['recall']} "
                        f"F1={m['overall']['f1']}")
            st.dataframe(pd.DataFrame([{"event type": LABELS.get(t, t), **v} for t, v in m["per_type"].items()]),
                         hide_index=True, width="stretch")

    with st.expander("Review history (audit log)"):
        problems = store.verify_integrity(session_id)
        if problems:
            st.error("Audit trail integrity check FAILED. The review history or events were changed outside "
                     "the app; treat this session's review data as unreliable.\n\n- " + "\n- ".join(problems))
        else:
            rows, head = store.review_chain_head(session_id)
            st.caption(f"Audit trail verified: {rows} hash-chained record(s)"
                       + (f", chain head `{head[:16]}…` (also included in JSON exports)." if head else "."))
        hist = store.review_history(session_id)
        if hist:
            st.dataframe(pd.DataFrame([{k: h[k] for k in ("at", "action", "event_id")} for h in hist]),
                         hide_index=True, width="stretch")
        else:
            st.caption("No review actions yet.")
