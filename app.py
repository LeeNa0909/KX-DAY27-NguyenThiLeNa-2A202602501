"""Streamlit approval UI for the HITL PR review agent.

Run with ``uv run streamlit run app.py``.
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path, list_threads
from exercises.exercise_4_audit import build_graph


load_dotenv()
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


for key, default in (
    ("thread_id", None),
    ("pr_url", ""),
    ("interrupt_payload", None),
    ("final", None),
    ("loaded_thread_id", None),
    ("session_table_version", 0),
):
    if key not in st.session_state:
        st.session_state[key] = default


async def list_recent_sessions(limit: int = 25) -> list[dict]:
    """Return recent audit sessions, ranking risk levels by severity."""
    return await list_threads(limit)


async def run_graph(
    pr_url: str,
    thread_id: str,
    resume_value=None,
    *,
    restore: bool = False,
):
    """Run, resume, or restore a graph using its durable SQLite checkpoint."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        if restore:
            snapshot = await graph.aget_state(config)
            pending = [
                interrupt
                for task in snapshot.tasks
                for interrupt in task.interrupts
            ]
            if pending:
                return {"__interrupt__": pending}
            return dict(snapshot.values)

        if resume_value is None:
            return await graph.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, config
            )
        return await graph.ainvoke(Command(resume=resume_value), config)


def render_approval_card(payload: dict) -> dict | None:
    """Render review details and return a reviewer choice when a button is clicked."""
    confidence = payload["confidence"]
    st.subheader(f"Approval requested · confidence {confidence:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for comment in payload.get("comments", []):
        line = comment.get("line") or "?"
        st.markdown(
            f"- **[{comment['severity']}]** `{comment['file']}:{line}` — "
            f"{comment['body']}"
        )

    with st.expander("Pull request diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    thread_id = st.session_state.thread_id or "new"
    feedback = st.text_area(
        "Feedback or edit request (required for Edit)",
        key=f"approval_feedback_{thread_id}",
    )
    approve_col, reject_col, edit_col = st.columns(3)
    if approve_col.button("Approve", type="primary", key=f"approve_{thread_id}"):
        return {"choice": "approve", "feedback": feedback}
    if reject_col.button("Reject", key=f"reject_{thread_id}"):
        return {"choice": "reject", "feedback": feedback}
    if edit_col.button("Edit", key=f"edit_{thread_id}"):
        if not feedback.strip():
            st.warning("Add an edit request before asking the agent to revise the review.")
            return None
        return {"choice": "edit", "feedback": feedback.strip()}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """Render risk context and collect an answer for every escalation question."""
    confidence = payload["confidence"]
    st.subheader(f"Strong escalation · confidence {confidence:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form(f"escalation_{st.session_state.thread_id}"):
        answers = {
            question: st.text_area(question, key=f"answer_{i}_{st.session_state.thread_id}")
            for i, question in enumerate(payload.get("questions", []))
        }
        submitted = st.form_submit_button("Submit answers")
    return answers if submitted else None


with st.sidebar:
    st.header("Recent sessions")
    try:
        recent_sessions = asyncio.run(list_recent_sessions())
    except Exception as exc:
        recent_sessions = []
        st.error(f"Could not read the audit trail: {exc}")

    if recent_sessions:
        session_rows = [
            {
                "thread_id": row["thread_id"],
                "PR": row["pr_url"],
                "Risk": row["worst_risk"],
                "Last event": row["last_event"],
                "Events": row["events"],
            }
            for row in recent_sessions
        ]
        selection = st.dataframe(
            session_rows,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"recent_sessions_{st.session_state.session_table_version}",
        )
        selected_rows = selection.selection.rows
        if selected_rows:
            selected = recent_sessions[selected_rows[0]]
            selected_thread = selected["thread_id"]
            if selected_thread != st.session_state.loaded_thread_id:
                st.session_state.thread_id = selected_thread
                st.session_state.pr_url = selected["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                try:
                    with st.spinner("Restoring this review session..."):
                        restored = asyncio.run(run_graph(
                            selected["pr_url"], selected_thread, restore=True
                        ))
                    if restored and "__interrupt__" in restored:
                        st.session_state.interrupt_payload = restored["__interrupt__"][0].value
                    else:
                        st.session_state.final = restored
                    st.session_state.loaded_thread_id = selected_thread
                except Exception as exc:
                    st.error(f"Could not restore session {selected_thread}: {exc}")
    else:
        st.caption("No review sessions yet.")


with st.form("start_review"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


if submitted and pr_url.strip():
    st.session_state.pr_url = pr_url.strip()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.loaded_thread_id = st.session_state.thread_id
    st.session_state.session_table_version += 1
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    try:
        with st.spinner("Fetching the PR and asking the LLM..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.final = result
    except Exception as exc:
        st.error(f"Review failed: {exc}")


payload = st.session_state.interrupt_payload
if payload is not None:
    if payload["kind"] == "approval_request":
        answer = render_approval_card(payload)
    elif payload["kind"] == "escalation":
        answer = render_escalation_card(payload)
    else:
        answer = None
        st.error(f"Unknown graph interrupt: {payload['kind']}")

    if answer is not None:
        try:
            with st.spinner("Resuming the review..."):
                result = asyncio.run(run_graph(
                    st.session_state.pr_url,
                    st.session_state.thread_id,
                    resume_value=answer,
                ))
            if "__interrupt__" in result:
                st.session_state.interrupt_payload = result["__interrupt__"][0].value
            else:
                st.session_state.interrupt_payload = None
                st.session_state.final = result
            st.rerun()
        except Exception as exc:
            st.error(f"Could not resume the review: {exc}")


final = st.session_state.final
if final is not None:
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    if action in {"committed", "auto_committed"}:
        st.success(f"{action} — review comment posted")
        if analysis is not None:
            if isinstance(analysis, dict):
                confidence = analysis["confidence"]
                summary = analysis["summary"]
            else:
                confidence = analysis.confidence
                summary = analysis.summary
            st.metric("Final confidence", f"{confidence:.0%}")
            st.markdown(summary)
        st.link_button("View review on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Review rejected; no comment was posted.")
    elif str(action).endswith("commit_failed"):
        st.error("GitHub could not accept the comment. Check the token and PR permissions.")
    else:
        st.info(f"Final action: {action}")

    st.caption(
        f"thread_id = {st.session_state.thread_id} · replay: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
