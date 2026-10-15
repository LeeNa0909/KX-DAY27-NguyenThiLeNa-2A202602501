"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 60%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
    enforce_critical_security_escalation,
)


console = Console()


def _escalation_questions(analysis: PRAnalysis) -> list[str]:
    if analysis.escalation_questions:
        return analysis.escalation_questions
    context = " ".join(
        [analysis.summary, *analysis.risk_factors, *(c.body for c in analysis.comments)]
    ).lower()
    questions = []
    if "priorit" in context or "valid value" in context:
        questions.append(
            "Which priority values should the application accept, and how should it handle an invalid or missing value?"
        )
    if any(word in context for word in ("schema", "database", "migration", "existing data")):
        questions.append(
            "How should existing task records be migrated or defaulted so the priority change preserves their data?"
        )
    if not questions:
        questions.append(
            "What product or production constraint could change the recommendation for this change?"
        )
    return questions[:4]


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = enforce_critical_security_escalation(llm.invoke([
            {"role": "system", "content": (
                "You are a careful senior software reviewer. Return a concise summary, "
                "concrete risk factors, and actionable comments in the requested schema. "
                "Only cite files and lines supported by the diff. Assess confidence "
                "from 0 to 1 as confidence that the review is complete, correct, and safe "
                "to post without human confirmation. Use 0.58–0.72 for manageable "
                "concerns such as missing tests, routine validation, or a small schema "
                "addition whose risks can be described in the review; those concerns "
                "alone do not justify confidence below 0.58. Reserve below 0.58 for "
                "critical unresolved context that could change the recommendation, such "
                "as unclear authentication, password hashing, credential storage, SQL "
                "safety, or a migration that could irreversibly damage existing data. "
                "When confidence is below 0.58, populate "
                "escalation_questions with 2 to 4 "
                "specific, context-rich questions for the reviewer. Reference files or "
                "sections in the diff where possible, and ask about intent or constraints "
                "that would change the review. Otherwise, leave escalation_questions empty. "
                "Critical security patterns are mandatory escalation triggers: MD5 or another "
                "fast unsalted password hash, plaintext credentials or tokens, SQL built with "
                "user input, or hard-coded identity in authentication. If any appears in the "
                "diff, confidence must be below 0.58 (usually 0.40-0.55) and you must include "
                "2 to 4 reviewer questions. Do not treat these as routine validation or missing tests."
            )},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ]))
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = _escalation_questions(a)

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    original = state["analysis"]
    answers = state.get("escalation_answers") or {}
    qa = "\n\n".join(f"Question: {question}\nReviewer answer: {answer}" for question, answer in answers.items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer context...[/dim]"):
        refined = enforce_critical_security_escalation(llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a careful senior software reviewer. Re-evaluate the pull "
                    "request using the reviewer context. Keep findings tied to the diff, "
                    "remove concerns resolved by the answers, and add any newly supported "
                    "findings. Return a complete PRAnalysis, including a calibrated "
                    "confidence score and any remaining escalation questions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PR title: {state['pr_title']}\n\n"
                    f"Diff:\n{state['pr_diff']}\n\n"
                    f"Initial analysis:\n{original.model_dump_json(indent=2)}\n\n"
                    f"Reviewer Q&A:\n{qa}"
                ),
            },
        ]))
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
