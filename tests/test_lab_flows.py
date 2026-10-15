from __future__ import annotations

import os
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich.console import Console

from common.github import PullRequest
from common.schemas import (
    AuditEntry,
    PRAnalysis,
    ReviewComment,
    enforce_critical_security_escalation,
)
from common.db import list_threads, replay_events, write_audit_event
from exercises import exercise_1_confidence as exercise_1
from exercises import exercise_4_audit as exercise_4

# Rich's default Windows console uses the active OEM code page. Capture output
# in memory so these tests behave the same in Windows terminals and CI.
exercise_1.console = Console(file=io.StringIO(), force_terminal=False)
exercise_4.console = Console(file=io.StringIO(), force_terminal=False)


PR_URL = "https://github.com/example/project/pull/7"


def make_pr() -> PullRequest:
    return PullRequest(
        url=PR_URL,
        owner="example",
        repo="project",
        number=7,
        title="Add a feature",
        author="student",
        base_ref="main",
        head_ref="feature",
        head_sha="abcdef123456",
        diff="diff --git a/task.py b/task.py\n+priority = 1\n",
        files_changed=["task.py"],
    )


def make_analysis(confidence: float, summary: str = "Adds task priority.") -> PRAnalysis:
    return PRAnalysis(
        summary=summary,
        risk_factors=["Input validation needs confirmation."] if confidence < 0.58 else [],
        comments=[ReviewComment(
            file="task.py",
            line=1,
            severity="suggestion",
            body="Validate the priority value.",
        )],
        confidence=confidence,
        confidence_reasoning="Confidence is calibrated for the available diff.",
        escalation_questions=(
            ["Which priority values should be accepted?"] if confidence < 0.58 else []
        ),
    )


class FakeSyncModel:
    def __init__(self, analysis: PRAnalysis):
        self.analysis = analysis

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self.analysis


class FakeAsyncModel:
    def __init__(self, *analyses: PRAnalysis):
        self.analyses = list(analyses)

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        if not self.analyses:
            raise AssertionError("Unexpected extra LLM call in flow")
        return self.analyses.pop(0)


class ConfidenceRoutingTests(unittest.TestCase):
    def test_critical_security_findings_cap_confidence_but_routine_review_does_not(self):
        critical = make_analysis(0.65).model_copy(update={
            "risk_factors": [
                "MD5 password hashing, plaintext token storage, SQL injection, "
                "and hard-coded user ID are unsafe."
            ]
        })
        capped = enforce_critical_security_escalation(critical)
        self.assertEqual(capped.confidence, 0.57)
        self.assertIn("policy cap applied", capped.confidence_reasoning)

        routine = enforce_critical_security_escalation(make_analysis(0.68))
        self.assertEqual(routine.confidence, 0.68)

    def test_auto_approval_human_approval_and_escalation_routes(self):
        expected = (
            (0.85, "auto_approve", "auto_approved"),
            (0.68, "human_approval", "pending_human_approval"),
            (0.55, "escalate", "pending_escalation"),
        )
        for confidence, decision, final_action in expected:
            with self.subTest(confidence=confidence):
                model = FakeSyncModel(make_analysis(confidence))
                with (
                    patch.object(exercise_1, "fetch_pr", return_value=make_pr()),
                    patch.object(exercise_1, "get_llm", return_value=model),
                ):
                    result = exercise_1.build_graph().invoke({"pr_url": PR_URL})
                self.assertEqual(result["decision"], decision)
                self.assertEqual(result["final_action"], final_action)


class ExerciseFourFlowTests(unittest.IsolatedAsyncioTestCase):
    async def run_flow(self, analyses, answers=()):
        model = FakeAsyncModel(*analyses)
        graph = exercise_4.build_graph(MemorySaver())
        cfg = {"configurable": {"thread_id": f"test-{id(self)}-{len(answers)}-{len(analyses)}"}}
        with (
            patch.object(exercise_4, "fetch_pr", return_value=make_pr()),
            patch.object(exercise_4, "get_llm", return_value=model),
            patch.object(exercise_4, "audit", new_callable=AsyncMock),
            patch.object(exercise_4, "_post", return_value="committed") as post,
        ):
            result = await graph.ainvoke(
                {"pr_url": PR_URL, "thread_id": cfg["configurable"]["thread_id"]}, cfg
            )
            interrupts = []
            answer_iter = iter(answers)
            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                interrupts.append(payload)
                result = await graph.ainvoke(
                    Command(resume=next(answer_iter)),
                    cfg,
                )
            return result, interrupts, post

    async def test_approve_posts_review(self):
        result, interrupts, post = await self.run_flow(
            [make_analysis(0.68)], [{"choice": "approve", "feedback": "Looks good."}]
        )
        self.assertEqual([item["kind"] for item in interrupts], ["approval_request"])
        self.assertEqual(result["final_action"], "committed")
        post.assert_called_once()

    async def test_edit_synthesizes_then_requests_final_approval(self):
        result, interrupts, post = await self.run_flow(
            [make_analysis(0.68), make_analysis(0.76, "Revised review.")],
            [
                {"choice": "edit", "feedback": "Clarify the validation recommendation."},
                {"choice": "approve", "feedback": ""},
            ],
        )
        self.assertEqual([item["kind"] for item in interrupts], [
            "approval_request", "approval_request"
        ])
        self.assertEqual(result["analysis"].summary, "Revised review.")
        self.assertEqual(result["final_action"], "committed")
        post.assert_called_once()

    async def test_escalation_answers_are_synthesized_before_approval(self):
        result, interrupts, post = await self.run_flow(
            [make_analysis(0.55), make_analysis(0.68, "Review refined with context.")],
            [
                {"Which priority values should be accepted?": "Only 0, 1, or 2."},
                {"choice": "approve", "feedback": ""},
            ],
        )
        self.assertEqual([item["kind"] for item in interrupts], [
            "escalation", "approval_request"
        ])
        self.assertEqual(result["analysis"].summary, "Review refined with context.")
        self.assertEqual(result["final_action"], "committed")
        post.assert_called_once()

    async def test_auto_approval_commits_without_interrupt(self):
        result, interrupts, post = await self.run_flow([make_analysis(0.85)])
        self.assertEqual(interrupts, [])
        self.assertEqual(result["final_action"], "auto_committed")
        post.assert_called_once()


class AuditReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_audit_write_replay_and_session_listing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_file = str(Path(temp_dir) / "audit.db")
            with patch.dict(os.environ, {"HITL_DB_PATH": db_file, "AUDIT_DATABASE_URL": ""}):
                for action, risk, decision in (
                    ("analyze", "med", "pending"),
                    ("escalate", "high", "escalate"),
                ):
                    await write_audit_event(
                        thread_id="replay-test",
                        pr_url=PR_URL,
                        entry=AuditEntry(
                            agent_id="test-agent",
                            action=action,
                            confidence=0.55,
                            risk_level=risk,
                            decision=decision,
                            reason="test event",
                            execution_time_ms=3,
                        ),
                    )

                events = await replay_events("replay-test")
                sessions = await list_threads()

        self.assertEqual([event["action"] for event in events], ["analyze", "escalate"])
        self.assertEqual(sessions[0]["thread_id"], "replay-test")
        self.assertEqual(sessions[0]["worst_risk"], "high")
        self.assertEqual(sessions[0]["events"], 2)

    async def test_postgres_backend_writes_and_reads_using_asyncpg(self):
        connection = AsyncMock()
        connection.fetch.side_effect = [
            [{"id": 1, "timestamp": "now", "agent_id": "test-agent",
              "action": "analyze", "confidence": 0.8, "risk_level": "low",
              "reviewer_id": None, "decision": "pending", "reason": None,
              "execution_time_ms": 1}],
            [{"thread_id": "pg-thread", "pr_url": PR_URL, "started": "now",
              "last_event": "now", "worst_risk": "low", "events": 1}],
        ]
        connect = AsyncMock(return_value=connection)
        fake_asyncpg = SimpleNamespace(connect=connect)

        with (
            patch.dict(os.environ, {"AUDIT_DATABASE_URL": "postgresql://user:pw@db/hitl"}),
            patch.dict(sys.modules, {"asyncpg": fake_asyncpg}),
        ):
            await write_audit_event(
                thread_id="pg-thread",
                pr_url=PR_URL,
                entry=AuditEntry(
                    agent_id="test-agent",
                    action="analyze",
                    confidence=0.8,
                    risk_level="low",
                    decision="pending",
                    execution_time_ms=1,
                ),
            )
            events = await replay_events("pg-thread")
            sessions = await list_threads()

        self.assertEqual(connect.await_count, 3)
        self.assertEqual(connection.execute.await_count, 4)  # schema + insert, per operation
        self.assertEqual(events[0]["action"], "analyze")
        self.assertEqual(sessions[0]["thread_id"], "pg-thread")
        self.assertEqual(connection.close.await_count, 3)


if __name__ == "__main__":
    unittest.main()
