"""Structured audit storage with PostgreSQL and zero-setup SQLite support.

Set ``AUDIT_DATABASE_URL`` to a PostgreSQL connection URL to use PostgreSQL.
When it is unset, audit events use the local SQLite database configured by
``HITL_DB_PATH``. LangGraph checkpoints remain in SQLite in either mode.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from common.schemas import AuditEntry


AUDIT_DIR = Path(__file__).resolve().parent.parent / "audit"
SQLITE_SCHEMA_FILE = AUDIT_DIR / "schema.sql"
POSTGRES_SCHEMA_FILE = AUDIT_DIR / "schema.postgres.sql"


def db_path() -> str:
    """Return the SQLite file path used by the LangGraph checkpointer."""
    return os.environ.get("HITL_DB_PATH", "hitl_audit.db")


def postgres_dsn() -> str | None:
    """Return an explicitly configured PostgreSQL URL, if present."""
    dsn = os.environ.get("AUDIT_DATABASE_URL", "").strip()
    if not dsn:
        return None
    if not dsn.startswith(("postgres://", "postgresql://")):
        raise ValueError("AUDIT_DATABASE_URL must start with postgres:// or postgresql://")
    return dsn


async def _ensure_sqlite_schema(conn: aiosqlite.Connection) -> None:
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_events'"
    ) as cur:
        if await cur.fetchone() is not None:
            return
    await conn.executescript(SQLITE_SCHEMA_FILE.read_text(encoding="utf-8"))
    await conn.commit()


@asynccontextmanager
async def db_conn() -> AsyncIterator[aiosqlite.Connection]:
    """Open a SQLite connection for callers that need direct SQL access.

    Application and replay code should use the backend-neutral functions below.
    """
    conn = await aiosqlite.connect(db_path())
    conn.row_factory = aiosqlite.Row
    try:
        await _ensure_sqlite_schema(conn)
        yield conn
    finally:
        await conn.close()


async def _ensure_postgres_schema(conn: Any) -> None:
    await conn.execute(POSTGRES_SCHEMA_FILE.read_text(encoding="utf-8"))


async def write_audit_event(
    *,
    thread_id: str,
    pr_url: str,
    entry: AuditEntry,
) -> None:
    """Append one structured audit row to the selected audit backend."""
    values = (
        entry.timestamp,
        thread_id,
        pr_url,
        entry.agent_id,
        entry.action,
        entry.confidence,
        entry.risk_level,
        entry.reviewer_id,
        entry.decision,
        entry.reason,
        entry.execution_time_ms,
    )
    dsn = postgres_dsn()
    if dsn:
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError(
                "PostgreSQL audit is configured, but asyncpg is missing. Run `uv sync`."
            ) from exc
        conn = await asyncpg.connect(dsn)
        try:
            await _ensure_postgres_schema(conn)
            await conn.execute(
                """
                INSERT INTO audit_events (
                    timestamp, thread_id, pr_url,
                    agent_id, action, confidence, risk_level,
                    reviewer_id, decision, reason, execution_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                *values,
            )
        finally:
            await conn.close()
        return

    async with db_conn() as conn:
        await conn.execute(
            """
            INSERT INTO audit_events (
                timestamp, thread_id, pr_url,
                agent_id, action, confidence, risk_level,
                reviewer_id, decision, reason, execution_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (entry.timestamp.isoformat(), *values[1:]),
        )
        await conn.commit()


async def replay_events(thread_id: str) -> list[dict[str, Any]]:
    """Return every audit event for a thread, in insertion order."""
    dsn = postgres_dsn()
    if dsn:
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError(
                "PostgreSQL audit is configured, but asyncpg is missing. Run `uv sync`."
            ) from exc
        conn = await asyncpg.connect(dsn)
        try:
            await _ensure_postgres_schema(conn)
            rows = await conn.fetch(
                """
                SELECT id, timestamp, agent_id, action, confidence, risk_level,
                       reviewer_id, decision, reason, execution_time_ms
                  FROM audit_events
                 WHERE thread_id = $1
                 ORDER BY id
                """,
                thread_id,
            )
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT id, timestamp, agent_id, action, confidence, risk_level,
                   reviewer_id, decision, reason, execution_time_ms
              FROM audit_events
             WHERE thread_id = ?
             ORDER BY id
            """,
            (thread_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def list_threads(limit: int = 25) -> list[dict[str, Any]]:
    """List recent sessions with risk ordered by severity, on either backend."""
    limit = max(1, min(int(limit), 500))
    dsn = postgres_dsn()
    if dsn:
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError(
                "PostgreSQL audit is configured, but asyncpg is missing. Run `uv sync`."
            ) from exc
        conn = await asyncpg.connect(dsn)
        try:
            await _ensure_postgres_schema(conn)
            rows = await conn.fetch(
                """
                SELECT thread_id, pr_url,
                       MIN(timestamp) AS started,
                       MAX(timestamp) AS last_event,
                       CASE MAX(CASE risk_level
                           WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                           WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END
                           AS worst_risk,
                       COUNT(*) AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT $1
                """,
                limit,
            )
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id, pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level
                       WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END
                       AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]
