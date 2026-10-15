-- PostgreSQL schema for the structured HITL audit log.
CREATE TABLE IF NOT EXISTS audit_events (
    id                BIGSERIAL PRIMARY KEY,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    thread_id         TEXT NOT NULL,
    pr_url            TEXT NOT NULL,
    agent_id          TEXT NOT NULL,
    action            TEXT NOT NULL,
    confidence        DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    risk_level        TEXT NOT NULL CHECK (risk_level IN ('low', 'med', 'high')),
    reviewer_id       TEXT,
    decision          TEXT NOT NULL,
    reason            TEXT,
    execution_time_ms INTEGER NOT NULL CHECK (execution_time_ms >= 0)
);

CREATE INDEX IF NOT EXISTS audit_events_thread_idx
    ON audit_events (thread_id, id);
CREATE INDEX IF NOT EXISTS audit_events_pr_idx
    ON audit_events (pr_url, timestamp DESC);
CREATE INDEX IF NOT EXISTS audit_events_decision_idx
    ON audit_events (decision);
CREATE INDEX IF NOT EXISTS audit_events_risk_idx
    ON audit_events (risk_level);
