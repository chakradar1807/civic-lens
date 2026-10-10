-- PostgreSQL table for storing the ML result of each report (works with or without PostGIS).
CREATE TABLE IF NOT EXISTS ai_triage (
    id                     BIGSERIAL PRIMARY KEY,
    report_id              BIGINT      NOT NULL UNIQUE,       -- FK to your reports/tickets table
    is_valid               BOOLEAN     NOT NULL,
    status                 VARCHAR(24) NOT NULL,              -- AUTO_DISPATCH | NEEDS_REVIEW | NEEDS_SECOND_REPORT | REJECTED
    reject_reason          VARCHAR(64),
    issue_type             VARCHAR(40),
    confidence             REAL,
    severity               SMALLINT CHECK (severity BETWEEN 1 AND 5),
    priority_score         SMALLINT,
    priority_level         VARCHAR(10),
    bbox                   JSONB,                             -- [x1, y1, x2, y2]
    department             VARCHAR(80),
    estimated_fix_minutes  INTEGER,
    materials              JSONB,
    tools                  JSONB,
    crew_size              SMALLINT,
    explanation            JSONB,
    validation_flags       JSONB,
    trust_penalty          REAL,
    image_hash             VARCHAR(16),                       -- 64-bit dHash (hex) to catch re-used photos
    model_mode             VARCHAR(8),                        -- 'yolo' or 'mock'
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_triage_status ON ai_triage (status);
CREATE INDEX IF NOT EXISTS idx_ai_triage_issue  ON ai_triage (issue_type);
CREATE INDEX IF NOT EXISTS idx_ai_triage_hash   ON ai_triage (image_hash);

-- Ticket statuses the backend should support:
--   PENDING_AI -> AUTO_DISPATCH / NEEDS_REVIEW / NEEDS_SECOND_REPORT / REJECTED -> IN_PROGRESS -> RESOLVED
