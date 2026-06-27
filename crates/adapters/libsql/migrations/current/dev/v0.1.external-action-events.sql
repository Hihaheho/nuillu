CREATE TABLE IF NOT EXISTS external_action_events (
  id INTEGER PRIMARY KEY,
  server_session_id TEXT NOT NULL,
  invocation_id TEXT NOT NULL,
  invoked_by_module TEXT NOT NULL,
  invoked_by_replica INTEGER NOT NULL,
  action_id TEXT NOT NULL,
  arguments_json TEXT NOT NULL,
  status TEXT NOT NULL,
  accepted INTEGER,
  message TEXT,
  requested_at_ms INTEGER NOT NULL,
  completed_at_ms INTEGER,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS external_action_events_recent_idx
ON external_action_events(id DESC);

CREATE INDEX IF NOT EXISTS external_action_events_requested_idx
ON external_action_events(requested_at_ms, id);

CREATE INDEX IF NOT EXISTS external_action_events_invocation_idx
ON external_action_events(server_session_id, invocation_id);
