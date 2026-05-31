CREATE TABLE IF NOT EXISTS llm_sessions (
  owner_module TEXT NOT NULL,
  owner_replica INTEGER NOT NULL,
  session_key TEXT NOT NULL,
  snapshot_json TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  PRIMARY KEY(owner_module, owner_replica, session_key)
);

CREATE TABLE IF NOT EXISTS llm_transcript_turns (
  id INTEGER PRIMARY KEY,
  server_session_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  owner TEXT NOT NULL,
  owner_module TEXT NOT NULL,
  owner_replica INTEGER NOT NULL,
  tier TEXT NOT NULL,
  source TEXT NOT NULL,
  operation TEXT NOT NULL,
  started_at_ms INTEGER NOT NULL,
  completed_at_ms INTEGER NOT NULL,
  trace_json TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS llm_transcript_turns_recent_idx
ON llm_transcript_turns(id DESC);

CREATE INDEX IF NOT EXISTS llm_transcript_turns_owner_idx
ON llm_transcript_turns(owner_module, owner_replica, id DESC);
