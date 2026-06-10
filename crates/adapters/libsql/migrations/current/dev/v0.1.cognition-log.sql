CREATE TABLE IF NOT EXISTS cognition_log_entries (
  id INTEGER PRIMARY KEY,
  owner_module TEXT NOT NULL,
  owner_replica INTEGER NOT NULL,
  occurred_at_ms INTEGER NOT NULL,
  text TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS cognition_log_entries_recent_idx
ON cognition_log_entries(id DESC);

CREATE INDEX IF NOT EXISTS cognition_log_entries_owner_time_idx
ON cognition_log_entries(owner_module, owner_replica, occurred_at_ms, id);
