CREATE TABLE IF NOT EXISTS memo_log_entries (
  id INTEGER PRIMARY KEY,
  owner_module TEXT NOT NULL,
  owner_replica INTEGER NOT NULL,
  memo_index INTEGER NOT NULL,
  written_at_ms INTEGER NOT NULL,
  cognitive INTEGER NOT NULL DEFAULT 0 CHECK(cognitive IN (0, 1)),
  content TEXT NOT NULL,
  payload_type TEXT,
  payload_json TEXT,
  created_at_ms INTEGER NOT NULL,
  UNIQUE(owner_module, owner_replica, memo_index),
  CHECK((payload_type IS NULL AND payload_json IS NULL) OR (payload_type IS NOT NULL AND payload_json IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS memo_log_entries_recent_idx
ON memo_log_entries(id DESC);

CREATE INDEX IF NOT EXISTS memo_log_entries_owner_index_idx
ON memo_log_entries(owner_module, owner_replica, memo_index DESC);
