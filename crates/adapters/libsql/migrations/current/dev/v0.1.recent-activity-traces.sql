CREATE TABLE IF NOT EXISTS one_shot_sensory_inputs (
  id INTEGER PRIMARY KEY,
  server_session_id TEXT NOT NULL,
  modality TEXT NOT NULL,
  direction TEXT,
  content TEXT NOT NULL,
  observed_at_ms INTEGER NOT NULL,
  created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS one_shot_sensory_inputs_recent_idx
ON one_shot_sensory_inputs(id DESC);

CREATE INDEX IF NOT EXISTS one_shot_sensory_inputs_observed_idx
ON one_shot_sensory_inputs(observed_at_ms, id);

CREATE TABLE IF NOT EXISTS ambient_sensory_snapshots (
  id INTEGER PRIMARY KEY,
  server_session_id TEXT NOT NULL,
  entries_json TEXT NOT NULL,
  observed_at_ms INTEGER NOT NULL,
  created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ambient_sensory_snapshots_recent_idx
ON ambient_sensory_snapshots(id DESC);

CREATE INDEX IF NOT EXISTS ambient_sensory_snapshots_observed_idx
ON ambient_sensory_snapshots(observed_at_ms, id);

CREATE TABLE IF NOT EXISTS utterance_events (
  id INTEGER PRIMARY KEY,
  server_session_id TEXT NOT NULL,
  event_kind TEXT NOT NULL,
  sender_module TEXT NOT NULL,
  sender_replica INTEGER NOT NULL,
  target TEXT NOT NULL,
  generation_id INTEGER NOT NULL,
  sequence INTEGER NOT NULL,
  content TEXT NOT NULL,
  reason TEXT,
  occurred_at_ms INTEGER NOT NULL,
  created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS utterance_events_recent_idx
ON utterance_events(id DESC);

CREATE INDEX IF NOT EXISTS utterance_events_generation_idx
ON utterance_events(sender_module, sender_replica, target, generation_id, sequence, id);
