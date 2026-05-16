CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY,
  memory_index TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL,
  kind INTEGER NOT NULL DEFAULT 1,
  rank INTEGER NOT NULL,
  occurred_at_ms INTEGER,
  stored_at_ms INTEGER NOT NULL,
  affect_arousal REAL NOT NULL DEFAULT 0.0,
  valence REAL NOT NULL DEFAULT 0.0,
  emotion TEXT NOT NULL DEFAULT '',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  source_ids TEXT,
  metadata_json TEXT,
  deleted_at_ms INTEGER
);

CREATE INDEX IF NOT EXISTS memories_rank_live_idx
ON memories(rank)
WHERE deleted_at_ms IS NULL;

CREATE TABLE IF NOT EXISTS memory_embedding_profiles (
  profile_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  table_name TEXT NOT NULL UNIQUE,
  created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memories_concepts (
  id INTEGER PRIMARY KEY,
  canonical_label TEXT NOT NULL,
  normalized_label TEXT NOT NULL UNIQUE,
  loose_type TEXT
);

CREATE TABLE IF NOT EXISTS memories_concept_aliases (
  concept_id INTEGER NOT NULL,
  alias TEXT NOT NULL,
  normalized_alias TEXT NOT NULL UNIQUE,
  FOREIGN KEY(concept_id) REFERENCES memories_concepts(id)
);

CREATE TABLE IF NOT EXISTS memories_memory_concepts (
  memory_id INTEGER NOT NULL,
  concept_id INTEGER NOT NULL,
  mention_text TEXT,
  confidence REAL NOT NULL,
  PRIMARY KEY(memory_id, concept_id, mention_text),
  FOREIGN KEY(memory_id) REFERENCES memories(id),
  FOREIGN KEY(concept_id) REFERENCES memories_concepts(id)
);

CREATE TABLE IF NOT EXISTS memories_tags (
  id INTEGER PRIMARY KEY,
  label TEXT NOT NULL,
  normalized_label TEXT NOT NULL,
  namespace TEXT NOT NULL,
  UNIQUE(namespace, normalized_label)
);

CREATE TABLE IF NOT EXISTS memories_memory_tags (
  memory_id INTEGER NOT NULL,
  tag_id INTEGER NOT NULL,
  confidence REAL NOT NULL,
  PRIMARY KEY(memory_id, tag_id),
  FOREIGN KEY(memory_id) REFERENCES memories(id),
  FOREIGN KEY(tag_id) REFERENCES memories_tags(id)
);

CREATE TABLE IF NOT EXISTS memories_links (
  id INTEGER PRIMARY KEY,
  from_memory_id INTEGER NOT NULL,
  to_memory_id INTEGER NOT NULL,
  relation INTEGER NOT NULL,
  freeform_relation TEXT,
  strength REAL NOT NULL,
  confidence REAL NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  FOREIGN KEY(from_memory_id) REFERENCES memories(id),
  FOREIGN KEY(to_memory_id) REFERENCES memories(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS memories_links_unique_idx
ON memories_links(
  from_memory_id,
  to_memory_id,
  relation,
  COALESCE(freeform_relation, '')
);

CREATE TABLE IF NOT EXISTS memories_search (
  memory_id INTEGER PRIMARY KEY,
  search_text TEXT NOT NULL,
  concept_text TEXT NOT NULL,
  tag_text TEXT NOT NULL,
  FOREIGN KEY(memory_id) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS policies (
  id INTEGER PRIMARY KEY,
  policy_index TEXT NOT NULL UNIQUE,
  trigger TEXT NOT NULL,
  behavior TEXT NOT NULL,
  rank INTEGER NOT NULL,
  expected_reward REAL NOT NULL,
  confidence REAL NOT NULL,
  value REAL NOT NULL,
  reward_tokens INTEGER NOT NULL,
  decay_remaining_secs INTEGER NOT NULL,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  last_reinforced_at_ms INTEGER,
  deleted_at_ms INTEGER
);

CREATE INDEX IF NOT EXISTS policies_rank_live_idx
ON policies(rank)
WHERE deleted_at_ms IS NULL;

CREATE TABLE IF NOT EXISTS policy_embedding_profiles (
  profile_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  table_name TEXT NOT NULL UNIQUE,
  created_at_ms INTEGER NOT NULL
);
