CREATE TABLE IF NOT EXISTS allocation_snapshots (
  owner_module TEXT NOT NULL,
  owner_replica INTEGER NOT NULL,
  snapshot_json TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  PRIMARY KEY(owner_module, owner_replica)
);
