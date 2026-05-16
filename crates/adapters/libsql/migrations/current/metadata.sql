CREATE TABLE IF NOT EXISTS nuillu_schema_version (
  schema_family TEXT PRIMARY KEY,
  major INTEGER NOT NULL,
  minor INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  bridge_from_major INTEGER,
  bridge_from_minor INTEGER
);

CREATE TABLE IF NOT EXISTS nuillu_schema_dev_tasks (
  schema_family TEXT NOT NULL,
  major INTEGER NOT NULL,
  minor INTEGER NOT NULL,
  task_tag TEXT NOT NULL,
  checksum TEXT NOT NULL,
  applied_at_ms INTEGER NOT NULL,
  PRIMARY KEY(schema_family, major, minor, task_tag)
);
