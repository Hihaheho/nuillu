ALTER TABLE cognition_log_entries
ADD COLUMN origin_module TEXT;

ALTER TABLE cognition_log_entries
ADD COLUMN origin_replica INTEGER;

ALTER TABLE cognition_log_entries
ADD COLUMN origin_memo_index INTEGER;

UPDATE cognition_log_entries
SET origin_module = owner_module
WHERE origin_module IS NULL;

UPDATE cognition_log_entries
SET origin_replica = owner_replica
WHERE origin_replica IS NULL;
