use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use chrono::{DateTime, Datelike as _, FixedOffset, NaiveDate, TimeZone as _, Utc};
use eure::{FromEure, value::Text};
use nuillu_memory::{MemoryCapabilities, MemoryConcept, MemoryKind, MemoryTag, NewMemory};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};

const MEMORY_SEED_DIR: &str = "memory-seeds";
const DEFAULT_TRANSIENT_MEMORY_DECAY_SECS: i64 = 86_400;
const DURABLE_MEMORY_DECAY_SECS: i64 = 0;

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct MemorySeedFile {
    #[eure(default)]
    memories: Vec<MemorySeedEntry>,
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct MemorySeedEntry {
    index: String,
    rank: MemorySeedRank,
    #[eure(default = "default_memory_kind")]
    kind: MemorySeedKind,
    #[eure(default)]
    occurred_at: Option<String>,
    #[eure(default)]
    decay_secs: Option<i64>,
    #[eure(default)]
    concepts: Vec<String>,
    #[eure(default)]
    tags: Vec<String>,
    content: Text,
}

#[derive(Debug, Clone)]
struct ResolvedMemorySeed {
    index: MemoryIndex,
    memory: NewMemory,
    decay_secs: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
enum MemorySeedRank {
    ShortTerm,
    MidTerm,
    LongTerm,
    Permanent,
    Identity,
}

impl From<MemorySeedRank> for MemoryRank {
    fn from(rank: MemorySeedRank) -> Self {
        match rank {
            MemorySeedRank::ShortTerm => Self::ShortTerm,
            MemorySeedRank::MidTerm => Self::MidTerm,
            MemorySeedRank::LongTerm => Self::LongTerm,
            MemorySeedRank::Permanent => Self::Permanent,
            MemorySeedRank::Identity => Self::Identity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
enum MemorySeedKind {
    Episode,
    Statement,
    Reflection,
    Hypothesis,
    Dream,
    Procedure,
    Plan,
}

impl From<MemorySeedKind> for MemoryKind {
    fn from(kind: MemorySeedKind) -> Self {
        match kind {
            MemorySeedKind::Episode => Self::Episode,
            MemorySeedKind::Statement => Self::Statement,
            MemorySeedKind::Reflection => Self::Reflection,
            MemorySeedKind::Hypothesis => Self::Hypothesis,
            MemorySeedKind::Dream => Self::Dream,
            MemorySeedKind::Procedure => Self::Procedure,
            MemorySeedKind::Plan => Self::Plan,
        }
    }
}

fn default_memory_kind() -> MemorySeedKind {
    MemorySeedKind::Statement
}

pub(super) async fn seed_memory_from_state_dir(
    state_dir: &Path,
    memory_caps: &MemoryCapabilities,
) -> anyhow::Result<usize> {
    seed_memory_from_dir(&state_dir.join(MEMORY_SEED_DIR), memory_caps).await
}

async fn seed_memory_from_dir(
    seed_dir: &Path,
    memory_caps: &MemoryCapabilities,
) -> anyhow::Result<usize> {
    if !seed_dir.exists() {
        return Ok(0);
    }
    if !seed_dir.is_dir() {
        anyhow::bail!(
            "memory seed path is not a directory: {}",
            seed_dir.display()
        );
    }

    let files = discover_seed_files(seed_dir)
        .with_context(|| format!("discover memory seed files under {}", seed_dir.display()))?;
    let seeds = parse_memory_seed_files(&files)?;
    let writer = memory_caps.writer();
    let mut count = 0;
    for seed in seeds {
        let index = seed.index.clone();
        writer
            .put_seeded_entry(seed.index, seed.memory, seed.decay_secs)
            .await
            .with_context(|| format!("seed memory {}", index.as_str()))?;
        count += 1;
    }
    Ok(count)
}

fn parse_memory_seed_files(files: &[PathBuf]) -> anyhow::Result<Vec<ResolvedMemorySeed>> {
    let mut indexes = BTreeMap::<String, PathBuf>::new();
    let mut seeds = Vec::new();
    for path in files {
        for seed in parse_memory_seed_file(path)? {
            let index = seed.index.as_str().to_owned();
            if let Some(previous_path) = indexes.insert(index.clone(), path.clone()) {
                anyhow::bail!(
                    "duplicate memory seed index {index:?} in {} and {}",
                    previous_path.display(),
                    path.display()
                );
            }
            seeds.push(seed);
        }
    }
    Ok(seeds)
}

fn parse_memory_seed_file(path: &Path) -> anyhow::Result<Vec<ResolvedMemorySeed>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("read memory seed file {}", path.display()))?;
    parse_memory_seed_content(&content, path)
}

fn parse_memory_seed_content(
    content: &str,
    path: &Path,
) -> anyhow::Result<Vec<ResolvedMemorySeed>> {
    let file: MemorySeedFile =
        eure::parse_content(content, path.to_path_buf()).map_err(|message| {
            anyhow::anyhow!(
                "failed to parse memory seed file {}: {message}",
                path.display()
            )
        })?;
    file.memories
        .into_iter()
        .enumerate()
        .map(|(index, memory)| resolve_memory_seed_entry(path, index, memory))
        .collect()
}

fn resolve_memory_seed_entry(
    path: &Path,
    entry_index: usize,
    seed: MemorySeedEntry,
) -> anyhow::Result<ResolvedMemorySeed> {
    let index = seed.index.trim().to_owned();
    if index.is_empty() {
        anyhow::bail!(
            "{} memories[{entry_index}].index must not be empty",
            path.display()
        );
    }

    let content = seed.content.content.trim();
    if content.is_empty() {
        anyhow::bail!(
            "{} memories[{entry_index}].content must not be empty",
            path.display()
        );
    }

    let rank = MemoryRank::from(seed.rank);
    let occurred_at = seed
        .occurred_at
        .as_deref()
        .map(parse_memory_seed_datetime)
        .transpose()
        .with_context(|| {
            format!(
                "{} memories[{entry_index}].occurred-at is invalid",
                path.display()
            )
        })?;
    let decay_secs = seed.decay_secs.unwrap_or_else(|| default_decay_secs(rank));
    if decay_secs < 0 {
        anyhow::bail!(
            "{} memories[{entry_index}].decay-secs must not be negative",
            path.display()
        );
    }

    let concepts = seed
        .concepts
        .into_iter()
        .map(|label| non_empty_label(path, entry_index, "concepts", label))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .map(MemoryConcept::new)
        .collect();
    let tags = seed
        .tags
        .into_iter()
        .map(|label| non_empty_label(path, entry_index, "tags", label))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .map(MemoryTag::operational)
        .collect();

    Ok(ResolvedMemorySeed {
        index: MemoryIndex::new(index),
        memory: NewMemory {
            content: MemoryContent::new(content),
            rank,
            occurred_at,
            kind: MemoryKind::from(seed.kind),
            concepts,
            tags,
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        },
        decay_secs,
    })
}

fn non_empty_label(
    path: &Path,
    entry_index: usize,
    field: &str,
    label: String,
) -> anyhow::Result<String> {
    let label = label.trim().to_owned();
    if label.is_empty() {
        anyhow::bail!(
            "{} memories[{entry_index}].{field} must not contain empty labels",
            path.display()
        );
    }
    Ok(label)
}

fn default_decay_secs(rank: MemoryRank) -> i64 {
    match rank {
        MemoryRank::Permanent | MemoryRank::Identity => DURABLE_MEMORY_DECAY_SECS,
        MemoryRank::ShortTerm | MemoryRank::MidTerm | MemoryRank::LongTerm => {
            DEFAULT_TRANSIENT_MEMORY_DECAY_SECS
        }
    }
}

fn parse_memory_seed_datetime(value: &str) -> anyhow::Result<DateTime<Utc>> {
    let value = value.trim();
    if let Ok(datetime) = DateTime::parse_from_rfc3339(value) {
        return Ok(datetime.with_timezone(&Utc));
    }

    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .with_context(|| "memory datetime must be RFC3339 datetime or YYYY-MM-DD")?;
    let offset = FixedOffset::east_opt(0).expect("zero offset is valid");
    offset
        .with_ymd_and_hms(date.year(), date.month(), date.day(), 0, 0, 0)
        .single()
        .ok_or_else(|| anyhow::anyhow!("memory datetime date is not representable: {value}"))
        .map(|datetime| datetime.with_timezone(&Utc))
}

fn discover_seed_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_seed_files(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_seed_files(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_seed_files(&path, files)?;
        } else if file_type.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension == "eure")
        {
            files.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

    use async_trait::async_trait;
    use chrono::TimeZone as _;
    use nuillu_blackboard::Blackboard;
    use nuillu_memory::{
        IndexedMemory, LinkedMemoryQuery, LinkedMemoryRecord, MemoryLink, MemoryLinkRelation,
        MemoryQuery, MemoryRecord, MemoryStore, NewMemoryLink,
    };
    use nuillu_module::ports::{Clock, PortError};
    use uuid::Uuid;

    use super::*;

    #[derive(Debug)]
    struct FixedClock(DateTime<Utc>);

    #[async_trait(?Send)]
    impl Clock for FixedClock {
        fn now(&self) -> DateTime<Utc> {
            self.0
        }

        async fn sleep_until(&self, _deadline: DateTime<Utc>) {}
    }

    #[derive(Clone, Default)]
    struct MemorySeedTestStore {
        records: Rc<RefCell<BTreeMap<String, MemoryRecord>>>,
    }

    #[async_trait(?Send)]
    impl MemoryStore for MemorySeedTestStore {
        async fn insert(
            &self,
            mem: NewMemory,
            stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            let index = MemoryIndex::new(Uuid::now_v7().to_string());
            let record = MemoryRecord {
                index: index.clone(),
                content: mem.content,
                rank: mem.rank,
                occurred_at: mem.occurred_at,
                stored_at,
                kind: mem.kind,
                concepts: mem.concepts,
                tags: mem.tags,
                affect_arousal: mem.affect_arousal,
                valence: mem.valence,
                emotion: mem.emotion,
            };
            self.records
                .borrow_mut()
                .insert(index.as_str().to_owned(), record.clone());
            Ok(record)
        }

        async fn put(&self, mem: IndexedMemory) -> Result<MemoryRecord, PortError> {
            let record = MemoryRecord {
                index: mem.index,
                content: mem.content,
                rank: mem.rank,
                occurred_at: mem.occurred_at,
                stored_at: mem.stored_at,
                kind: mem.kind,
                concepts: mem.concepts,
                tags: mem.tags,
                affect_arousal: mem.affect_arousal,
                valence: mem.valence,
                emotion: mem.emotion,
            };
            self.records
                .borrow_mut()
                .insert(record.index.as_str().to_owned(), record.clone());
            Ok(record)
        }

        async fn compact(
            &self,
            mem: NewMemory,
            _sources: &[MemoryIndex],
            stored_at: DateTime<Utc>,
        ) -> Result<MemoryRecord, PortError> {
            self.insert(mem, stored_at).await
        }

        async fn put_compacted(
            &self,
            mem: IndexedMemory,
            _sources: &[MemoryIndex],
        ) -> Result<MemoryRecord, PortError> {
            self.put(mem).await
        }

        async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
            Ok(self.records.borrow().get(index.as_str()).cloned())
        }

        async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(self
                .records
                .borrow()
                .values()
                .filter(|record| record.rank == rank)
                .cloned()
                .collect())
        }

        async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
            Ok(self.records.borrow().values().cloned().collect())
        }

        async fn linked(
            &self,
            _q: &LinkedMemoryQuery,
        ) -> Result<Vec<LinkedMemoryRecord>, PortError> {
            Ok(Vec::new())
        }

        async fn upsert_link(
            &self,
            link: NewMemoryLink,
            updated_at: DateTime<Utc>,
        ) -> Result<MemoryLink, PortError> {
            Ok(MemoryLink {
                from_memory: link.from_memory,
                to_memory: link.to_memory,
                relation: MemoryLinkRelation::Related,
                freeform_relation: None,
                strength: 1.0,
                confidence: 1.0,
                updated_at,
            })
        }

        async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
            self.records.borrow_mut().remove(index.as_str());
            Ok(())
        }
    }

    fn parse_seed(content: &str) -> anyhow::Result<Vec<ResolvedMemorySeed>> {
        parse_memory_seed_content(content, Path::new("seed.eure"))
    }

    #[test]
    fn parses_valid_memory_seed_with_defaults() {
        let seeds = parse_seed(
            r#"
@ memories[] {
  index: nui-identity-name
  rank: identity
  occurred-at: 2026-06-07
  concepts = ["Nui", "Ryo"]
  tags = ["identity", "boot-seed"]
  content: Nui is an agent runtime used by Ryo.
}
"#,
        )
        .unwrap();

        assert_eq!(seeds.len(), 1);
        let seed = &seeds[0];
        assert_eq!(seed.index.as_str(), "nui-identity-name");
        assert_eq!(seed.memory.rank, MemoryRank::Identity);
        assert_eq!(seed.memory.kind, MemoryKind::Statement);
        assert_eq!(seed.decay_secs, 0);
        assert_eq!(
            seed.memory.occurred_at,
            Some(Utc.with_ymd_and_hms(2026, 6, 7, 0, 0, 0).unwrap())
        );
        assert_eq!(
            seed.memory.concepts,
            vec![MemoryConcept::new("Nui"), MemoryConcept::new("Ryo")]
        );
        assert_eq!(
            seed.memory.tags,
            vec![
                MemoryTag::operational("identity"),
                MemoryTag::operational("boot-seed")
            ]
        );
        assert_eq!(
            seed.memory.content.as_str(),
            "Nui is an agent runtime used by Ryo."
        );
    }

    #[test]
    fn parses_multiline_content_rfc3339_and_explicit_metadata() {
        let seeds = parse_seed(
            r#"
@ memories[] {
  index: ryo-prefers-status
  rank: permanent
  kind: reflection
  occurred-at: 2026-06-07T01:02:03+09:00
  decay-secs = 42
  content = ```
  Ryo prefers concise status updates.
  Keep implementation notes concrete.
  ```
}
"#,
        )
        .unwrap();

        let seed = &seeds[0];
        assert_eq!(seed.memory.kind, MemoryKind::Reflection);
        assert_eq!(seed.decay_secs, 42);
        assert_eq!(
            seed.memory.occurred_at,
            Some(
                DateTime::parse_from_rfc3339("2026-06-07T01:02:03+09:00")
                    .unwrap()
                    .with_timezone(&Utc)
            )
        );
        assert!(seed.memory.content.as_str().contains("concise status"));
        assert!(
            seed.memory
                .content
                .as_str()
                .contains("implementation notes")
        );
    }

    #[test]
    fn defaults_transient_memory_decay() {
        let seeds = parse_seed(
            r#"
@ memories[] {
  index: short-lived
  rank: short-term
  content: Temporary evidence.
}
"#,
        )
        .unwrap();

        assert_eq!(seeds[0].decay_secs, 86_400);
    }

    #[test]
    fn rejects_invalid_memory_seed_values() {
        let empty = parse_seed(
            r#"
@ memories[] {
  index: empty-content
  rank: identity
  content = ""
}
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(empty.contains("content must not be empty"), "{empty}");

        let invalid_datetime = parse_seed(
            r#"
@ memories[] {
  index: invalid-datetime
  rank: identity
  occurred-at: not-a-date
  content: Content.
}
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(
            invalid_datetime.contains("occurred-at is invalid"),
            "{invalid_datetime}"
        );
    }

    #[test]
    fn rejects_duplicate_indexes_across_seed_files() {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join(format!("memory-seed-{}", Uuid::now_v7()));
        fs::create_dir_all(&root).unwrap();
        let first = root.join("first.eure");
        let second = root.join("nested").join("second.eure");
        fs::create_dir_all(second.parent().unwrap()).unwrap();
        fs::write(
            &first,
            r#"
@ memories[] {
  index: duplicate
  rank: identity
  content: First.
}
"#,
        )
        .unwrap();
        fs::write(
            &second,
            r#"
@ memories[] {
  index: duplicate
  rank: permanent
  content: Second.
}
"#,
        )
        .unwrap();

        let files = discover_seed_files(&root).unwrap();
        let error = parse_memory_seed_files(&files).unwrap_err().to_string();

        assert!(error.contains("duplicate memory seed index"), "{error}");
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn seed_dir_loads_before_identity_bootstrap() {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join(format!("memory-seed-bootstrap-{}", Uuid::now_v7()));
        let seed_dir = root.join(MEMORY_SEED_DIR);
        fs::create_dir_all(&seed_dir).unwrap();
        fs::write(
            seed_dir.join("identity.eure"),
            r#"
@ memories[] {
  index: identity-seed
  rank: identity
  content: Nui remembers this identity seed.
}

@ memories[] {
  index: permanent-seed
  rank: permanent
  content: Nui remembers this permanent seed.
}
"#,
        )
        .unwrap();
        let blackboard = Blackboard::new();
        let store = MemorySeedTestStore::default();
        let memory_caps = MemoryCapabilities::new(
            blackboard.clone(),
            Rc::new(FixedClock(
                Utc.with_ymd_and_hms(2026, 6, 7, 0, 0, 0).unwrap(),
            )),
            Rc::new(store.clone()),
            Vec::new(),
        );

        let count = seed_memory_from_state_dir(&root, &memory_caps)
            .await
            .unwrap();
        memory_caps.bootstrap_identity_memories().await.unwrap();

        assert_eq!(count, 2);
        let identities = blackboard.read(|bb| bb.identity_memories().to_vec()).await;
        assert_eq!(identities.len(), 1);
        assert_eq!(identities[0].index.as_str(), "identity-seed");
        assert_eq!(
            identities[0].content.as_str(),
            "Nui remembers this identity seed."
        );
        assert_eq!(
            store.records.borrow().get("permanent-seed").unwrap().rank,
            MemoryRank::Permanent
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn missing_seed_dir_is_noop() {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join(format!("memory-seed-missing-{}", Uuid::now_v7()));
        fs::create_dir_all(&root).unwrap();
        let memory_caps = MemoryCapabilities::new(
            Blackboard::new(),
            Rc::new(FixedClock(
                Utc.with_ymd_and_hms(2026, 6, 7, 0, 0, 0).unwrap(),
            )),
            Rc::new(MemorySeedTestStore::default()),
            Vec::new(),
        );

        let count = seed_memory_from_state_dir(&root, &memory_caps)
            .await
            .unwrap();

        assert_eq!(count, 0);
        fs::remove_dir_all(root).unwrap();
    }
}
