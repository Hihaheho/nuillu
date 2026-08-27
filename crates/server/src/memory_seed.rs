use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs, io,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use chrono::{DateTime, Datelike as _, FixedOffset, NaiveDate, TimeZone as _, Utc};
use eure::{FromEure, value::Text};
use nuillu_memory::{MemoryConcept, MemoryKind, MemoryNamespace, MemoryTag, NewMemory};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank, ScopeId, SubsystemId};

use crate::ports::{MemorySeedPort, MemorySeedSummary, MemorySeedTarget};

const MEMORY_SEED_DIR: &str = "memory-seeds";
const DEFAULT_TRANSIENT_MEMORY_DECAY_SECS: i64 = 86_400;
const DURABLE_MEMORY_DECAY_SECS: i64 = 0;

#[derive(Debug, Clone)]
pub struct FileMemorySeedPort {
    state_dir: PathBuf,
}

impl FileMemorySeedPort {
    pub fn new(state_dir: impl Into<PathBuf>) -> Self {
        Self {
            state_dir: state_dir.into(),
        }
    }
}

#[async_trait::async_trait(?Send)]
impl MemorySeedPort for FileMemorySeedPort {
    async fn seed(&self, targets: &[MemorySeedTarget]) -> anyhow::Result<MemorySeedSummary> {
        seed_memory_from_state_dir(&self.state_dir, targets).await
    }
}

#[derive(Debug, Clone, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct MemorySeedFile {
    #[eure(default)]
    scope_path: Option<String>,
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

#[derive(Debug, Clone)]
struct ParsedMemorySeedFile {
    path: PathBuf,
    scope_path: MemorySeedScopePath,
    memories: Vec<ResolvedMemorySeed>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
struct MemorySeedScopePath(Vec<SubsystemId>);

impl MemorySeedScopePath {
    fn parse(value: &str) -> anyhow::Result<Self> {
        if value == "/" {
            return Ok(Self::default());
        }
        let path = value
            .strip_prefix('/')
            .ok_or_else(|| anyhow::anyhow!("scope path must start with '/'"))?;
        if path.is_empty() || path.split('/').any(str::is_empty) {
            anyhow::bail!("scope path must not contain an empty segment");
        }
        path.split('/')
            .map(|segment| {
                SubsystemId::new(segment).map_err(|error| {
                    anyhow::anyhow!("invalid subsystem {segment:?} in scope path: {error}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(Self)
    }

    fn from_scope(scope: &ScopeId) -> Self {
        Self(
            scope
                .path()
                .iter()
                .map(|instance| instance.subsystem.clone())
                .collect(),
        )
    }
}

impl fmt::Display for MemorySeedScopePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("/");
        }
        for subsystem in &self.0 {
            write!(f, "/{subsystem}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct TargetedMemorySeed {
    target: usize,
    path: PathBuf,
    memory: ResolvedMemorySeed,
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
    targets: &[MemorySeedTarget],
) -> anyhow::Result<MemorySeedSummary> {
    seed_memory_from_dir(&state_dir.join(MEMORY_SEED_DIR), targets).await
}

async fn seed_memory_from_dir(
    seed_dir: &Path,
    targets: &[MemorySeedTarget],
) -> anyhow::Result<MemorySeedSummary> {
    if !seed_dir.exists() {
        return Ok(MemorySeedSummary::default());
    }
    if !seed_dir.is_dir() {
        anyhow::bail!(
            "memory seed path is not a directory: {}",
            seed_dir.display()
        );
    }

    let files = discover_seed_files(seed_dir)
        .with_context(|| format!("discover memory seed files under {}", seed_dir.display()))?;
    let files = parse_memory_seed_files(&files)?;
    let mut targets_by_scope = BTreeMap::new();
    let mut targets_by_path = BTreeMap::<MemorySeedScopePath, Vec<usize>>::new();
    for (index, target) in targets.iter().enumerate() {
        if targets_by_scope
            .insert(target.scope().clone(), index)
            .is_some()
        {
            anyhow::bail!("duplicate memory seed target for scope {}", target.scope());
        }
        targets_by_path
            .entry(MemorySeedScopePath::from_scope(target.scope()))
            .or_default()
            .push(index);
    }
    let mut persistent_indexes = BTreeMap::<String, PathBuf>::new();
    let mut seeds = Vec::new();
    for file in files {
        let Some(file_targets) = targets_by_path.get(&file.scope_path) else {
            anyhow::bail!(
                "memory seed file {} targets scope path {} which is not present in the subsystem topology",
                file.path.display(),
                file.scope_path
            );
        };
        for memory in file.memories {
            for &target in file_targets {
                let mut targeted_memory = memory.clone();
                targeted_memory.index = persistent_seed_index(
                    targets[target].namespace(),
                    targets[target].scope(),
                    &memory.index,
                );
                let index = targeted_memory.index.as_str().to_owned();
                if let Some(previous_path) = persistent_indexes.get(&index)
                    && previous_path != &file.path
                {
                    anyhow::bail!(
                        "duplicate resolved memory seed index {index:?} in {} and {}",
                        previous_path.display(),
                        file.path.display()
                    );
                }
                persistent_indexes.insert(index, file.path.clone());
                seeds.push(TargetedMemorySeed {
                    target,
                    path: file.path.clone(),
                    memory: targeted_memory,
                });
            }
        }
    }

    let mut seeded_targets = BTreeSet::new();
    for seed in seeds {
        let index = seed.memory.index.clone();
        targets[seed.target]
            .memory()
            .writer()
            .put_seeded_entry(
                seed.memory.index,
                seed.memory.memory,
                seed.memory.decay_secs,
            )
            .await
            .with_context(|| {
                format!(
                    "seed memory {} from {} into scope {}",
                    index.as_str(),
                    seed.path.display(),
                    targets[seed.target].scope()
                )
            })?;
        seeded_targets.insert(seed.target);
    }
    Ok(MemorySeedSummary {
        memories: persistent_indexes.len(),
        scopes: seeded_targets.len(),
    })
}

fn persistent_seed_index(
    namespace: &MemoryNamespace,
    scope: &ScopeId,
    index: &MemoryIndex,
) -> MemoryIndex {
    match namespace {
        MemoryNamespace::Global => index.clone(),
        MemoryNamespace::Local(_) => {
            MemoryIndex::new(format!("local-seed:{scope}:{}", index.as_str()))
        }
    }
}

fn parse_memory_seed_files(files: &[PathBuf]) -> anyhow::Result<Vec<ParsedMemorySeedFile>> {
    let mut indexes = BTreeMap::<(MemorySeedScopePath, String), PathBuf>::new();
    let mut parsed = Vec::new();
    for path in files {
        let file = parse_memory_seed_file(path)?;
        for seed in &file.memories {
            let index = seed.index.as_str().to_owned();
            if let Some(previous_path) =
                indexes.insert((file.scope_path.clone(), index.clone()), path.clone())
            {
                anyhow::bail!(
                    "duplicate memory seed index {index:?} for scope path {} in {} and {}",
                    file.scope_path,
                    previous_path.display(),
                    path.display()
                );
            }
        }
        parsed.push(file);
    }
    Ok(parsed)
}

fn parse_memory_seed_file(path: &Path) -> anyhow::Result<ParsedMemorySeedFile> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("read memory seed file {}", path.display()))?;
    parse_memory_seed_content(&content, path)
}

fn parse_memory_seed_content(content: &str, path: &Path) -> anyhow::Result<ParsedMemorySeedFile> {
    let file: MemorySeedFile =
        eure::parse_content(content, path.to_path_buf()).map_err(|message| {
            anyhow::anyhow!(
                "failed to parse memory seed file {}: {message}",
                path.display()
            )
        })?;
    let scope_path = file.scope_path.as_deref().unwrap_or("/");
    let scope_path = MemorySeedScopePath::parse(scope_path).with_context(|| {
        format!(
            "{} scope-path {scope_path:?} is not a canonical scope path",
            path.display()
        )
    })?;
    let memories = file
        .memories
        .into_iter()
        .enumerate()
        .map(|(index, memory)| resolve_memory_seed_entry(path, index, memory))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(ParsedMemorySeedFile {
        path: path.to_path_buf(),
        scope_path,
        memories,
    })
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
    use nuillu_blackboard::Blackboard;
    use nuillu_memory::{
        IndexedMemory, LinkedMemoryQuery, LinkedMemoryRecord, MemoryCapabilities, MemoryLink,
        MemoryLinkRelation, MemoryQuery, MemoryRecord, MemoryStore, NewMemoryLink,
    };
    use nuillu_module::ports::{Clock, PortError};
    use nuillu_types::{ReplicaIndex, SubsystemInstanceId};
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
        parse_memory_seed_content(content, Path::new("seed.eure")).map(|file| file.memories)
    }

    fn scope(value: &str) -> ScopeId {
        value
            .strip_prefix('/')
            .unwrap()
            .split('/')
            .filter(|segment| !segment.is_empty())
            .fold(ScopeId::root(), |scope, segment| {
                let (subsystem, replica) = segment.split_once('[').unwrap();
                let replica = replica.strip_suffix(']').unwrap().parse::<u8>().unwrap();
                scope.child(SubsystemInstanceId::new(
                    SubsystemId::new(subsystem).unwrap(),
                    ReplicaIndex::new(replica),
                ))
            })
    }

    fn memory_seed_target(
        memory: &MemoryCapabilities,
        blackboard: &Blackboard,
        scope: ScopeId,
        namespace: MemoryNamespace,
    ) -> MemorySeedTarget {
        let scoped_memory = memory
            .with_namespace(namespace.clone())
            .scoped(blackboard.scoped(scope.clone()));
        MemorySeedTarget::new(scope, namespace, scoped_memory)
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

        let invalid_scope = parse_memory_seed_content(
            r#"
scope-path: /arm[0]
"#,
            Path::new("seed.eure"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            invalid_scope.contains("is not a canonical scope path"),
            "{invalid_scope}"
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

        let targets = [MemorySeedTarget::new(
            ScopeId::root(),
            MemoryNamespace::Global,
            memory_caps.clone(),
        )];
        let seeded = seed_memory_from_state_dir(&root, &targets).await.unwrap();
        memory_caps.bootstrap_identity_memories().await.unwrap();

        assert_eq!(
            seeded,
            MemorySeedSummary {
                memories: 2,
                scopes: 1
            }
        );
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

        let targets = [MemorySeedTarget::new(
            ScopeId::root(),
            MemoryNamespace::Global,
            memory_caps,
        )];
        let seeded = seed_memory_from_state_dir(&root, &targets).await.unwrap();

        assert_eq!(seeded, MemorySeedSummary::default());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn local_scope_path_seeds_all_replicas_isolated_and_idempotent() {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join(format!("memory-seed-local-scopes-{}", Uuid::now_v7()));
        let seed_dir = root.join(MEMORY_SEED_DIR);
        fs::create_dir_all(&seed_dir).unwrap();
        fs::write(
            seed_dir.join("arm.eure"),
            r#"
scope-path: /arm

@ memories[] {
  index: shared-identity-key
  rank: identity
  content: Identity shared by every arm replica.
}
"#,
        )
        .unwrap();

        let blackboard = Blackboard::new();
        let store = MemorySeedTestStore::default();
        let memory = MemoryCapabilities::new(
            blackboard.clone(),
            Rc::new(FixedClock(
                Utc.with_ymd_and_hms(2026, 8, 27, 0, 0, 0).unwrap(),
            )),
            Rc::new(store.clone()),
            Vec::new(),
        );
        let arm_zero = scope("/arm[0]");
        let arm_one = scope("/arm[1]");
        let finger = scope("/arm[0]/finger[0]");
        let targets = vec![
            memory_seed_target(
                &memory,
                &blackboard,
                ScopeId::root(),
                MemoryNamespace::Global,
            ),
            memory_seed_target(
                &memory,
                &blackboard,
                arm_zero.clone(),
                MemoryNamespace::Local(arm_zero.clone()),
            ),
            memory_seed_target(
                &memory,
                &blackboard,
                arm_one.clone(),
                MemoryNamespace::Local(arm_one.clone()),
            ),
            memory_seed_target(
                &memory,
                &blackboard,
                finger.clone(),
                MemoryNamespace::Local(finger.clone()),
            ),
        ];

        let expected = MemorySeedSummary {
            memories: 2,
            scopes: 2,
        };
        assert_eq!(
            seed_memory_from_state_dir(&root, &targets).await.unwrap(),
            expected
        );
        assert_eq!(
            seed_memory_from_state_dir(&root, &targets).await.unwrap(),
            expected
        );
        for target in &targets {
            target.memory().bootstrap_identity_memories().await.unwrap();
        }

        assert_eq!(
            store.records.borrow().keys().cloned().collect::<Vec<_>>(),
            vec![
                "local-seed:/arm[0]:shared-identity-key".to_owned(),
                "local-seed:/arm[1]:shared-identity-key".to_owned(),
            ]
        );
        let root_identities = blackboard.read(|bb| bb.identity_memories().to_vec()).await;
        let arm_zero_identities = blackboard
            .scoped(arm_zero)
            .read(|bb| bb.identity_memories().to_vec())
            .await;
        let arm_one_identities = blackboard
            .scoped(arm_one)
            .read(|bb| bb.identity_memories().to_vec())
            .await;
        let finger_identities = blackboard
            .scoped(finger)
            .read(|bb| bb.identity_memories().to_vec())
            .await;
        assert!(root_identities.is_empty());
        assert_eq!(
            arm_zero_identities[0].index.as_str(),
            "local-seed:/arm[0]:shared-identity-key"
        );
        assert_eq!(
            arm_zero_identities[0].content.as_str(),
            "Identity shared by every arm replica."
        );
        assert_eq!(
            arm_one_identities[0].index.as_str(),
            "local-seed:/arm[1]:shared-identity-key"
        );
        assert_eq!(
            arm_one_identities[0].content.as_str(),
            "Identity shared by every arm replica."
        );
        assert!(finger_identities.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn global_scope_seed_remains_shared_but_targets_scoped_metadata() {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join(format!("memory-seed-global-scope-{}", Uuid::now_v7()));
        let seed_dir = root.join(MEMORY_SEED_DIR);
        fs::create_dir_all(&seed_dir).unwrap();
        fs::write(
            seed_dir.join("arm.eure"),
            r#"
scope-path: /arm

@ memories[] {
  index: shared-global-identity
  rank: identity
  content: Globally shared identity.
}
"#,
        )
        .unwrap();

        let blackboard = Blackboard::new();
        let store = MemorySeedTestStore::default();
        let memory = MemoryCapabilities::new(
            blackboard.clone(),
            Rc::new(FixedClock(
                Utc.with_ymd_and_hms(2026, 8, 27, 0, 0, 0).unwrap(),
            )),
            Rc::new(store.clone()),
            Vec::new(),
        );
        let arm_zero = scope("/arm[0]");
        let arm_one = scope("/arm[1]");
        let targets = vec![
            memory_seed_target(
                &memory,
                &blackboard,
                ScopeId::root(),
                MemoryNamespace::Global,
            ),
            memory_seed_target(
                &memory,
                &blackboard,
                arm_zero.clone(),
                MemoryNamespace::Global,
            ),
            memory_seed_target(
                &memory,
                &blackboard,
                arm_one.clone(),
                MemoryNamespace::Global,
            ),
        ];

        // Both arm replicas mirror one shared global memory, so the summary reports a single
        // seeded memory even though two scopes were written through.
        assert_eq!(
            seed_memory_from_state_dir(&root, &targets).await.unwrap(),
            MemorySeedSummary {
                memories: 1,
                scopes: 2
            }
        );
        for target in &targets {
            target.memory().bootstrap_identity_memories().await.unwrap();
        }

        assert_eq!(store.records.borrow().len(), 1);
        assert_eq!(
            blackboard
                .read(|bb| bb.identity_memories()[0].index.clone())
                .await
                .as_str(),
            "shared-global-identity"
        );
        assert_eq!(
            blackboard
                .scoped(arm_zero)
                .read(|bb| bb.identity_memories()[0].index.clone())
                .await
                .as_str(),
            "shared-global-identity"
        );
        assert_eq!(
            blackboard
                .scoped(arm_one)
                .read(|bb| bb.identity_memories()[0].index.clone())
                .await
                .as_str(),
            "shared-global-identity"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejects_seed_for_scope_outside_expanded_targets() {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join(format!("memory-seed-missing-scope-{}", Uuid::now_v7()));
        let seed_dir = root.join(MEMORY_SEED_DIR);
        fs::create_dir_all(&seed_dir).unwrap();
        fs::write(
            seed_dir.join("missing.eure"),
            r#"
scope-path: /arm

@ memories[] {
  index: unavailable
  rank: permanent
  content: This must not be seeded.
}
"#,
        )
        .unwrap();
        let blackboard = Blackboard::new();
        let store = MemorySeedTestStore::default();
        let memory = MemoryCapabilities::new(
            blackboard.clone(),
            Rc::new(FixedClock(
                Utc.with_ymd_and_hms(2026, 8, 27, 0, 0, 0).unwrap(),
            )),
            Rc::new(store.clone()),
            Vec::new(),
        );
        let targets = [memory_seed_target(
            &memory,
            &blackboard,
            ScopeId::root(),
            MemoryNamespace::Global,
        )];

        let error = seed_memory_from_state_dir(&root, &targets)
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("scope path /arm"), "{error}");
        assert!(store.records.borrow().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejects_indexes_that_collide_after_global_scope_resolution() {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join(format!("memory-seed-global-collision-{}", Uuid::now_v7()));
        let seed_dir = root.join(MEMORY_SEED_DIR);
        fs::create_dir_all(&seed_dir).unwrap();
        fs::write(
            seed_dir.join("root.eure"),
            r#"
@ memories[] {
  index: same-global-index
  rank: permanent
  content: Root declaration.
}
"#,
        )
        .unwrap();
        fs::write(
            seed_dir.join("arm.eure"),
            r#"
scope-path: /arm

@ memories[] {
  index: same-global-index
  rank: permanent
  content: Scoped declaration.
}
"#,
        )
        .unwrap();
        let blackboard = Blackboard::new();
        let store = MemorySeedTestStore::default();
        let memory = MemoryCapabilities::new(
            blackboard.clone(),
            Rc::new(FixedClock(
                Utc.with_ymd_and_hms(2026, 8, 27, 0, 0, 0).unwrap(),
            )),
            Rc::new(store.clone()),
            Vec::new(),
        );
        let arm = scope("/arm[0]");
        let targets = [
            memory_seed_target(
                &memory,
                &blackboard,
                ScopeId::root(),
                MemoryNamespace::Global,
            ),
            memory_seed_target(&memory, &blackboard, arm, MemoryNamespace::Global),
        ];

        let error = seed_memory_from_state_dir(&root, &targets)
            .await
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("duplicate resolved memory seed index"),
            "{error}"
        );
        assert!(store.records.borrow().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn sibling_scope_path_files_seed_distinct_identity_per_scope() {
        let root = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join(format!("memory-seed-sibling-scopes-{}", Uuid::now_v7()));
        let seed_dir = root.join(MEMORY_SEED_DIR);
        fs::create_dir_all(&seed_dir).unwrap();
        for (file, scope_path, index, content) in [
            ("root.eure", None, "whole-identity", "I am the whole body."),
            (
                "left.eure",
                Some("/left-leg"),
                "leg-identity",
                "I am the left leg.",
            ),
            (
                "center.eure",
                Some("/center-leg"),
                "leg-identity",
                "I am the center leg.",
            ),
            (
                "right.eure",
                Some("/right-leg"),
                "leg-identity",
                "I am the right leg.",
            ),
        ] {
            let scope_path = scope_path
                .map(|path| format!("scope-path: {path}\n"))
                .unwrap_or_default();
            fs::write(
                seed_dir.join(file),
                format!(
                    r#"
{scope_path}
@ memories[] {{
  index: {index}
  rank: identity
  content: {content}
}}
"#
                ),
            )
            .unwrap();
        }

        let blackboard = Blackboard::new();
        let memory = MemoryCapabilities::new(
            blackboard.clone(),
            Rc::new(FixedClock(
                Utc.with_ymd_and_hms(2026, 8, 27, 0, 0, 0).unwrap(),
            )),
            Rc::new(MemorySeedTestStore::default()),
            Vec::new(),
        );
        let mut targets = vec![memory_seed_target(
            &memory,
            &blackboard,
            ScopeId::root(),
            MemoryNamespace::Global,
        )];
        for leg in ["left-leg", "center-leg", "right-leg"] {
            let scope = scope(&format!("/{leg}[0]"));
            targets.push(memory_seed_target(
                &memory,
                &blackboard,
                scope.clone(),
                MemoryNamespace::Local(scope),
            ));
        }

        assert_eq!(
            seed_memory_from_state_dir(&root, &targets).await.unwrap(),
            MemorySeedSummary {
                memories: 4,
                scopes: 4
            }
        );
        for target in &targets {
            target.memory().bootstrap_identity_memories().await.unwrap();
        }

        let actual = targets
            .iter()
            .map(|target| {
                let blackboard = blackboard.scoped(target.scope().clone());
                async move {
                    blackboard
                        .read(|bb| {
                            (
                                target.scope().to_string(),
                                bb.identity_memories()[0].index.as_str().to_owned(),
                                bb.identity_memories()[0].content.as_str().to_owned(),
                            )
                        })
                        .await
                }
            })
            .collect::<Vec<_>>();
        let actual = futures::future::join_all(actual).await;
        assert_eq!(
            actual,
            vec![
                (
                    "/".to_owned(),
                    "whole-identity".to_owned(),
                    "I am the whole body.".to_owned(),
                ),
                (
                    "/left-leg[0]".to_owned(),
                    "local-seed:/left-leg[0]:leg-identity".to_owned(),
                    "I am the left leg.".to_owned(),
                ),
                (
                    "/center-leg[0]".to_owned(),
                    "local-seed:/center-leg[0]:leg-identity".to_owned(),
                    "I am the center leg.".to_owned(),
                ),
                (
                    "/right-leg[0]".to_owned(),
                    "local-seed:/right-leg[0]:leg-identity".to_owned(),
                    "I am the right leg.".to_owned(),
                ),
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }
}
