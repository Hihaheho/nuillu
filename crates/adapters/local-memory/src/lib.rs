//! Local memory-directory adapter.
//!
//! This adapter exposes one backend through both memory write/read and
//! memory text search ports. Memory content is stored only as
//! `<memory-index>.txt` files under the constructor-provided directory.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use nuillu_module::ports::{
    FileSearchHit, FileSearchProvider, FileSearchQuery, IndexedMemory, MemoryQuery, MemoryRecord,
    MemoryStore, NewMemory, PortError,
};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};
use regex::{Regex, RegexBuilder};

#[derive(Debug)]
pub struct LocalMemoryAdapter {
    dir: PathBuf,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    next_id: u64,
    ranks: HashMap<MemoryIndex, MemoryRank>,
}

impl LocalMemoryAdapter {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, PortError> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(io_error)?;
        let next_id = next_id(&dir)?;
        Ok(Self {
            dir,
            state: Mutex::new(State {
                next_id,
                ranks: HashMap::new(),
            }),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_for(&self, index: &MemoryIndex) -> Result<PathBuf, PortError> {
        ensure_plain_index(index)?;
        Ok(self.dir.join(format!("{}.txt", index.as_str())))
    }

    fn read_record(&self, index: MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        let path = self.path_for(&index)?;
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error(error)),
        };
        let rank = self
            .state
            .lock()
            .map_err(|_| PortError::Backend("local memory state lock poisoned".into()))?
            .ranks
            .get(&index)
            .copied()
            .unwrap_or(MemoryRank::ShortTerm);
        Ok(Some(MemoryRecord {
            index,
            content: MemoryContent::new(content),
            rank,
        }))
    }
}

#[async_trait(?Send)]
impl MemoryStore for LocalMemoryAdapter {
    async fn insert(&self, mem: NewMemory) -> Result<MemoryIndex, PortError> {
        let index = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| PortError::Backend("local memory state lock poisoned".into()))?;
            let index = MemoryIndex::new(state.next_id.to_string());
            state.next_id += 1;
            state.ranks.insert(index.clone(), mem.rank);
            index
        };
        fs::write(self.path_for(&index)?, mem.content.as_str()).map_err(io_error)?;
        Ok(index)
    }

    async fn put(&self, mem: IndexedMemory) -> Result<(), PortError> {
        fs::write(self.path_for(&mem.index)?, mem.content.as_str()).map_err(io_error)?;
        self.state
            .lock()
            .map_err(|_| PortError::Backend("local memory state lock poisoned".into()))?
            .ranks
            .insert(mem.index, mem.rank);
        Ok(())
    }

    async fn compact(
        &self,
        mem: NewMemory,
        sources: &[MemoryIndex],
    ) -> Result<MemoryIndex, PortError> {
        let index = self.insert(mem).await?;
        for source in sources {
            self.delete(source).await?;
        }
        Ok(index)
    }

    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        sources: &[MemoryIndex],
    ) -> Result<(), PortError> {
        self.put(mem).await?;
        for source in sources {
            self.delete(source).await?;
        }
        Ok(())
    }

    async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        self.read_record(index.clone())
    }

    async fn search(&self, _q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        Err(PortError::Backend(
            "local memory adapter does not implement vector search".into(),
        ))
    }

    async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
        let path = self.path_for(index)?;
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io_error(error)),
        }
        self.state
            .lock()
            .map_err(|_| PortError::Backend("local memory state lock poisoned".into()))?
            .ranks
            .remove(index);
        Ok(())
    }
}

#[async_trait(?Send)]
impl FileSearchProvider for LocalMemoryAdapter {
    async fn search(&self, query: &FileSearchQuery) -> Result<Vec<FileSearchHit>, PortError> {
        if query.pattern.is_empty() || query.max_matches == 0 {
            return Ok(Vec::new());
        }

        let matcher = Matcher::new(query)?;
        let mut hits = Vec::new();
        let mut files = memory_files(&self.dir)?;
        files.sort_by(|a, b| a.index.as_str().cmp(b.index.as_str()));

        for file in files {
            let text = fs::read_to_string(&file.path).map_err(io_error)?;
            collect_hits(&file.index, &text, query, &matcher, &mut hits);
            if hits.len() >= query.max_matches {
                hits.truncate(query.max_matches);
                break;
            }
        }

        Ok(hits)
    }
}

#[derive(Debug)]
struct MemoryFile {
    index: MemoryIndex,
    path: PathBuf,
}

enum Matcher {
    Literal {
        needle: String,
        case_sensitive: bool,
        invert_match: bool,
    },
    Regex {
        regex: Regex,
        invert_match: bool,
    },
}

impl Matcher {
    fn new(query: &FileSearchQuery) -> Result<Self, PortError> {
        if query.regex {
            let regex = RegexBuilder::new(&query.pattern)
                .case_insensitive(!query.case_sensitive)
                .build()
                .map_err(|error| PortError::Backend(error.to_string()))?;
            Ok(Self::Regex {
                regex,
                invert_match: query.invert_match,
            })
        } else {
            Ok(Self::Literal {
                needle: if query.case_sensitive {
                    query.pattern.clone()
                } else {
                    query.pattern.to_lowercase()
                },
                case_sensitive: query.case_sensitive,
                invert_match: query.invert_match,
            })
        }
    }

    fn is_match(&self, line: &str) -> bool {
        match self {
            Self::Literal {
                needle,
                case_sensitive,
                invert_match,
            } => {
                let haystack = if *case_sensitive {
                    line.to_owned()
                } else {
                    line.to_lowercase()
                };
                haystack.contains(needle) ^ invert_match
            }
            Self::Regex {
                regex,
                invert_match,
            } => regex.is_match(line) ^ invert_match,
        }
    }
}

fn memory_files(dir: &Path) -> Result<Vec<MemoryFile>, PortError> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("txt") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let index = MemoryIndex::new(stem.to_owned());
        ensure_plain_index(&index)?;
        files.push(MemoryFile { index, path });
    }
    Ok(files)
}

fn collect_hits(
    index: &MemoryIndex,
    text: &str,
    query: &FileSearchQuery,
    matcher: &Matcher,
    hits: &mut Vec<FileSearchHit>,
) {
    let lines = text.lines().collect::<Vec<_>>();
    for (line_index, line) in lines.iter().enumerate() {
        if hits.len() >= query.max_matches {
            return;
        }
        if !matcher.is_match(line) {
            continue;
        }

        let start = line_index.saturating_sub(query.context);
        let end = (line_index + query.context + 1).min(lines.len());
        hits.push(FileSearchHit {
            path: index.as_str().to_owned(),
            line: line_index + 1,
            snippet: lines[start..end].join("\n"),
        });
    }
}

fn next_id(dir: &Path) -> Result<u64, PortError> {
    let mut max = 0;
    for file in memory_files(dir)? {
        if let Ok(id) = file.index.as_str().parse::<u64>() {
            max = max.max(id + 1);
        }
    }
    Ok(max)
}

fn ensure_plain_index(index: &MemoryIndex) -> Result<(), PortError> {
    let id = index.as_str();
    if id.is_empty() || id.contains('/') || id.contains('\\') || id == "." || id == ".." {
        return Err(PortError::Backend(
            "memory index is not a plain file stem".into(),
        ));
    }
    Ok(())
}

fn io_error(error: std::io::Error) -> PortError {
    PortError::Backend(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    #[tokio::test]
    async fn store_writes_index_txt_and_search_reads_only_memory_files() {
        let dir = test_dir();
        let adapter = LocalMemoryAdapter::new(&dir).unwrap();
        let index = adapter
            .insert(NewMemory {
                content: MemoryContent::new("one\ntwo rust\nthree"),
                rank: MemoryRank::LongTerm,
            })
            .await
            .unwrap();
        fs::write(dir.join("not-memory.md"), "rust outside memory").unwrap();
        fs::create_dir_all(dir.join("nested")).unwrap();
        fs::write(dir.join("nested").join("2.txt"), "rust outside memory").unwrap();

        let hits = FileSearchProvider::search(
            &adapter,
            &FileSearchQuery {
                pattern: "RUST".into(),
                regex: false,
                invert_match: false,
                case_sensitive: false,
                context: 1,
                max_matches: 10,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            fs::read_to_string(dir.join(format!("{index}.txt"))).unwrap(),
            "one\ntwo rust\nthree"
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, index.as_str());
        assert_eq!(hits[0].line, 2);
        assert_eq!(hits[0].snippet, "one\ntwo rust\nthree");
    }

    #[tokio::test]
    async fn get_and_delete_use_same_memory_directory() {
        let dir = test_dir();
        let adapter = LocalMemoryAdapter::new(&dir).unwrap();
        let index = adapter
            .insert(NewMemory {
                content: MemoryContent::new("remember this"),
                rank: MemoryRank::Permanent,
            })
            .await
            .unwrap();

        let record = adapter.get(&index).await.unwrap().unwrap();
        assert_eq!(record.content.as_str(), "remember this");
        assert_eq!(record.rank, MemoryRank::Permanent);

        adapter.delete(&index).await.unwrap();
        assert!(adapter.get(&index).await.unwrap().is_none());
    }

    fn test_dir() -> PathBuf {
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("lutum-local-memory-adapter-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
