//! LanceDB-backed memory adapter.
//!
//! The public `nuillu-module` memory port deals only in content and query
//! text. This crate owns the adapter-local embedding boundary needed to
//! store and search vectors in LanceDB.

mod schema;

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::Table;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
pub use lutum_lancedb_core::LanceDbEmbedder;
use nuillu_module::ports::{
    IndexedMemory, MemoryQuery, MemoryRecord, MemoryStore, NewMemory, PortError,
};
use nuillu_types::{MemoryContent, MemoryIndex, MemoryRank};
use uuid::Uuid;

use crate::schema::{COL_CONTENT, COL_ID, COL_RANK, COL_VECTOR, memory_schema};

#[derive(Clone, Debug)]
pub struct LanceDbMemoryStoreConfig {
    pub uri: String,
    pub table_name: String,
    pub vector_dims: usize,
    pub create_if_missing: bool,
    pub create_indices: bool,
}

impl LanceDbMemoryStoreConfig {
    pub fn local(path: impl Into<PathBuf>, vector_dims: usize) -> Self {
        Self {
            uri: path.into().to_string_lossy().to_string(),
            table_name: "memories".to_string(),
            vector_dims,
            create_if_missing: true,
            create_indices: true,
        }
    }
}

#[derive(Clone)]
pub struct LanceDbMemoryStore {
    table: Table,
    schema: SchemaRef,
    vector_dims: usize,
    embedder: Arc<dyn LanceDbEmbedder>,
}

impl LanceDbMemoryStore {
    pub async fn connect(
        config: LanceDbMemoryStoreConfig,
        embedder: Arc<dyn LanceDbEmbedder>,
    ) -> Result<Self, PortError> {
        if config.vector_dims == 0 {
            return Err(PortError::InvalidInput(
                "LanceDB vector_dims must be greater than zero".into(),
            ));
        }
        let embedder_dims = embedder.dimensions();
        if embedder_dims != config.vector_dims {
            return Err(PortError::InvalidInput(format!(
                "LanceDB vector dimension mismatch: config={}, embedder={embedder_dims}",
                config.vector_dims
            )));
        }

        let schema = memory_schema(config.vector_dims as i32);
        let db = lancedb::connect(&config.uri)
            .execute()
            .await
            .map_err(map_lancedb_error)?;

        let table = match db.open_table(&config.table_name).execute().await {
            Ok(table) => table,
            Err(open_error) if config.create_if_missing => {
                tracing::info!(
                    table = %config.table_name,
                    error = %open_error,
                    "LanceDB table not found; creating table"
                );
                db.create_empty_table(&config.table_name, schema.clone())
                    .execute()
                    .await
                    .map_err(map_lancedb_error)?
            }
            Err(error) => return Err(map_lancedb_error(error)),
        };

        let store = Self {
            table,
            schema,
            vector_dims: config.vector_dims,
            embedder,
        };

        if config.create_indices {
            store.ensure_indices().await?;
        }

        Ok(store)
    }

    pub async fn ensure_indices(&self) -> Result<(), PortError> {
        self.create_index_if_needed(&[COL_VECTOR]).await?;
        self.create_index_if_needed(&[COL_RANK]).await?;
        Ok(())
    }

    async fn create_index_if_needed(&self, columns: &[&str]) -> Result<(), PortError> {
        match self
            .table
            .create_index(columns, Index::Auto)
            .execute()
            .await
        {
            Ok(()) => Ok(()),
            Err(error) if is_index_already_exists(&error) => Ok(()),
            Err(error) if is_empty_vector_index_unsupported(&error) => {
                tracing::warn!(
                    columns = ?columns,
                    error = %error,
                    "LanceDB index creation skipped; retry ensure_indices after writing data"
                );
                Ok(())
            }
            Err(error) => Err(map_lancedb_error(error)),
        }
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, PortError> {
        let embedding = self.embedder.embed(text).await?;
        self.validate_dims("embedding", embedding.len())?;
        Ok(embedding)
    }

    fn validate_dims(&self, label: &str, actual: usize) -> Result<(), PortError> {
        if actual != self.vector_dims {
            return Err(PortError::InvalidInput(format!(
                "{label} dimension mismatch: expected {}, got {actual}",
                self.vector_dims
            )));
        }
        Ok(())
    }

    fn new_memory_to_batch(
        &self,
        id: &str,
        mem: &NewMemory,
        embedding: &[f32],
        source_ids_json: Option<String>,
    ) -> Result<RecordBatch, PortError> {
        self.memory_to_batch(
            id,
            mem.content.as_str(),
            mem.rank,
            embedding,
            source_ids_json,
        )
    }

    fn indexed_memory_to_batch(
        &self,
        mem: &IndexedMemory,
        embedding: &[f32],
        source_ids_json: Option<String>,
    ) -> Result<RecordBatch, PortError> {
        self.memory_to_batch(
            mem.index.as_str(),
            mem.content.as_str(),
            mem.rank,
            embedding,
            source_ids_json,
        )
    }

    fn memory_to_batch(
        &self,
        id: &str,
        content: &str,
        rank: MemoryRank,
        embedding: &[f32],
        source_ids_json: Option<String>,
    ) -> Result<RecordBatch, PortError> {
        self.validate_dims("embedding", embedding.len())?;
        let now = now_ms();
        let vector_values = embedding.iter().copied().map(Some).collect::<Vec<_>>();
        let vector = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            [Some(vector_values)],
            self.vector_dims as i32,
        );

        RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id])),
                Arc::new(StringArray::from(vec![content])),
                Arc::new(Int32Array::from(vec![rank_to_i32(rank)])),
                Arc::new(vector),
                Arc::new(Int64Array::from(vec![now])),
                Arc::new(Int64Array::from(vec![now])),
                Arc::new(StringArray::from(vec![source_ids_json])),
                Arc::new(StringArray::from(vec![None::<String>])),
            ],
        )
        .map_err(|error| PortError::InvalidData(error.to_string()))
    }

    async fn delete_by_id_string(&self, id: &str) -> Result<(), PortError> {
        let predicate = format!("{COL_ID} = {}", sql_string_literal(id));
        self.table
            .delete(&predicate)
            .await
            .map(|_| ())
            .map_err(map_lancedb_error)
    }

    async fn delete_many_by_id(&self, ids: &[MemoryIndex]) -> Result<(), PortError> {
        if ids.is_empty() {
            return Ok(());
        }

        let joined = ids
            .iter()
            .map(|id| sql_string_literal(id.as_str()))
            .collect::<Vec<_>>()
            .join(", ");
        let predicate = format!("{COL_ID} IN ({joined})");
        self.table
            .delete(&predicate)
            .await
            .map(|_| ())
            .map_err(map_lancedb_error)
    }

    #[cfg(test)]
    async fn source_ids_json(&self, index: &MemoryIndex) -> Result<Option<String>, PortError> {
        let id = index.as_str();
        let predicate = format!("{COL_ID} = {}", sql_string_literal(id));
        let batches = self
            .table
            .query()
            .only_if(predicate)
            .limit(1)
            .execute()
            .await
            .map_err(map_lancedb_error)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|error| PortError::Backend(error.to_string()))?;

        let Some(batch) = batches.first() else {
            return Ok(None);
        };
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let source_ids = string_column(batch, schema::COL_SOURCE_IDS)?;
        if source_ids.is_null(0) {
            return Ok(None);
        }
        Ok(Some(source_ids.value(0).to_owned()))
    }
}

#[async_trait(?Send)]
impl MemoryStore for LanceDbMemoryStore {
    async fn insert(&self, mem: NewMemory) -> Result<MemoryIndex, PortError> {
        let id = Uuid::new_v4().to_string();
        let embedding = self.embed_text(mem.content.as_str()).await?;
        let batch = self.new_memory_to_batch(&id, &mem, &embedding, None)?;
        self.table
            .add(batch)
            .execute()
            .await
            .map_err(map_lancedb_error)?;
        Ok(MemoryIndex::new(id))
    }

    async fn put(&self, mem: IndexedMemory) -> Result<(), PortError> {
        let embedding = self.embed_text(mem.content.as_str()).await?;
        self.delete_by_id_string(mem.index.as_str()).await?;
        let batch = self.indexed_memory_to_batch(&mem, &embedding, None)?;
        self.table
            .add(batch)
            .execute()
            .await
            .map_err(map_lancedb_error)?;
        Ok(())
    }

    async fn compact(
        &self,
        mem: NewMemory,
        sources: &[MemoryIndex],
    ) -> Result<MemoryIndex, PortError> {
        let id = Uuid::new_v4().to_string();
        let source_ids_json = source_ids_json(sources)?;
        let embedding = self.embed_text(mem.content.as_str()).await?;
        let batch = self.new_memory_to_batch(&id, &mem, &embedding, Some(source_ids_json))?;

        self.table
            .add(batch)
            .execute()
            .await
            .map_err(map_lancedb_error)?;
        self.delete_many_by_id(sources).await?;
        Ok(MemoryIndex::new(id))
    }

    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        sources: &[MemoryIndex],
    ) -> Result<(), PortError> {
        let source_ids_json = source_ids_json(sources)?;
        let embedding = self.embed_text(mem.content.as_str()).await?;
        self.delete_by_id_string(mem.index.as_str()).await?;
        let batch = self.indexed_memory_to_batch(&mem, &embedding, Some(source_ids_json))?;

        self.table
            .add(batch)
            .execute()
            .await
            .map_err(map_lancedb_error)?;
        self.delete_many_by_id(sources).await?;
        Ok(())
    }

    async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        let predicate = format!("{COL_ID} = {}", sql_string_literal(index.as_str()));
        let batches = self
            .table
            .query()
            .only_if(predicate)
            .limit(1)
            .execute()
            .await
            .map_err(map_lancedb_error)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|error| PortError::Backend(error.to_string()))?;

        let mut records = records_from_batches(&batches)?;
        Ok(records.pop())
    }

    async fn search(&self, q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        if q.limit == 0 {
            return Ok(Vec::new());
        }
        let embedding = self.embed_text(&q.text).await?;
        let mut query = self
            .table
            .query()
            .nearest_to(embedding.as_slice())
            .map_err(map_lancedb_error)?
            .limit(q.limit);

        if let Some(rank) = q.filter_rank {
            query = query.only_if(format!("{COL_RANK} = {}", rank_to_i32(rank)));
        }

        let batches = query
            .execute()
            .await
            .map_err(map_lancedb_error)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|error| PortError::Backend(error.to_string()))?;

        records_from_batches(&batches)
    }

    async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
        self.delete_by_id_string(index.as_str()).await
    }
}

fn records_from_batches(batches: &[RecordBatch]) -> Result<Vec<MemoryRecord>, PortError> {
    let mut out = Vec::new();

    for batch in batches {
        let ids = string_column(batch, COL_ID)?;
        let contents = string_column(batch, COL_CONTENT)?;
        let ranks = int32_column(batch, COL_RANK)?;

        for row in 0..batch.num_rows() {
            out.push(MemoryRecord {
                index: MemoryIndex::new(ids.value(row).to_owned()),
                content: MemoryContent::new(contents.value(row).to_owned()),
                rank: rank_from_i32(ranks.value(row))?,
            });
        }
    }

    Ok(out)
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, PortError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| PortError::InvalidData(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| PortError::InvalidData(format!("column is not Utf8: {name}")))
}

fn int32_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int32Array, PortError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| PortError::InvalidData(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| PortError::InvalidData(format!("column is not Int32: {name}")))
}

#[allow(dead_code)]
fn vector_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a FixedSizeListArray, PortError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| PortError::InvalidData(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| PortError::InvalidData(format!("column is not FixedSizeList: {name}")))
}

#[allow(dead_code)]
fn vector_at(
    vectors: &FixedSizeListArray,
    row: usize,
    vector_dims: usize,
) -> Result<Vec<f32>, PortError> {
    let values = vectors
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| PortError::InvalidData("vector values are not Float32".into()))?;
    let start = row * vector_dims;
    let end = start + vector_dims;
    if end > values.len() {
        return Err(PortError::InvalidData(format!(
            "vector out of bounds: row={row}, dims={vector_dims}, values_len={}",
            values.len()
        )));
    }
    Ok((start..end).map(|i| values.value(i)).collect())
}

fn source_ids_json(ids: &[MemoryIndex]) -> Result<String, PortError> {
    let source_ids = ids
        .iter()
        .map(|id| id.as_str().to_owned())
        .collect::<Vec<_>>();
    serde_json::to_string(&source_ids).map_err(|error| PortError::Backend(error.to_string()))
}

fn rank_to_i32(rank: MemoryRank) -> i32 {
    match rank {
        MemoryRank::ShortTerm => 0,
        MemoryRank::MidTerm => 1,
        MemoryRank::LongTerm => 2,
        MemoryRank::Permanent => 3,
    }
}

fn rank_from_i32(value: i32) -> Result<MemoryRank, PortError> {
    match value {
        0 => Ok(MemoryRank::ShortTerm),
        1 => Ok(MemoryRank::MidTerm),
        2 => Ok(MemoryRank::LongTerm),
        3 => Ok(MemoryRank::Permanent),
        _ => Err(PortError::InvalidData(format!(
            "invalid memory rank: {value}"
        ))),
    }
}

fn sql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn map_lancedb_error(error: lancedb::Error) -> PortError {
    let message = error.to_string();
    let lower = message.to_ascii_lowercase();
    if lower.contains("not found") || lower.contains("does not exist") {
        PortError::NotFound(message)
    } else if lower.contains("invalid") {
        PortError::InvalidInput(message)
    } else if lower.contains("schema") {
        PortError::InvalidData(message)
    } else {
        PortError::Backend(message)
    }
}

fn is_index_already_exists(error: &lancedb::Error) -> bool {
    let lower = error.to_string().to_ascii_lowercase();
    lower.contains("already exists") || lower.contains("already has")
}

fn is_empty_vector_index_unsupported(error: &lancedb::Error) -> bool {
    let lower = error.to_string().to_ascii_lowercase();
    lower.contains("creating empty vector indices")
        || lower.contains("cannot be created without training")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug)]
    struct TestEmbedder {
        dims: usize,
    }

    impl TestEmbedder {
        fn new(dims: usize) -> Arc<Self> {
            Arc::new(Self { dims })
        }
    }

    #[async_trait(?Send)]
    impl LanceDbEmbedder for TestEmbedder {
        fn dimensions(&self) -> usize {
            self.dims
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError> {
            let mut out = vec![0.0; self.dims];
            for token in text.split_whitespace() {
                match token {
                    "alpha" => out[0] += 1.0,
                    "beta" if self.dims > 1 => out[1] += 1.0,
                    "gamma" if self.dims > 2 => out[2] += 1.0,
                    _ => {
                        let index = token.bytes().fold(0_usize, |acc, byte| {
                            acc.wrapping_mul(31).wrapping_add(byte as usize)
                        }) % self.dims;
                        out[index] += 0.25;
                    }
                }
            }
            Ok(out)
        }
    }

    #[derive(Debug)]
    struct WrongDimEmbedder;

    #[async_trait(?Send)]
    impl LanceDbEmbedder for WrongDimEmbedder {
        fn dimensions(&self) -> usize {
            3
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, PortError> {
            Ok(vec![1.0, 0.0])
        }
    }

    #[tokio::test]
    async fn connect_creates_missing_table() {
        let store = store_with_indices(false).await;
        let id = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();

        assert!(store.get(&id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn connect_with_default_indices_can_write() {
        let store = store_with_indices(true).await;
        let id = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();

        assert!(store.get(&id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn connect_missing_table_without_create_returns_not_found() {
        let dir = test_dir();
        let embedder = TestEmbedder::new(3);
        let mut config = LanceDbMemoryStoreConfig::local(dir, 3);
        config.create_if_missing = false;
        config.create_indices = false;

        let error = expect_connect_error(LanceDbMemoryStore::connect(config, embedder).await);

        assert!(matches!(
            error,
            PortError::NotFound(_) | PortError::Backend(_)
        ));
    }

    #[tokio::test]
    async fn connect_rejects_bad_dimensions() {
        let dir = test_dir();
        let embedder = TestEmbedder::new(3);
        let zero = LanceDbMemoryStoreConfig::local(&dir, 0);
        assert!(matches!(
            expect_connect_error(LanceDbMemoryStore::connect(zero, embedder.clone()).await),
            PortError::InvalidInput(_)
        ));

        let mismatch = LanceDbMemoryStoreConfig::local(dir, 2);
        assert!(matches!(
            expect_connect_error(LanceDbMemoryStore::connect(mismatch, embedder).await),
            PortError::InvalidInput(_)
        ));
    }

    #[tokio::test]
    async fn insert_then_get_roundtrips_content_and_rank() {
        let store = store_with_indices(false).await;
        let id = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha beta"),
                rank: MemoryRank::LongTerm,
            })
            .await
            .unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.index, id);
        assert_eq!(got.content.as_str(), "alpha beta");
        assert_eq!(got.rank, MemoryRank::LongTerm);
    }

    #[tokio::test]
    async fn get_unknown_returns_none() {
        let store = store_with_indices(false).await;
        let got = store.get(&MemoryIndex::new("missing")).await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn search_uses_nearest_limit_and_rank_filter() {
        let store = store_with_indices(false).await;
        let alpha = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();
        let beta = store
            .insert(NewMemory {
                content: MemoryContent::new("beta"),
                rank: MemoryRank::LongTerm,
            })
            .await
            .unwrap();
        store
            .insert(NewMemory {
                content: MemoryContent::new("gamma"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();

        let hits = store
            .search(&MemoryQuery {
                text: "alpha".into(),
                limit: 1,
                filter_rank: None,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, alpha);

        let filtered = store
            .search(&MemoryQuery {
                text: "alpha".into(),
                limit: 10,
                filter_rank: Some(MemoryRank::LongTerm),
            })
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].index, beta);
    }

    #[tokio::test]
    async fn search_returns_empty_for_zero_limit() {
        let store = store_with_indices(false).await;
        store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();

        let hits = store
            .search(&MemoryQuery {
                text: "alpha".into(),
                limit: 0,
                filter_rank: None,
            })
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn embedding_dimension_mismatch_is_rejected() {
        let dir = test_dir();
        let embedder = Arc::new(WrongDimEmbedder);
        let mut config = LanceDbMemoryStoreConfig::local(dir, 3);
        config.create_indices = false;
        let store = LanceDbMemoryStore::connect(config, embedder).await.unwrap();

        let error = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap_err();

        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn put_inserts_and_replaces_explicit_id() {
        let store = store_with_indices(false).await;
        let id = MemoryIndex::new("replica-1");
        store
            .put(IndexedMemory {
                index: id.clone(),
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();
        store
            .put(IndexedMemory {
                index: id.clone(),
                content: MemoryContent::new("beta"),
                rank: MemoryRank::Permanent,
            })
            .await
            .unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.content.as_str(), "beta");
        assert_eq!(got.rank, MemoryRank::Permanent);
    }

    #[tokio::test]
    async fn delete_removes_rows_and_is_idempotent() {
        let store = store_with_indices(false).await;
        let id = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();

        store.delete(&id).await.unwrap();
        store.delete(&id).await.unwrap();
        assert!(store.get(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn compact_inserts_merged_row_and_removes_sources() {
        let store = store_with_indices(false).await;
        let first = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();
        let second = store
            .insert(NewMemory {
                content: MemoryContent::new("beta"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();

        let merged = store
            .compact(
                NewMemory {
                    content: MemoryContent::new("alpha beta"),
                    rank: MemoryRank::LongTerm,
                },
                &[first.clone(), second.clone()],
            )
            .await
            .unwrap();

        assert!(store.get(&first).await.unwrap().is_none());
        assert!(store.get(&second).await.unwrap().is_none());
        let got = store.get(&merged).await.unwrap().unwrap();
        assert_eq!(got.content.as_str(), "alpha beta");
        assert_eq!(got.rank, MemoryRank::LongTerm);

        let source_ids = store.source_ids_json(&merged).await.unwrap().unwrap();
        let parsed: Vec<String> = serde_json::from_str(&source_ids).unwrap();
        assert_eq!(parsed, vec![first.to_string(), second.to_string()]);
    }

    #[tokio::test]
    async fn put_compacted_mirrors_replica_flow() {
        let store = store_with_indices(false).await;
        let first = MemoryIndex::new("first");
        let second = MemoryIndex::new("second");
        store
            .put(IndexedMemory {
                index: first.clone(),
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();
        store
            .put(IndexedMemory {
                index: second.clone(),
                content: MemoryContent::new("beta"),
                rank: MemoryRank::ShortTerm,
            })
            .await
            .unwrap();

        let merged = MemoryIndex::new("merged");
        store
            .put_compacted(
                IndexedMemory {
                    index: merged.clone(),
                    content: MemoryContent::new("alpha beta"),
                    rank: MemoryRank::LongTerm,
                },
                &[first.clone(), second.clone()],
            )
            .await
            .unwrap();

        assert!(store.get(&first).await.unwrap().is_none());
        assert!(store.get(&second).await.unwrap().is_none());
        let got = store.get(&merged).await.unwrap().unwrap();
        assert_eq!(got.content.as_str(), "alpha beta");

        let source_ids = store.source_ids_json(&merged).await.unwrap().unwrap();
        let parsed: Vec<String> = serde_json::from_str(&source_ids).unwrap();
        assert_eq!(parsed, vec![first.to_string(), second.to_string()]);
    }

    async fn store_with_indices(create_indices: bool) -> LanceDbMemoryStore {
        let dir = test_dir();
        let embedder = TestEmbedder::new(3);
        let mut config = LanceDbMemoryStoreConfig::local(dir, 3);
        config.create_indices = create_indices;
        LanceDbMemoryStore::connect(config, embedder).await.unwrap()
    }

    fn expect_connect_error(result: Result<LanceDbMemoryStore, PortError>) -> PortError {
        match result {
            Ok(_) => panic!("expected LanceDB connect to fail"),
            Err(error) => error,
        }
    }

    fn test_dir() -> PathBuf {
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("lutum-lancedb-adapter-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
