//! Local libSQL-backed memory adapter.
//!
//! Memory content is stored separately from embeddings. Each embedding
//! profile owns a dimension-specific table so one memory can carry multiple
//! embedding versions without losing libSQL's `F32_BLOB(N)` typing.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use libsql::{Connection, Transaction, params};
use nuillu_memory::{IndexedMemory, MemoryQuery, MemoryRecord, MemoryStore, NewMemory};
pub use nuillu_module::ports::Embedder;
use nuillu_module::ports::PortError;
use nuillu_reward::{
    IndexedPolicy, NewPolicy, PolicyQuery, PolicyRecord, PolicySearchHit, PolicyStore,
};
use nuillu_types::{
    MemoryContent, MemoryIndex, MemoryRank, PolicyIndex, PolicyRank, SignedUnitF32, UnitF32,
};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const DEFAULT_MEMORY_TABLE: &str = "memories";
const DEFAULT_POLICY_TABLE: &str = "policies";
const PROFILE_REGISTRY_TABLE: &str = "memory_embedding_profiles";
const POLICY_PROFILE_REGISTRY_TABLE: &str = "policy_embedding_profiles";
const MAX_VECTOR_DIMS: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingProfile {
    pub name: String,
    pub version: String,
    pub dimensions: usize,
}

impl EmbeddingProfile {
    pub fn new(name: impl Into<String>, version: impl Into<String>, dimensions: usize) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            dimensions,
        }
    }

    pub fn default_for_dimensions(dimensions: usize) -> Self {
        Self::new("default", "v1", dimensions)
    }

    pub fn profile_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update([0]);
        hasher.update(self.version.as_bytes());
        hasher.update([0]);
        hasher.update(self.dimensions.to_string().as_bytes());
        let digest = hasher.finalize();
        format!("p{}", short_hex(&digest[..8]))
    }

    fn validate(&self) -> Result<(), PortError> {
        if self.name.trim().is_empty() {
            return Err(PortError::InvalidInput(
                "embedding profile name must not be empty".into(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(PortError::InvalidInput(
                "embedding profile version must not be empty".into(),
            ));
        }
        validate_dimensions("embedding profile dimensions", self.dimensions)
    }
}

#[derive(Clone, Debug)]
pub struct LibsqlMemoryStoreConfig {
    pub path: PathBuf,
    pub table_name: String,
    pub active_profile: EmbeddingProfile,
}

impl LibsqlMemoryStoreConfig {
    pub fn local(path: impl Into<PathBuf>, vector_dims: usize) -> Self {
        Self {
            path: path.into(),
            table_name: DEFAULT_MEMORY_TABLE.to_owned(),
            active_profile: EmbeddingProfile::default_for_dimensions(vector_dims),
        }
    }

    pub fn with_active_profile(mut self, active_profile: EmbeddingProfile) -> Self {
        self.active_profile = active_profile;
        self
    }
}

#[derive(Clone, Debug)]
pub struct LibsqlPolicyStoreConfig {
    pub path: PathBuf,
    pub table_name: String,
    pub active_profile: EmbeddingProfile,
}

impl LibsqlPolicyStoreConfig {
    pub fn local(path: impl Into<PathBuf>, vector_dims: usize) -> Self {
        Self {
            path: path.into(),
            table_name: DEFAULT_POLICY_TABLE.to_owned(),
            active_profile: EmbeddingProfile::default_for_dimensions(vector_dims),
        }
    }

    pub fn with_active_profile(mut self, active_profile: EmbeddingProfile) -> Self {
        self.active_profile = active_profile;
        self
    }
}

#[derive(Clone)]
pub struct LibsqlMemoryStore {
    conn: Connection,
    table_name: String,
    active_profile: EmbeddingProfile,
    profile_id: String,
    embedding_table_name: String,
    embedder: Arc<dyn Embedder>,
}

#[derive(Clone)]
pub struct LibsqlPolicyStore {
    conn: Connection,
    table_name: String,
    active_profile: EmbeddingProfile,
    profile_id: String,
    embedding_table_name: String,
    embedder: Arc<dyn Embedder>,
}

impl LibsqlMemoryStore {
    pub async fn connect(
        config: LibsqlMemoryStoreConfig,
        embedder: Box<dyn Embedder>,
    ) -> Result<Self, PortError> {
        let database = libsql::Builder::new_local(config.path)
            .build()
            .await
            .map_err(map_libsql_error)?;
        let conn = database.connect().map_err(map_libsql_error)?;
        Self::from_connection(conn, config.table_name, config.active_profile, embedder).await
    }

    pub async fn from_connection(
        conn: Connection,
        table_name: impl Into<String>,
        active_profile: EmbeddingProfile,
        embedder: Box<dyn Embedder>,
    ) -> Result<Self, PortError> {
        let table_name = table_name.into();
        validate_identifier("memory table name", &table_name)?;
        active_profile.validate()?;

        let embedder: Arc<dyn Embedder> = Arc::from(embedder);
        let embedder_dims = embedder.dimensions();
        if embedder_dims != active_profile.dimensions {
            return Err(PortError::InvalidInput(format!(
                "libSQL embedding dimension mismatch: profile={}, embedder={embedder_dims}",
                active_profile.dimensions
            )));
        }

        let profile_id = active_profile.profile_id();
        let embedding_table_name = format!("{table_name}_embeddings_{profile_id}");
        validate_identifier("embedding table name", &embedding_table_name)?;

        let store = Self {
            conn,
            table_name,
            active_profile,
            profile_id,
            embedding_table_name,
            embedder,
        };
        store.migrate().await?;
        Ok(store)
    }

    pub fn active_profile(&self) -> &EmbeddingProfile {
        &self.active_profile
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn embedding_table_name(&self) -> &str {
        &self.embedding_table_name
    }

    pub async fn backfill_active_profile(&self, limit: usize) -> Result<usize, PortError> {
        if limit == 0 {
            return Ok(0);
        }

        let sql = format!(
            r#"
            SELECT m.id, m.content, m.updated_at_ms
            FROM {memories} AS m
            LEFT JOIN {embeddings} AS e ON e.memory_id = m.id
            WHERE m.deleted_at_ms IS NULL
              AND (
                e.memory_id IS NULL
                OR e.content_updated_at_ms != m.updated_at_ms
              )
            ORDER BY m.id ASC
            LIMIT ?1
            "#,
            memories = self.table_name,
            embeddings = self.embedding_table_name,
        );

        let mut rows = self
            .conn
            .query(&sql, [limit_to_i64(limit)])
            .await
            .map_err(map_libsql_error)?;
        let mut pending = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            pending.push(PendingEmbedding {
                memory_id: row.get(0).map_err(map_libsql_error)?,
                content: row.get(1).map_err(map_libsql_error)?,
                content_updated_at_ms: row.get(2).map_err(map_libsql_error)?,
            });
        }

        let mut written = 0;
        for item in pending {
            let embedding_json = self.embed_json(&item.content).await?;
            self.upsert_embedding_conn(
                item.memory_id,
                &embedding_json,
                now_ms(),
                item.content_updated_at_ms,
            )
            .await?;
            written += 1;
        }
        Ok(written)
    }

    async fn migrate(&self) -> Result<(), PortError> {
        let memory_rank_idx = format!("{}_rank_live_idx", self.table_name);
        validate_identifier("memory rank index name", &memory_rank_idx)?;

        let ddl = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {memories} (
              id INTEGER PRIMARY KEY,
              memory_index TEXT NOT NULL UNIQUE,
              content TEXT NOT NULL,
              rank INTEGER NOT NULL,
              occurred_at_ms INTEGER,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL,
              source_ids TEXT,
              metadata_json TEXT,
              deleted_at_ms INTEGER
            );

            CREATE INDEX IF NOT EXISTS {memory_rank_idx}
            ON {memories}(rank)
            WHERE deleted_at_ms IS NULL;

            CREATE TABLE IF NOT EXISTS {profiles} (
              profile_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              version TEXT NOT NULL,
              dimensions INTEGER NOT NULL,
              table_name TEXT NOT NULL UNIQUE,
              created_at_ms INTEGER NOT NULL
            );
            "#,
            memories = self.table_name,
            memory_rank_idx = memory_rank_idx,
            profiles = PROFILE_REGISTRY_TABLE,
        );
        self.conn
            .execute_batch(&ddl)
            .await
            .map_err(map_libsql_error)?;
        self.add_column_if_missing(&self.table_name, "occurred_at_ms", "INTEGER")
            .await?;

        self.register_active_profile().await?;
        self.create_active_embedding_table().await
    }

    async fn add_column_if_missing(
        &self,
        table_name: &str,
        column_name: &str,
        column_type: &str,
    ) -> Result<(), PortError> {
        validate_identifier("migration table name", table_name)?;
        validate_identifier("migration column name", column_name)?;
        let pragma = format!("PRAGMA table_info({table_name})");
        let mut rows = self
            .conn
            .query(&pragma, ())
            .await
            .map_err(map_libsql_error)?;
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            let name: String = row.get(1).map_err(map_libsql_error)?;
            if name == column_name {
                return Ok(());
            }
        }

        let sql = format!("ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}");
        self.conn
            .execute(&sql, ())
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn register_active_profile(&self) -> Result<(), PortError> {
        let mut rows = self
            .conn
            .query(
                &format!(
                    r#"
                    SELECT name, version, dimensions, table_name
                    FROM {PROFILE_REGISTRY_TABLE}
                    WHERE profile_id = ?1
                    LIMIT 1
                    "#
                ),
                [self.profile_id.as_str()],
            )
            .await
            .map_err(map_libsql_error)?;

        if let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            let name: String = row.get(0).map_err(map_libsql_error)?;
            let version: String = row.get(1).map_err(map_libsql_error)?;
            let dimensions: i64 = row.get(2).map_err(map_libsql_error)?;
            let table_name: String = row.get(3).map_err(map_libsql_error)?;
            let expected_dimensions = self.active_profile.dimensions as i64;
            if name != self.active_profile.name
                || version != self.active_profile.version
                || dimensions != expected_dimensions
                || table_name != self.embedding_table_name
            {
                return Err(PortError::InvalidData(format!(
                    "embedding profile registry conflict for {}",
                    self.profile_id
                )));
            }
            return Ok(());
        }
        drop(rows);

        self.conn
            .execute(
                &format!(
                    r#"
                    INSERT INTO {PROFILE_REGISTRY_TABLE} (
                      profile_id,
                      name,
                      version,
                      dimensions,
                      table_name,
                      created_at_ms
                    )
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                    "#
                ),
                params![
                    self.profile_id.as_str(),
                    self.active_profile.name.as_str(),
                    self.active_profile.version.as_str(),
                    self.active_profile.dimensions as i64,
                    self.embedding_table_name.as_str(),
                    now_ms(),
                ],
            )
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn create_active_embedding_table(&self) -> Result<(), PortError> {
        let ddl = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {embeddings} (
              memory_id INTEGER PRIMARY KEY,
              embedding F32_BLOB({dimensions}) NOT NULL,
              embedded_at_ms INTEGER NOT NULL,
              content_updated_at_ms INTEGER NOT NULL,
              FOREIGN KEY(memory_id) REFERENCES {memories}(id)
            );
            "#,
            embeddings = self.embedding_table_name,
            dimensions = self.active_profile.dimensions,
            memories = self.table_name,
        );
        self.conn
            .execute_batch(&ddl)
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn embed_json(&self, text: &str) -> Result<String, PortError> {
        let embedding = self.embedder.embed(text).await?;
        self.validate_embedding(&embedding)?;
        serde_json::to_string(&embedding).map_err(|error| PortError::Backend(error.to_string()))
    }

    fn validate_embedding(&self, embedding: &[f32]) -> Result<(), PortError> {
        if embedding.len() != self.active_profile.dimensions {
            return Err(PortError::InvalidInput(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.active_profile.dimensions,
                embedding.len()
            )));
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(PortError::InvalidData(
                "embedding contains NaN or infinity".into(),
            ));
        }
        Ok(())
    }

    async fn upsert_memory_tx(
        &self,
        tx: &Transaction,
        index: &MemoryIndex,
        content: &str,
        rank: MemoryRank,
        occurred_at: Option<chrono::DateTime<chrono::Utc>>,
        source_ids_json: Option<&str>,
        now: i64,
    ) -> Result<(i64, i64), PortError> {
        let sql = format!(
            r#"
            INSERT INTO {memories} (
              memory_index,
              content,
              rank,
              occurred_at_ms,
              created_at_ms,
              updated_at_ms,
              source_ids,
              metadata_json,
              deleted_at_ms
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, NULL)
            ON CONFLICT(memory_index) DO UPDATE SET
              content = excluded.content,
              rank = excluded.rank,
              occurred_at_ms = excluded.occurred_at_ms,
              updated_at_ms = excluded.updated_at_ms,
              source_ids = excluded.source_ids,
              deleted_at_ms = NULL
            "#,
            memories = self.table_name,
        );
        tx.execute(
            &sql,
            params![
                index.as_str(),
                content,
                rank_to_i64(rank),
                occurred_at.map(|at| at.timestamp_millis()),
                now,
                now,
                source_ids_json,
            ],
        )
        .await
        .map_err(map_libsql_error)?;

        self.memory_id_and_updated_at_tx(tx, index).await
    }

    async fn memory_id_and_updated_at_tx(
        &self,
        tx: &Transaction,
        index: &MemoryIndex,
    ) -> Result<(i64, i64), PortError> {
        let sql = format!(
            r#"
            SELECT id, updated_at_ms
            FROM {memories}
            WHERE memory_index = ?1
            LIMIT 1
            "#,
            memories = self.table_name,
        );
        let mut rows = tx
            .query(&sql, [index.as_str()])
            .await
            .map_err(map_libsql_error)?;
        let Some(row) = rows.next().await.map_err(map_libsql_error)? else {
            return Err(PortError::Backend(format!(
                "memory row was not found after upsert: {}",
                index.as_str()
            )));
        };
        Ok((
            row.get(0).map_err(map_libsql_error)?,
            row.get(1).map_err(map_libsql_error)?,
        ))
    }

    async fn upsert_embedding_tx(
        &self,
        tx: &Transaction,
        memory_id: i64,
        embedding_json: &str,
        embedded_at_ms: i64,
        content_updated_at_ms: i64,
    ) -> Result<(), PortError> {
        let sql = self.upsert_embedding_sql();
        tx.execute(
            &sql,
            params![
                memory_id,
                embedding_json,
                embedded_at_ms,
                content_updated_at_ms,
            ],
        )
        .await
        .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn upsert_embedding_conn(
        &self,
        memory_id: i64,
        embedding_json: &str,
        embedded_at_ms: i64,
        content_updated_at_ms: i64,
    ) -> Result<(), PortError> {
        let sql = self.upsert_embedding_sql();
        self.conn
            .execute(
                &sql,
                params![
                    memory_id,
                    embedding_json,
                    embedded_at_ms,
                    content_updated_at_ms,
                ],
            )
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }

    fn upsert_embedding_sql(&self) -> String {
        format!(
            r#"
            INSERT INTO {embeddings} (
              memory_id,
              embedding,
              embedded_at_ms,
              content_updated_at_ms
            )
            VALUES (?1, vector32(?2), ?3, ?4)
            ON CONFLICT(memory_id) DO UPDATE SET
              embedding = excluded.embedding,
              embedded_at_ms = excluded.embedded_at_ms,
              content_updated_at_ms = excluded.content_updated_at_ms
            "#,
            embeddings = self.embedding_table_name,
        )
    }

    async fn soft_delete_many_tx(
        &self,
        tx: &Transaction,
        indexes: &[MemoryIndex],
        now: i64,
    ) -> Result<(), PortError> {
        if indexes.is_empty() {
            return Ok(());
        }

        let sql = format!(
            r#"
            UPDATE {memories}
            SET deleted_at_ms = ?2,
                updated_at_ms = ?2
            WHERE memory_index = ?1
            "#,
            memories = self.table_name,
        );
        for index in indexes {
            tx.execute(&sql, params![index.as_str(), now])
                .await
                .map_err(map_libsql_error)?;
        }
        Ok(())
    }

    fn row_to_record(row: &libsql::Row) -> Result<MemoryRecord, PortError> {
        let index: String = row.get(0).map_err(map_libsql_error)?;
        let content: String = row.get(1).map_err(map_libsql_error)?;
        let rank: i64 = row.get(2).map_err(map_libsql_error)?;
        let occurred_at_ms: Option<i64> = row.get(3).map_err(map_libsql_error)?;
        Ok(MemoryRecord {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank: rank_from_i64(rank)?,
            occurred_at: occurred_at_ms.and_then(chrono::DateTime::from_timestamp_millis),
        })
    }

    async fn put_indexed(
        &self,
        mem: IndexedMemory,
        source_ids_json: Option<String>,
    ) -> Result<(), PortError> {
        let embedding_json = self.embed_json(mem.content.as_str()).await?;
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let (memory_id, updated_at_ms) = self
            .upsert_memory_tx(
                &tx,
                &mem.index,
                mem.content.as_str(),
                mem.rank,
                mem.occurred_at,
                source_ids_json.as_deref(),
                now,
            )
            .await?;
        self.upsert_embedding_tx(&tx, memory_id, &embedding_json, now, updated_at_ms)
            .await?;
        tx.commit().await.map_err(map_libsql_error)?;
        Ok(())
    }
}

impl LibsqlPolicyStore {
    pub async fn connect(
        config: LibsqlPolicyStoreConfig,
        embedder: Box<dyn Embedder>,
    ) -> Result<Self, PortError> {
        let database = libsql::Builder::new_local(config.path)
            .build()
            .await
            .map_err(map_libsql_error)?;
        let conn = database.connect().map_err(map_libsql_error)?;
        Self::from_connection(conn, config.table_name, config.active_profile, embedder).await
    }

    pub async fn from_connection(
        conn: Connection,
        table_name: impl Into<String>,
        active_profile: EmbeddingProfile,
        embedder: Box<dyn Embedder>,
    ) -> Result<Self, PortError> {
        let table_name = table_name.into();
        validate_identifier("policy table name", &table_name)?;
        active_profile.validate()?;

        let embedder: Arc<dyn Embedder> = Arc::from(embedder);
        let embedder_dims = embedder.dimensions();
        if embedder_dims != active_profile.dimensions {
            return Err(PortError::InvalidInput(format!(
                "libSQL policy embedding dimension mismatch: profile={}, embedder={embedder_dims}",
                active_profile.dimensions
            )));
        }

        let profile_id = active_profile.profile_id();
        let embedding_table_name = format!("{table_name}_trigger_embeddings_{profile_id}");
        validate_identifier("policy embedding table name", &embedding_table_name)?;

        let store = Self {
            conn,
            table_name,
            active_profile,
            profile_id,
            embedding_table_name,
            embedder,
        };
        store.migrate().await?;
        Ok(store)
    }

    pub fn active_profile(&self) -> &EmbeddingProfile {
        &self.active_profile
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn embedding_table_name(&self) -> &str {
        &self.embedding_table_name
    }

    pub async fn backfill_active_profile(&self, limit: usize) -> Result<usize, PortError> {
        if limit == 0 {
            return Ok(0);
        }

        let sql = format!(
            r#"
            SELECT p.id, p.trigger, p.updated_at_ms
            FROM {policies} AS p
            LEFT JOIN {embeddings} AS e ON e.policy_id = p.id
            WHERE p.deleted_at_ms IS NULL
              AND (
                e.policy_id IS NULL
                OR e.trigger_updated_at_ms != p.updated_at_ms
              )
            ORDER BY p.id ASC
            LIMIT ?1
            "#,
            policies = self.table_name,
            embeddings = self.embedding_table_name,
        );

        let mut rows = self
            .conn
            .query(&sql, [limit_to_i64(limit)])
            .await
            .map_err(map_libsql_error)?;
        let mut pending = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            pending.push(PendingPolicyEmbedding {
                policy_id: row.get(0).map_err(map_libsql_error)?,
                trigger: row.get(1).map_err(map_libsql_error)?,
                trigger_updated_at_ms: row.get(2).map_err(map_libsql_error)?,
            });
        }

        let mut written = 0;
        for item in pending {
            let embedding_json = self.embed_json(&item.trigger).await?;
            self.upsert_embedding_conn(
                item.policy_id,
                &embedding_json,
                now_ms(),
                item.trigger_updated_at_ms,
            )
            .await?;
            written += 1;
        }
        Ok(written)
    }

    async fn migrate(&self) -> Result<(), PortError> {
        let policy_rank_idx = format!("{}_rank_live_idx", self.table_name);
        validate_identifier("policy rank index name", &policy_rank_idx)?;

        let ddl = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {policies} (
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

            CREATE INDEX IF NOT EXISTS {policy_rank_idx}
            ON {policies}(rank)
            WHERE deleted_at_ms IS NULL;

            CREATE TABLE IF NOT EXISTS {profiles} (
              profile_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              version TEXT NOT NULL,
              dimensions INTEGER NOT NULL,
              table_name TEXT NOT NULL UNIQUE,
              created_at_ms INTEGER NOT NULL
            );
            "#,
            policies = self.table_name,
            policy_rank_idx = policy_rank_idx,
            profiles = POLICY_PROFILE_REGISTRY_TABLE,
        );
        self.conn
            .execute_batch(&ddl)
            .await
            .map_err(map_libsql_error)?;

        self.register_active_profile().await?;
        self.create_active_embedding_table().await
    }

    async fn register_active_profile(&self) -> Result<(), PortError> {
        let mut rows = self
            .conn
            .query(
                &format!(
                    r#"
                    SELECT name, version, dimensions, table_name
                    FROM {POLICY_PROFILE_REGISTRY_TABLE}
                    WHERE profile_id = ?1
                    LIMIT 1
                    "#
                ),
                [self.profile_id.as_str()],
            )
            .await
            .map_err(map_libsql_error)?;

        if let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            let name: String = row.get(0).map_err(map_libsql_error)?;
            let version: String = row.get(1).map_err(map_libsql_error)?;
            let dimensions: i64 = row.get(2).map_err(map_libsql_error)?;
            let table_name: String = row.get(3).map_err(map_libsql_error)?;
            let expected_dimensions = self.active_profile.dimensions as i64;
            if name != self.active_profile.name
                || version != self.active_profile.version
                || dimensions != expected_dimensions
                || table_name != self.embedding_table_name
            {
                return Err(PortError::InvalidData(format!(
                    "policy embedding profile registry conflict for {}",
                    self.profile_id
                )));
            }
            return Ok(());
        }
        drop(rows);

        self.conn
            .execute(
                &format!(
                    r#"
                    INSERT INTO {POLICY_PROFILE_REGISTRY_TABLE} (
                      profile_id,
                      name,
                      version,
                      dimensions,
                      table_name,
                      created_at_ms
                    )
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                    "#
                ),
                params![
                    self.profile_id.as_str(),
                    self.active_profile.name.as_str(),
                    self.active_profile.version.as_str(),
                    self.active_profile.dimensions as i64,
                    self.embedding_table_name.as_str(),
                    now_ms(),
                ],
            )
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn create_active_embedding_table(&self) -> Result<(), PortError> {
        let ddl = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {embeddings} (
              policy_id INTEGER PRIMARY KEY,
              trigger_embedding F32_BLOB({dimensions}) NOT NULL,
              embedded_at_ms INTEGER NOT NULL,
              trigger_updated_at_ms INTEGER NOT NULL,
              FOREIGN KEY(policy_id) REFERENCES {policies}(id)
            );
            "#,
            embeddings = self.embedding_table_name,
            dimensions = self.active_profile.dimensions,
            policies = self.table_name,
        );
        self.conn
            .execute_batch(&ddl)
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn embed_json(&self, text: &str) -> Result<String, PortError> {
        let embedding = self.embedder.embed(text).await?;
        self.validate_embedding(&embedding)?;
        serde_json::to_string(&embedding).map_err(|error| PortError::Backend(error.to_string()))
    }

    fn validate_embedding(&self, embedding: &[f32]) -> Result<(), PortError> {
        if embedding.len() != self.active_profile.dimensions {
            return Err(PortError::InvalidInput(format!(
                "policy embedding dimension mismatch: expected {}, got {}",
                self.active_profile.dimensions,
                embedding.len()
            )));
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(PortError::InvalidData(
                "policy embedding contains NaN or infinity".into(),
            ));
        }
        Ok(())
    }

    async fn upsert_policy_tx(
        &self,
        tx: &Transaction,
        policy: IndexedPolicy,
        now: i64,
    ) -> Result<(i64, i64), PortError> {
        let sql = format!(
            r#"
            INSERT INTO {policies} (
              policy_index,
              trigger,
              behavior,
              rank,
              expected_reward,
              confidence,
              value,
              reward_tokens,
              decay_remaining_secs,
              created_at_ms,
              updated_at_ms,
              last_reinforced_at_ms,
              deleted_at_ms
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10, NULL, NULL)
            ON CONFLICT(policy_index) DO UPDATE SET
              trigger = excluded.trigger,
              behavior = excluded.behavior,
              rank = excluded.rank,
              expected_reward = excluded.expected_reward,
              confidence = excluded.confidence,
              value = excluded.value,
              reward_tokens = excluded.reward_tokens,
              decay_remaining_secs = excluded.decay_remaining_secs,
              updated_at_ms = excluded.updated_at_ms,
              deleted_at_ms = NULL
            "#,
            policies = self.table_name,
        );
        tx.execute(
            &sql,
            params![
                policy.index.as_str(),
                policy.trigger.as_str(),
                policy.behavior.as_str(),
                policy_rank_to_i64(policy.rank),
                f64::from(policy.expected_reward.get()),
                f64::from(policy.confidence.get()),
                f64::from(policy.value.get()),
                i64::from(policy.reward_tokens),
                policy.decay_remaining_secs,
                now,
            ],
        )
        .await
        .map_err(map_libsql_error)?;

        self.policy_id_and_updated_at_tx(tx, &policy.index).await
    }

    async fn policy_id_and_updated_at_tx(
        &self,
        tx: &Transaction,
        index: &PolicyIndex,
    ) -> Result<(i64, i64), PortError> {
        let sql = format!(
            r#"
            SELECT id, updated_at_ms
            FROM {policies}
            WHERE policy_index = ?1
            LIMIT 1
            "#,
            policies = self.table_name,
        );
        let mut rows = tx
            .query(&sql, [index.as_str()])
            .await
            .map_err(map_libsql_error)?;
        let Some(row) = rows.next().await.map_err(map_libsql_error)? else {
            return Err(PortError::Backend(format!(
                "policy row was not found after upsert: {}",
                index.as_str()
            )));
        };
        Ok((
            row.get(0).map_err(map_libsql_error)?,
            row.get(1).map_err(map_libsql_error)?,
        ))
    }

    async fn upsert_embedding_tx(
        &self,
        tx: &Transaction,
        policy_id: i64,
        embedding_json: &str,
        embedded_at_ms: i64,
        trigger_updated_at_ms: i64,
    ) -> Result<(), PortError> {
        let sql = self.upsert_embedding_sql();
        tx.execute(
            &sql,
            params![
                policy_id,
                embedding_json,
                embedded_at_ms,
                trigger_updated_at_ms,
            ],
        )
        .await
        .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn upsert_embedding_conn(
        &self,
        policy_id: i64,
        embedding_json: &str,
        embedded_at_ms: i64,
        trigger_updated_at_ms: i64,
    ) -> Result<(), PortError> {
        let sql = self.upsert_embedding_sql();
        self.conn
            .execute(
                &sql,
                params![
                    policy_id,
                    embedding_json,
                    embedded_at_ms,
                    trigger_updated_at_ms,
                ],
            )
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }

    fn upsert_embedding_sql(&self) -> String {
        format!(
            r#"
            INSERT INTO {embeddings} (
              policy_id,
              trigger_embedding,
              embedded_at_ms,
              trigger_updated_at_ms
            )
            VALUES (?1, vector32(?2), ?3, ?4)
            ON CONFLICT(policy_id) DO UPDATE SET
              trigger_embedding = excluded.trigger_embedding,
              embedded_at_ms = excluded.embedded_at_ms,
              trigger_updated_at_ms = excluded.trigger_updated_at_ms
            "#,
            embeddings = self.embedding_table_name,
        )
    }

    fn row_to_record(row: &libsql::Row) -> Result<PolicyRecord, PortError> {
        let index: String = row.get(0).map_err(map_libsql_error)?;
        let trigger: String = row.get(1).map_err(map_libsql_error)?;
        let behavior: String = row.get(2).map_err(map_libsql_error)?;
        let rank: i64 = row.get(3).map_err(map_libsql_error)?;
        let expected_reward: f64 = row.get(4).map_err(map_libsql_error)?;
        let confidence: f64 = row.get(5).map_err(map_libsql_error)?;
        let value: f64 = row.get(6).map_err(map_libsql_error)?;
        let reward_tokens: i64 = row.get(7).map_err(map_libsql_error)?;
        let decay_remaining_secs: i64 = row.get(8).map_err(map_libsql_error)?;
        Ok(PolicyRecord {
            index: PolicyIndex::new(index),
            trigger,
            behavior,
            rank: policy_rank_from_i64(rank)?,
            expected_reward: signed_unit_from_f64("policy expected_reward", expected_reward)?,
            confidence: unit_from_f64("policy confidence", confidence)?,
            value: signed_unit_from_f64("policy value", value)?,
            reward_tokens: u32::try_from(reward_tokens).map_err(|_| {
                PortError::InvalidData(format!(
                    "invalid policy reward token count: {reward_tokens}"
                ))
            })?,
            decay_remaining_secs,
        })
    }

    async fn put_indexed(&self, policy: IndexedPolicy) -> Result<(), PortError> {
        let embedding_json = self.embed_json(&policy.trigger).await?;
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let (policy_id, updated_at_ms) = self.upsert_policy_tx(&tx, policy, now).await?;
        self.upsert_embedding_tx(&tx, policy_id, &embedding_json, now, updated_at_ms)
            .await?;
        tx.commit().await.map_err(map_libsql_error)?;
        Ok(())
    }
}

#[async_trait(?Send)]
impl PolicyStore for LibsqlPolicyStore {
    async fn insert(&self, policy: NewPolicy) -> Result<PolicyIndex, PortError> {
        let index = PolicyIndex::new(Uuid::now_v7().to_string());
        self.put_indexed(IndexedPolicy {
            index: index.clone(),
            trigger: policy.trigger,
            behavior: policy.behavior,
            rank: policy.rank,
            expected_reward: policy.expected_reward,
            confidence: policy.confidence,
            value: policy.value,
            reward_tokens: policy.reward_tokens,
            decay_remaining_secs: policy.decay_remaining_secs,
        })
        .await?;
        Ok(index)
    }

    async fn put(&self, policy: IndexedPolicy) -> Result<(), PortError> {
        self.put_indexed(policy).await
    }

    async fn get(&self, index: &PolicyIndex) -> Result<Option<PolicyRecord>, PortError> {
        let sql = format!(
            r#"
            SELECT policy_index, trigger, behavior, rank, expected_reward,
                   confidence, value, reward_tokens, decay_remaining_secs
            FROM {policies}
            WHERE policy_index = ?1
              AND deleted_at_ms IS NULL
            LIMIT 1
            "#,
            policies = self.table_name,
        );
        let mut rows = self
            .conn
            .query(&sql, [index.as_str()])
            .await
            .map_err(map_libsql_error)?;
        match rows.next().await.map_err(map_libsql_error)? {
            Some(row) => Ok(Some(Self::row_to_record(&row)?)),
            None => Ok(None),
        }
    }

    async fn list_by_rank(&self, rank: PolicyRank) -> Result<Vec<PolicyRecord>, PortError> {
        let sql = format!(
            r#"
            SELECT policy_index, trigger, behavior, rank, expected_reward,
                   confidence, value, reward_tokens, decay_remaining_secs
            FROM {policies}
            WHERE rank = ?1
              AND deleted_at_ms IS NULL
            ORDER BY policy_index ASC
            "#,
            policies = self.table_name,
        );
        let mut rows = self
            .conn
            .query(&sql, [policy_rank_to_i64(rank)])
            .await
            .map_err(map_libsql_error)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            out.push(Self::row_to_record(&row)?);
        }
        Ok(out)
    }

    async fn search(&self, q: &PolicyQuery) -> Result<Vec<PolicySearchHit>, PortError> {
        if q.limit == 0 {
            return Ok(Vec::new());
        }

        let embedding_json = self.embed_json(&q.trigger).await?;
        let sql = format!(
            r#"
            SELECT p.policy_index, p.trigger, p.behavior, p.rank,
                   p.expected_reward, p.confidence, p.value,
                   p.reward_tokens, p.decay_remaining_secs,
                   vector_distance_cos(e.trigger_embedding, vector32(?1)) AS distance
            FROM {policies} AS p
            JOIN {embeddings} AS e ON e.policy_id = p.id
            WHERE p.deleted_at_ms IS NULL
              AND e.trigger_updated_at_ms = p.updated_at_ms
              AND (p.rank = ?2 OR p.decay_remaining_secs > 0)
            ORDER BY distance ASC,
                     p.id ASC
            LIMIT ?3
            "#,
            policies = self.table_name,
            embeddings = self.embedding_table_name,
        );
        let mut rows = self
            .conn
            .query(
                &sql,
                params![
                    embedding_json,
                    policy_rank_to_i64(PolicyRank::Core),
                    limit_to_i64(q.limit)
                ],
            )
            .await
            .map_err(map_libsql_error)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            let distance: f64 = row.get(9).map_err(map_libsql_error)?;
            out.push(PolicySearchHit {
                policy: Self::row_to_record(&row)?,
                similarity: (1.0 - distance as f32).clamp(-1.0, 1.0),
            });
        }
        Ok(out)
    }

    async fn reinforce(
        &self,
        index: &PolicyIndex,
        value_delta: f32,
        reward_tokens_delta: u32,
        expected_reward_delta: f32,
        confidence_delta: f32,
    ) -> Result<PolicyRecord, PortError> {
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let sql = format!(
            r#"
            SELECT policy_index, trigger, behavior, rank, expected_reward,
                   confidence, value, reward_tokens, decay_remaining_secs
            FROM {policies}
            WHERE policy_index = ?1
              AND deleted_at_ms IS NULL
            LIMIT 1
            "#,
            policies = self.table_name,
        );
        let mut rows = tx
            .query(&sql, [index.as_str()])
            .await
            .map_err(map_libsql_error)?;
        let Some(row) = rows.next().await.map_err(map_libsql_error)? else {
            return Err(PortError::NotFound(index.as_str().to_owned()));
        };
        let mut record = Self::row_to_record(&row)?;
        drop(rows);

        record.expected_reward =
            SignedUnitF32::clamp(record.expected_reward.get() + expected_reward_delta);
        record.value = SignedUnitF32::clamp(record.value.get() + value_delta);
        record.confidence = UnitF32::clamp(record.confidence.get() + confidence_delta);
        record.reward_tokens = record.reward_tokens.saturating_add(reward_tokens_delta);
        record.rank = rank_after_reinforcement(
            record.rank,
            record.value,
            record.confidence,
            record.reward_tokens,
        );

        let update = format!(
            r#"
            UPDATE {policies}
            SET rank = ?2,
                expected_reward = ?3,
                confidence = ?4,
                value = ?5,
                reward_tokens = ?6,
                updated_at_ms = ?7,
                last_reinforced_at_ms = ?7
            WHERE policy_index = ?1
              AND deleted_at_ms IS NULL
            "#,
            policies = self.table_name,
        );
        tx.execute(
            &update,
            params![
                record.index.as_str(),
                policy_rank_to_i64(record.rank),
                f64::from(record.expected_reward.get()),
                f64::from(record.confidence.get()),
                f64::from(record.value.get()),
                i64::from(record.reward_tokens),
                now,
            ],
        )
        .await
        .map_err(map_libsql_error)?;
        tx.commit().await.map_err(map_libsql_error)?;
        Ok(record)
    }

    async fn delete(&self, index: &PolicyIndex) -> Result<(), PortError> {
        let now = now_ms();
        let sql = format!(
            r#"
            UPDATE {policies}
            SET deleted_at_ms = ?2,
                updated_at_ms = ?2
            WHERE policy_index = ?1
            "#,
            policies = self.table_name,
        );
        self.conn
            .execute(&sql, params![index.as_str(), now])
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }
}

#[async_trait(?Send)]
impl MemoryStore for LibsqlMemoryStore {
    async fn insert(&self, mem: NewMemory) -> Result<MemoryIndex, PortError> {
        let index = MemoryIndex::new(Uuid::now_v7().to_string());
        let embedding_json = self.embed_json(mem.content.as_str()).await?;
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let (memory_id, updated_at_ms) = self
            .upsert_memory_tx(
                &tx,
                &index,
                mem.content.as_str(),
                mem.rank,
                mem.occurred_at,
                None,
                now,
            )
            .await?;
        self.upsert_embedding_tx(&tx, memory_id, &embedding_json, now, updated_at_ms)
            .await?;
        tx.commit().await.map_err(map_libsql_error)?;
        Ok(index)
    }

    async fn put(&self, mem: IndexedMemory) -> Result<(), PortError> {
        self.put_indexed(mem, None).await
    }

    async fn compact(
        &self,
        mem: NewMemory,
        sources: &[MemoryIndex],
    ) -> Result<MemoryIndex, PortError> {
        let index = MemoryIndex::new(Uuid::now_v7().to_string());
        let embedding_json = self.embed_json(mem.content.as_str()).await?;
        let source_ids_json = source_ids_json(sources)?;
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let (memory_id, updated_at_ms) = self
            .upsert_memory_tx(
                &tx,
                &index,
                mem.content.as_str(),
                mem.rank,
                mem.occurred_at,
                Some(&source_ids_json),
                now,
            )
            .await?;
        self.upsert_embedding_tx(&tx, memory_id, &embedding_json, now, updated_at_ms)
            .await?;
        self.soft_delete_many_tx(&tx, sources, now).await?;
        tx.commit().await.map_err(map_libsql_error)?;
        Ok(index)
    }

    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        sources: &[MemoryIndex],
    ) -> Result<(), PortError> {
        let source_ids_json = source_ids_json(sources)?;
        let embedding_json = self.embed_json(mem.content.as_str()).await?;
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let (memory_id, updated_at_ms) = self
            .upsert_memory_tx(
                &tx,
                &mem.index,
                mem.content.as_str(),
                mem.rank,
                mem.occurred_at,
                Some(&source_ids_json),
                now,
            )
            .await?;
        self.upsert_embedding_tx(&tx, memory_id, &embedding_json, now, updated_at_ms)
            .await?;
        self.soft_delete_many_tx(&tx, sources, now).await?;
        tx.commit().await.map_err(map_libsql_error)?;
        Ok(())
    }

    async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        let sql = format!(
            r#"
            SELECT memory_index, content, rank, occurred_at_ms
            FROM {memories}
            WHERE memory_index = ?1
              AND deleted_at_ms IS NULL
            LIMIT 1
            "#,
            memories = self.table_name,
        );
        let mut rows = self
            .conn
            .query(&sql, [index.as_str()])
            .await
            .map_err(map_libsql_error)?;
        match rows.next().await.map_err(map_libsql_error)? {
            Some(row) => Ok(Some(Self::row_to_record(&row)?)),
            None => Ok(None),
        }
    }

    async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
        let sql = format!(
            r#"
            SELECT memory_index, content, rank, occurred_at_ms
            FROM {memories}
            WHERE rank = ?1
              AND deleted_at_ms IS NULL
            ORDER BY memory_index ASC
            "#,
            memories = self.table_name,
        );
        let mut rows = self
            .conn
            .query(&sql, [rank_to_i64(rank)])
            .await
            .map_err(map_libsql_error)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            out.push(Self::row_to_record(&row)?);
        }
        Ok(out)
    }

    async fn search(&self, q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        if q.limit == 0 {
            return Ok(Vec::new());
        }

        let embedding_json = self.embed_json(&q.text).await?;
        let mut out = Vec::new();

        let sql = format!(
            r#"
            SELECT m.memory_index, m.content, m.rank, m.occurred_at_ms
            FROM {memories} AS m
            JOIN {embeddings} AS e ON e.memory_id = m.id
            WHERE m.deleted_at_ms IS NULL
              AND e.content_updated_at_ms = m.updated_at_ms
            ORDER BY vector_distance_cos(e.embedding, vector32(?1)) ASC,
                     m.id ASC
            LIMIT ?2
            "#,
            memories = self.table_name,
            embeddings = self.embedding_table_name,
        );
        let mut rows = self
            .conn
            .query(&sql, params![embedding_json, limit_to_i64(q.limit)])
            .await
            .map_err(map_libsql_error)?;
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            out.push(Self::row_to_record(&row)?);
        }

        Ok(out)
    }

    async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
        let now = now_ms();
        let sql = format!(
            r#"
            UPDATE {memories}
            SET deleted_at_ms = ?2,
                updated_at_ms = ?2
            WHERE memory_index = ?1
            "#,
            memories = self.table_name,
        );
        self.conn
            .execute(&sql, params![index.as_str(), now])
            .await
            .map_err(map_libsql_error)?;
        Ok(())
    }
}

#[derive(Debug)]
struct PendingEmbedding {
    memory_id: i64,
    content: String,
    content_updated_at_ms: i64,
}

#[derive(Debug)]
struct PendingPolicyEmbedding {
    policy_id: i64,
    trigger: String,
    trigger_updated_at_ms: i64,
}

fn validate_dimensions(label: &str, dimensions: usize) -> Result<(), PortError> {
    if dimensions == 0 {
        return Err(PortError::InvalidInput(format!(
            "{label} must be greater than zero"
        )));
    }
    if dimensions > MAX_VECTOR_DIMS {
        return Err(PortError::InvalidInput(format!(
            "{label} must be <= {MAX_VECTOR_DIMS}, got {dimensions}"
        )));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), PortError> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err(PortError::InvalidInput(format!(
            "{label} must not be empty"
        )));
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(PortError::InvalidInput(format!(
            "{label} must start with an ASCII letter or underscore: {value}"
        )));
    }
    if chars.any(|ch| !(ch == '_' || ch.is_ascii_alphanumeric())) {
        return Err(PortError::InvalidInput(format!(
            "{label} must contain only ASCII letters, digits, and underscores: {value}"
        )));
    }
    Ok(())
}

fn source_ids_json(ids: &[MemoryIndex]) -> Result<String, PortError> {
    let source_ids = ids
        .iter()
        .map(|id| id.as_str().to_owned())
        .collect::<Vec<_>>();
    serde_json::to_string(&source_ids).map_err(|error| PortError::Backend(error.to_string()))
}

fn rank_to_i64(rank: MemoryRank) -> i64 {
    match rank {
        MemoryRank::ShortTerm => 0,
        MemoryRank::MidTerm => 1,
        MemoryRank::LongTerm => 2,
        MemoryRank::Permanent => 3,
        MemoryRank::Identity => 4,
    }
}

fn rank_from_i64(value: i64) -> Result<MemoryRank, PortError> {
    match value {
        0 => Ok(MemoryRank::ShortTerm),
        1 => Ok(MemoryRank::MidTerm),
        2 => Ok(MemoryRank::LongTerm),
        3 => Ok(MemoryRank::Permanent),
        4 => Ok(MemoryRank::Identity),
        _ => Err(PortError::InvalidData(format!(
            "invalid memory rank: {value}"
        ))),
    }
}

fn policy_rank_to_i64(rank: PolicyRank) -> i64 {
    match rank {
        PolicyRank::Tentative => 0,
        PolicyRank::Provisional => 1,
        PolicyRank::Established => 2,
        PolicyRank::Habit => 3,
        PolicyRank::Core => 4,
    }
}

fn policy_rank_from_i64(value: i64) -> Result<PolicyRank, PortError> {
    match value {
        0 => Ok(PolicyRank::Tentative),
        1 => Ok(PolicyRank::Provisional),
        2 => Ok(PolicyRank::Established),
        3 => Ok(PolicyRank::Habit),
        4 => Ok(PolicyRank::Core),
        _ => Err(PortError::InvalidData(format!(
            "invalid policy rank: {value}"
        ))),
    }
}

fn signed_unit_from_f64(label: &str, value: f64) -> Result<SignedUnitF32, PortError> {
    if !value.is_finite() {
        return Err(PortError::InvalidData(format!("{label} is not finite")));
    }
    SignedUnitF32::new(value as f32)
        .map_err(|error| PortError::InvalidData(format!("{label}: {error}")))
}

fn unit_from_f64(label: &str, value: f64) -> Result<UnitF32, PortError> {
    if !value.is_finite() {
        return Err(PortError::InvalidData(format!("{label} is not finite")));
    }
    UnitF32::new(value as f32).map_err(|error| PortError::InvalidData(format!("{label}: {error}")))
}

fn rank_after_reinforcement(
    current: PolicyRank,
    value: SignedUnitF32,
    confidence: UnitF32,
    reward_tokens: u32,
) -> PolicyRank {
    if matches!(current, PolicyRank::Core) {
        return PolicyRank::Core;
    }
    let value = value.get();
    let confidence = confidence.get();
    if reward_tokens >= 16 && value >= 0.7 && confidence >= 0.7 {
        PolicyRank::Habit
    } else if reward_tokens >= 8 && value >= 0.45 && confidence >= 0.5 {
        PolicyRank::Established
    } else if reward_tokens >= 2 && value >= 0.2 {
        PolicyRank::Provisional
    } else {
        PolicyRank::Tentative
    }
}

fn limit_to_i64(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(i64::MAX)
}

fn short_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn map_libsql_error(error: libsql::Error) -> PortError {
    let message = error.to_string();
    let lower = message.to_ascii_lowercase();
    if lower.contains("not found") || lower.contains("does not exist") {
        PortError::NotFound(message)
    } else if lower.contains("invalid") || lower.contains("constraint") {
        PortError::InvalidInput(message)
    } else {
        PortError::Backend(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug)]
    struct TestEmbedder {
        dims: usize,
    }

    impl TestEmbedder {
        fn new(dims: usize) -> Box<dyn Embedder> {
            Box::new(Self { dims })
        }
    }

    #[async_trait(?Send)]
    impl Embedder for TestEmbedder {
        fn dimensions(&self) -> usize {
            self.dims
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError> {
            let mut out = vec![0.001; self.dims];
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
    impl Embedder for WrongDimEmbedder {
        fn dimensions(&self) -> usize {
            3
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, PortError> {
            Ok(vec![1.0, 0.0])
        }
    }

    #[tokio::test]
    async fn connect_creates_schema() {
        let store = store_for_profile(test_db_path(), profile("a", "v1", 3)).await;

        let profiles = count_rows(&store, PROFILE_REGISTRY_TABLE).await;
        let memories = count_rows(&store, DEFAULT_MEMORY_TABLE).await;

        assert_eq!(profiles, 1);
        assert_eq!(memories, 0);
    }

    #[tokio::test]
    async fn connect_rejects_dimension_mismatch() {
        let result = LibsqlMemoryStore::connect(
            LibsqlMemoryStoreConfig::local(test_db_path(), 2),
            TestEmbedder::new(3),
        )
        .await;

        assert!(matches!(result, Err(PortError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn insert_then_get_roundtrips_content_and_rank() {
        let store = store().await;
        let occurred_at = chrono::Utc.with_ymd_and_hms(2025, 5, 10, 0, 0, 0).unwrap();
        let id = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha beta"),
                rank: MemoryRank::LongTerm,
                occurred_at: Some(occurred_at),
            })
            .await
            .unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.index, id);
        assert_eq!(got.content.as_str(), "alpha beta");
        assert_eq!(got.rank, MemoryRank::LongTerm);
        assert_eq!(got.occurred_at, Some(occurred_at));
    }

    #[tokio::test]
    async fn migrate_adds_occurred_at_column_to_existing_memory_table() {
        let path = test_db_path();
        let database = libsql::Builder::new_local(path.clone())
            .build()
            .await
            .unwrap();
        let conn = database.connect().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE memories (
              id INTEGER PRIMARY KEY,
              memory_index TEXT NOT NULL UNIQUE,
              content TEXT NOT NULL,
              rank INTEGER NOT NULL,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL,
              source_ids TEXT,
              metadata_json TEXT,
              deleted_at_ms INTEGER
            );
            "#,
        )
        .await
        .unwrap();
        drop(conn);
        drop(database);

        let store = LibsqlMemoryStore::connect(
            LibsqlMemoryStoreConfig::local(path, 3),
            TestEmbedder::new(3),
        )
        .await
        .unwrap();

        assert!(column_exists(&store, DEFAULT_MEMORY_TABLE, "occurred_at_ms").await);
    }

    #[tokio::test]
    async fn identity_rank_mapping_uses_next_stable_value() {
        assert_eq!(rank_to_i64(MemoryRank::Identity), 4);
        assert_eq!(rank_from_i64(4).unwrap(), MemoryRank::Identity);
    }

    #[tokio::test]
    async fn list_by_rank_returns_live_identity_records_in_index_order() {
        let store = store().await;
        store
            .put(IndexedMemory {
                index: MemoryIndex::new("identity-b"),
                content: MemoryContent::new("second"),
                rank: MemoryRank::Identity,
                occurred_at: None,
            })
            .await
            .unwrap();
        store
            .put(IndexedMemory {
                index: MemoryIndex::new("identity-a"),
                content: MemoryContent::new("first"),
                rank: MemoryRank::Identity,
                occurred_at: None,
            })
            .await
            .unwrap();
        store
            .put(IndexedMemory {
                index: MemoryIndex::new("ordinary"),
                content: MemoryContent::new("ordinary"),
                rank: MemoryRank::Permanent,
                occurred_at: None,
            })
            .await
            .unwrap();
        store.delete(&MemoryIndex::new("identity-b")).await.unwrap();

        let records = store.list_by_rank(MemoryRank::Identity).await.unwrap();

        assert_eq!(
            records
                .iter()
                .map(|record| (record.index.as_str(), record.content.as_str(), record.rank))
                .collect::<Vec<_>>(),
            vec![("identity-a", "first", MemoryRank::Identity)]
        );
    }

    #[tokio::test]
    async fn put_inserts_replaces_and_revives() {
        let store = store().await;
        let id = MemoryIndex::new("replica-1");
        store
            .put(IndexedMemory {
                index: id.clone(),
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();
        store.delete(&id).await.unwrap();
        assert!(store.get(&id).await.unwrap().is_none());

        store
            .put(IndexedMemory {
                index: id.clone(),
                content: MemoryContent::new("beta"),
                rank: MemoryRank::Permanent,
                occurred_at: None,
            })
            .await
            .unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.content.as_str(), "beta");
        assert_eq!(got.rank, MemoryRank::Permanent);
    }

    #[tokio::test]
    async fn search_uses_cosine_order_and_limit() {
        let store = store().await;
        let alpha = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();
        store
            .insert(NewMemory {
                content: MemoryContent::new("beta"),
                rank: MemoryRank::LongTerm,
                occurred_at: None,
            })
            .await
            .unwrap();
        store
            .insert(NewMemory {
                content: MemoryContent::new("gamma"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        let hits = store
            .search(&MemoryQuery {
                text: "alpha".into(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, alpha);
    }

    #[tokio::test]
    async fn search_returns_empty_for_zero_limit() {
        let store = store().await;
        store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        let hits = store
            .search(&MemoryQuery {
                text: "alpha".into(),
                limit: 0,
            })
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn delete_hides_rows_and_is_idempotent() {
        let store = store().await;
        let id = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        store.delete(&id).await.unwrap();
        store.delete(&id).await.unwrap();

        assert!(store.get(&id).await.unwrap().is_none());
        let hits = store
            .search(&MemoryQuery {
                text: "alpha".into(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn compact_inserts_merged_row_preserves_sources_and_hides_sources() {
        let store = store().await;
        let first = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();
        let second = store
            .insert(NewMemory {
                content: MemoryContent::new("beta"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        let merged = store
            .compact(
                NewMemory {
                    content: MemoryContent::new("alpha beta"),
                    rank: MemoryRank::LongTerm,
                    occurred_at: None,
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

        let parsed = source_ids_for(&store, &merged).await;
        assert_eq!(parsed, vec![first.to_string(), second.to_string()]);
    }

    #[tokio::test]
    async fn put_compacted_mirrors_replica_flow() {
        let store = store().await;
        let first = MemoryIndex::new("first");
        let second = MemoryIndex::new("second");
        store
            .put(IndexedMemory {
                index: first.clone(),
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();
        store
            .put(IndexedMemory {
                index: second.clone(),
                content: MemoryContent::new("beta"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
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
                    occurred_at: None,
                },
                &[first.clone(), second.clone()],
            )
            .await
            .unwrap();

        assert!(store.get(&first).await.unwrap().is_none());
        assert!(store.get(&second).await.unwrap().is_none());
        assert_eq!(
            store.get(&merged).await.unwrap().unwrap().content.as_str(),
            "alpha beta"
        );

        let parsed = source_ids_for(&store, &merged).await;
        assert_eq!(parsed, vec![first.to_string(), second.to_string()]);
    }

    #[tokio::test]
    async fn wrong_embedding_dimensions_are_rejected_on_write() {
        let store = LibsqlMemoryStore::connect(
            LibsqlMemoryStoreConfig::local(test_db_path(), 3),
            Box::new(WrongDimEmbedder) as Box<dyn Embedder>,
        )
        .await
        .unwrap();

        let error = store
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap_err();

        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn same_memory_can_hold_multiple_profiles_with_different_dimensions() {
        let path = test_db_path();
        let profile_a = profile("test", "a", 3);
        let profile_b = profile("test", "b", 2);
        let store_a = store_for_profile(path.clone(), profile_a).await;
        let id = store_a
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        let store_b = store_for_profile(path, profile_b).await;
        assert_eq!(store_b.backfill_active_profile(10).await.unwrap(), 1);

        assert_eq!(
            count_rows(&store_a, store_a.embedding_table_name()).await,
            1
        );
        assert_eq!(
            count_rows(&store_b, store_b.embedding_table_name()).await,
            1
        );
        assert!(store_b.get(&id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn search_uses_only_active_profile_embeddings() {
        let path = test_db_path();
        let store_a = store_for_profile(path.clone(), profile("test", "a", 3)).await;
        let alpha = store_a
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        let store_b = store_for_profile(path, profile("test", "b", 2)).await;
        store_b
            .put(IndexedMemory {
                index: MemoryIndex::new("beta-only-b"),
                content: MemoryContent::new("beta"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        let hits_a = store_a
            .search(&MemoryQuery {
                text: "beta".into(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(hits_a.len(), 1);
        assert_eq!(hits_a[0].index, alpha);

        let hits_b = store_b
            .search(&MemoryQuery {
                text: "beta".into(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].index, MemoryIndex::new("beta-only-b"));
    }

    #[tokio::test]
    async fn profile_embeddings_become_stale_until_backfilled() {
        let path = test_db_path();
        let store_a = store_for_profile(path.clone(), profile("test", "a", 3)).await;
        let id = store_a
            .insert(NewMemory {
                content: MemoryContent::new("alpha"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        let store_b = store_for_profile(path, profile("test", "b", 2)).await;
        assert_eq!(store_b.backfill_active_profile(10).await.unwrap(), 1);
        assert_eq!(
            store_b
                .search(&MemoryQuery {
                    text: "alpha".into(),
                    limit: 10,
                })
                .await
                .unwrap()
                .len(),
            1
        );

        store_a
            .put(IndexedMemory {
                index: id,
                content: MemoryContent::new("beta"),
                rank: MemoryRank::ShortTerm,
                occurred_at: None,
            })
            .await
            .unwrap();

        assert!(
            store_b
                .search(&MemoryQuery {
                    text: "beta".into(),
                    limit: 10,
                })
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(store_b.backfill_active_profile(10).await.unwrap(), 1);
        assert_eq!(
            store_b
                .search(&MemoryQuery {
                    text: "beta".into(),
                    limit: 10,
                })
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn profile_registry_rejects_conflicting_metadata() {
        let path = test_db_path();
        let profile = profile("test", "a", 3);
        let store = store_for_profile(path.clone(), profile.clone()).await;
        let profile_id = store.profile_id().to_owned();

        store
            .conn
            .execute(
                &format!(
                    r#"
                    UPDATE {PROFILE_REGISTRY_TABLE}
                    SET dimensions = ?2
                    WHERE profile_id = ?1
                    "#
                ),
                params![profile_id, 99_i64],
            )
            .await
            .unwrap();

        let result = LibsqlMemoryStore::connect(
            LibsqlMemoryStoreConfig::local(path, profile.dimensions).with_active_profile(profile),
            TestEmbedder::new(3),
        )
        .await;

        assert!(matches!(result, Err(PortError::InvalidData(_))));
    }

    #[tokio::test]
    async fn invalid_rank_data_is_reported() {
        let store = store().await;
        let id = MemoryIndex::new("bad-rank");
        store
            .conn
            .execute(
                &format!(
                    r#"
                    INSERT INTO {DEFAULT_MEMORY_TABLE} (
                      memory_index,
                      content,
                      rank,
                      created_at_ms,
                      updated_at_ms
                    )
                    VALUES (?1, ?2, ?3, ?4, ?5)
                    "#
                ),
                params![id.as_str(), "bad", 99_i64, now_ms(), now_ms()],
            )
            .await
            .unwrap();

        let error = store.get(&id).await.unwrap_err();
        assert!(matches!(error, PortError::InvalidData(_)));
    }

    #[tokio::test]
    async fn policy_insert_then_get_roundtrips_payload_and_metadata() {
        let store = policy_store().await;
        let id = store
            .insert(NewPolicy {
                trigger: "alpha situation".into(),
                behavior: "do the alpha behavior".into(),
                rank: PolicyRank::Established,
                expected_reward: SignedUnitF32::clamp(0.4),
                confidence: UnitF32::clamp(0.6),
                value: SignedUnitF32::clamp(0.5),
                reward_tokens: 7,
                decay_remaining_secs: 123,
            })
            .await
            .unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.index, id);
        assert_eq!(got.trigger, "alpha situation");
        assert_eq!(got.behavior, "do the alpha behavior");
        assert_eq!(got.rank, PolicyRank::Established);
        assert_eq!(got.expected_reward.get(), 0.4);
        assert_eq!(got.confidence.get(), 0.6);
        assert_eq!(got.value.get(), 0.5);
        assert_eq!(got.reward_tokens, 7);
        assert_eq!(got.decay_remaining_secs, 123);
    }

    #[tokio::test]
    async fn policy_search_uses_trigger_embedding_not_behavior_or_values() {
        let store = policy_store().await;
        let alpha = PolicyIndex::new("alpha-trigger");
        store
            .put(IndexedPolicy {
                index: alpha.clone(),
                trigger: "alpha".into(),
                behavior: "plain behavior".into(),
                rank: PolicyRank::Tentative,
                expected_reward: SignedUnitF32::clamp(-1.0),
                confidence: UnitF32::ZERO,
                value: SignedUnitF32::clamp(-1.0),
                reward_tokens: 0,
                decay_remaining_secs: 60,
            })
            .await
            .unwrap();
        store
            .put(IndexedPolicy {
                index: PolicyIndex::new("behavior-alpha-only"),
                trigger: "beta".into(),
                behavior: "alpha alpha alpha alpha alpha".into(),
                rank: PolicyRank::Habit,
                expected_reward: SignedUnitF32::clamp(1.0),
                confidence: UnitF32::ONE,
                value: SignedUnitF32::clamp(1.0),
                reward_tokens: 99,
                decay_remaining_secs: 60,
            })
            .await
            .unwrap();

        let hits = store
            .search(&PolicyQuery {
                trigger: "alpha".into(),
                limit: 2,
            })
            .await
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].policy.index, alpha);
    }

    #[tokio::test]
    async fn policy_search_filters_expired_non_core_policies() {
        let store = policy_store().await;
        store
            .put(IndexedPolicy {
                index: PolicyIndex::new("expired"),
                trigger: "alpha".into(),
                behavior: "expired behavior".into(),
                rank: PolicyRank::Established,
                expected_reward: SignedUnitF32::ZERO,
                confidence: UnitF32::ZERO,
                value: SignedUnitF32::ZERO,
                reward_tokens: 0,
                decay_remaining_secs: 0,
            })
            .await
            .unwrap();
        store
            .put(IndexedPolicy {
                index: PolicyIndex::new("core"),
                trigger: "alpha".into(),
                behavior: "core behavior".into(),
                rank: PolicyRank::Core,
                expected_reward: SignedUnitF32::ZERO,
                confidence: UnitF32::ZERO,
                value: SignedUnitF32::ZERO,
                reward_tokens: 0,
                decay_remaining_secs: 0,
            })
            .await
            .unwrap();

        let hits = store
            .search(&PolicyQuery {
                trigger: "alpha".into(),
                limit: 8,
            })
            .await
            .unwrap();

        assert_eq!(
            hits.into_iter()
                .map(|hit| hit.policy.index.as_str().to_owned())
                .collect::<Vec<_>>(),
            vec!["core"]
        );
    }

    #[tokio::test]
    async fn policy_reinforce_clamps_scalars_and_applies_rank_thresholds() {
        let store = policy_store().await;
        let id = PolicyIndex::new("reinforced");
        store
            .put(IndexedPolicy {
                index: id.clone(),
                trigger: "alpha".into(),
                behavior: "do alpha".into(),
                rank: PolicyRank::Tentative,
                expected_reward: SignedUnitF32::clamp(0.9),
                confidence: UnitF32::clamp(0.95),
                value: SignedUnitF32::clamp(0.1),
                reward_tokens: 15,
                decay_remaining_secs: 60,
            })
            .await
            .unwrap();

        let record = store.reinforce(&id, 2.0, 1, 2.0, 2.0).await.unwrap();

        assert_eq!(record.expected_reward.get(), 1.0);
        assert_eq!(record.confidence.get(), 1.0);
        assert_eq!(record.value.get(), 1.0);
        assert_eq!(record.reward_tokens, 16);
        assert_eq!(record.rank, PolicyRank::Habit);
    }

    #[tokio::test]
    async fn policy_reinforce_negative_outcome_lowers_value_without_reward_token() {
        let store = policy_store().await;
        let id = PolicyIndex::new("negative");
        store
            .put(IndexedPolicy {
                index: id.clone(),
                trigger: "alpha".into(),
                behavior: "do alpha".into(),
                rank: PolicyRank::Provisional,
                expected_reward: SignedUnitF32::clamp(0.2),
                confidence: UnitF32::clamp(0.6),
                value: SignedUnitF32::clamp(0.4),
                reward_tokens: 3,
                decay_remaining_secs: 60,
            })
            .await
            .unwrap();

        let record = store.reinforce(&id, -0.6, 0, -0.4, -0.1).await.unwrap();

        assert!((record.value.get() - -0.2).abs() < f32::EPSILON * 2.0);
        assert_eq!(record.expected_reward.get(), -0.2);
        assert_eq!(record.confidence.get(), 0.5);
        assert_eq!(record.reward_tokens, 3);
        assert_eq!(record.rank, PolicyRank::Tentative);
    }

    async fn store() -> LibsqlMemoryStore {
        LibsqlMemoryStore::connect(
            LibsqlMemoryStoreConfig::local(test_db_path(), 3),
            TestEmbedder::new(3),
        )
        .await
        .unwrap()
    }

    async fn policy_store() -> LibsqlPolicyStore {
        LibsqlPolicyStore::connect(
            LibsqlPolicyStoreConfig::local(test_db_path(), 3),
            TestEmbedder::new(3),
        )
        .await
        .unwrap()
    }

    async fn store_for_profile(path: PathBuf, profile: EmbeddingProfile) -> LibsqlMemoryStore {
        let dims = profile.dimensions;
        LibsqlMemoryStore::connect(
            LibsqlMemoryStoreConfig::local(path, dims).with_active_profile(profile),
            TestEmbedder::new(dims),
        )
        .await
        .unwrap()
    }

    fn profile(name: &str, version: &str, dimensions: usize) -> EmbeddingProfile {
        EmbeddingProfile::new(name, version, dimensions)
    }

    async fn count_rows(store: &LibsqlMemoryStore, table: &str) -> i64 {
        let mut rows = store
            .conn
            .query(&format!("SELECT COUNT(*) FROM {table}"), ())
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        row.get(0).unwrap()
    }

    async fn column_exists(store: &LibsqlMemoryStore, table: &str, column: &str) -> bool {
        let mut rows = store
            .conn
            .query(&format!("PRAGMA table_info({table})"), ())
            .await
            .unwrap();
        while let Some(row) = rows.next().await.unwrap() {
            let name: String = row.get(1).unwrap();
            if name == column {
                return true;
            }
        }
        false
    }

    async fn source_ids_for(store: &LibsqlMemoryStore, index: &MemoryIndex) -> Vec<String> {
        let mut rows = store
            .conn
            .query(
                &format!(
                    r#"
                    SELECT source_ids
                    FROM {DEFAULT_MEMORY_TABLE}
                    WHERE memory_index = ?1
                    "#
                ),
                [index.as_str()],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let source_ids: String = row.get(0).unwrap();
        serde_json::from_str(&source_ids).unwrap()
    }

    fn test_db_path() -> PathBuf {
        let dir = std::env::current_dir()
            .unwrap()
            .join(".tmp")
            .join("lutum-libsql-adapter-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("memory.db")
    }
}
