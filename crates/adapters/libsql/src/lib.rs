//! Local libSQL-backed memory adapter.
//!
//! Memory content is stored separately from embeddings. Each embedding
//! profile owns a dimension-specific table so one memory can carry multiple
//! embedding versions without losing libSQL's `F32_BLOB(N)` typing.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use libsql::{Connection, Transaction, Value, params, params_from_iter};
use nuillu_memory::{
    IndexedMemory, LinkedMemoryQuery, LinkedMemoryRecord, MemoryConcept, MemoryKind, MemoryLink,
    MemoryLinkDirection, MemoryLinkRelation, MemoryQuery, MemoryRecord, MemoryStore, MemoryTag,
    NewMemory, NewMemoryLink,
};
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

mod migrations;

use migrations::run_agent_migrations;

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
        format!("p{}", hex_string(&digest[..8]))
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
pub struct LibsqlAgentStoreConfig {
    pub path: PathBuf,
    pub memory_table_name: String,
    pub policy_table_name: String,
    pub memory_active_profile: EmbeddingProfile,
    pub policy_active_profile: EmbeddingProfile,
}

impl LibsqlAgentStoreConfig {
    pub fn local(
        path: impl Into<PathBuf>,
        memory_vector_dims: usize,
        policy_vector_dims: usize,
    ) -> Self {
        Self {
            path: path.into(),
            memory_table_name: DEFAULT_MEMORY_TABLE.to_owned(),
            policy_table_name: DEFAULT_POLICY_TABLE.to_owned(),
            memory_active_profile: EmbeddingProfile::default_for_dimensions(memory_vector_dims),
            policy_active_profile: EmbeddingProfile::default_for_dimensions(policy_vector_dims),
        }
    }

    pub fn with_memory_active_profile(mut self, active_profile: EmbeddingProfile) -> Self {
        self.memory_active_profile = active_profile;
        self
    }

    pub fn with_policy_active_profile(mut self, active_profile: EmbeddingProfile) -> Self {
        self.policy_active_profile = active_profile;
        self
    }
}

#[derive(Clone)]
pub struct LibsqlAgentStore {
    memory: LibsqlMemoryStore,
    policy: LibsqlPolicyStore,
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

impl LibsqlAgentStore {
    pub async fn connect(
        config: LibsqlAgentStoreConfig,
        memory_embedder: Box<dyn Embedder>,
        policy_embedder: Box<dyn Embedder>,
    ) -> Result<Self, PortError> {
        let database = libsql::Builder::new_local(config.path)
            .build()
            .await
            .map_err(map_libsql_error)?;
        let conn = database.connect().map_err(map_libsql_error)?;
        run_agent_migrations(&conn).await?;
        let memory = LibsqlMemoryStore::from_connection(
            conn.clone(),
            config.memory_table_name,
            config.memory_active_profile,
            memory_embedder,
        )
        .await?;
        let policy = LibsqlPolicyStore::from_connection(
            conn.clone(),
            config.policy_table_name,
            config.policy_active_profile,
            policy_embedder,
        )
        .await?;
        Ok(Self { memory, policy })
    }

    pub fn memory_store(&self) -> LibsqlMemoryStore {
        self.memory.clone()
    }

    pub fn policy_store(&self) -> LibsqlPolicyStore {
        self.policy.clone()
    }
}

impl LibsqlMemoryStore {
    async fn from_connection(
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
        store.initialize_profile_schema().await?;
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

    fn concepts_table_name(&self) -> String {
        format!("{}_concepts", self.table_name)
    }

    fn memory_concepts_table_name(&self) -> String {
        format!("{}_memory_concepts", self.table_name)
    }

    fn tags_table_name(&self) -> String {
        format!("{}_tags", self.table_name)
    }

    fn memory_tags_table_name(&self) -> String {
        format!("{}_memory_tags", self.table_name)
    }

    fn links_table_name(&self) -> String {
        format!("{}_links", self.table_name)
    }

    fn search_table_name(&self) -> String {
        format!("{}_search", self.table_name)
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

    async fn initialize_profile_schema(&self) -> Result<(), PortError> {
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

    #[allow(clippy::too_many_arguments)]
    async fn upsert_memory_tx(
        &self,
        tx: &Transaction,
        index: &MemoryIndex,
        content: &str,
        kind: MemoryKind,
        rank: MemoryRank,
        occurred_at: Option<chrono::DateTime<chrono::Utc>>,
        stored_at: chrono::DateTime<chrono::Utc>,
        affect_arousal: f32,
        valence: f32,
        emotion: &str,
        source_ids_json: Option<&str>,
        now: i64,
    ) -> Result<(i64, i64), PortError> {
        let sql = format!(
            r#"
            INSERT INTO {memories} (
              memory_index,
              content,
              kind,
              rank,
              occurred_at_ms,
              stored_at_ms,
              affect_arousal,
              valence,
              emotion,
              created_at_ms,
              updated_at_ms,
              source_ids,
              metadata_json,
              deleted_at_ms
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, NULL, NULL)
            ON CONFLICT(memory_index) DO UPDATE SET
              content = excluded.content,
              kind = excluded.kind,
              rank = excluded.rank,
              occurred_at_ms = excluded.occurred_at_ms,
              stored_at_ms = excluded.stored_at_ms,
              affect_arousal = excluded.affect_arousal,
              valence = excluded.valence,
              emotion = excluded.emotion,
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
                kind_to_i64(kind),
                rank_to_i64(rank),
                occurred_at.map(|at| at.timestamp_millis()),
                stored_at.timestamp_millis(),
                clamp_confidence(affect_arousal),
                clamp_signed_unit(valence),
                emotion.trim(),
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

    async fn memory_id_for_index_tx(
        &self,
        tx: &Transaction,
        index: &MemoryIndex,
    ) -> Result<Option<i64>, PortError> {
        let sql = format!(
            r#"
            SELECT id
            FROM {memories}
            WHERE memory_index = ?1
              AND deleted_at_ms IS NULL
            LIMIT 1
            "#,
            memories = self.table_name,
        );
        let mut rows = tx
            .query(&sql, [index.as_str()])
            .await
            .map_err(map_libsql_error)?;
        Ok(match rows.next().await.map_err(map_libsql_error)? {
            Some(row) => Some(row.get(0).map_err(map_libsql_error)?),
            None => None,
        })
    }

    async fn replace_sidecars_tx(
        &self,
        tx: &Transaction,
        memory_id: i64,
        content: &str,
        concepts: &[MemoryConcept],
        tags: &[MemoryTag],
    ) -> Result<(), PortError> {
        let memory_concepts = self.memory_concepts_table_name();
        let memory_tags = self.memory_tags_table_name();
        let search = self.search_table_name();

        tx.execute(
            &format!("DELETE FROM {memory_concepts} WHERE memory_id = ?1"),
            [memory_id],
        )
        .await
        .map_err(map_libsql_error)?;
        tx.execute(
            &format!("DELETE FROM {memory_tags} WHERE memory_id = ?1"),
            [memory_id],
        )
        .await
        .map_err(map_libsql_error)?;

        for concept in concepts {
            let Some(concept_id) = self.upsert_concept_tx(tx, concept).await? else {
                continue;
            };
            tx.execute(
                &format!(
                    r#"
                    INSERT OR REPLACE INTO {memory_concepts} (
                      memory_id,
                      concept_id,
                      mention_text,
                      confidence
                    )
                    VALUES (?1, ?2, ?3, ?4)
                    "#
                ),
                params![
                    memory_id,
                    concept_id,
                    concept.mention_text.as_deref(),
                    clamp_confidence(concept.confidence),
                ],
            )
            .await
            .map_err(map_libsql_error)?;
        }

        for tag in tags {
            let Some(tag_id) = self.upsert_tag_tx(tx, tag).await? else {
                continue;
            };
            tx.execute(
                &format!(
                    r#"
                    INSERT OR REPLACE INTO {memory_tags} (
                      memory_id,
                      tag_id,
                      confidence
                    )
                    VALUES (?1, ?2, ?3)
                    "#
                ),
                params![memory_id, tag_id, clamp_confidence(tag.confidence)],
            )
            .await
            .map_err(map_libsql_error)?;
        }

        let concept_text = concepts
            .iter()
            .map(|concept| concept.label.trim())
            .filter(|label| !label.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let tag_text = tags
            .iter()
            .map(|tag| format!("{}:{}", tag.namespace.trim(), tag.label.trim()))
            .filter(|tag| tag != ":")
            .collect::<Vec<_>>()
            .join(" ");
        tx.execute(
            &format!(
                r#"
                INSERT INTO {search} (
                  memory_id,
                  search_text,
                  concept_text,
                  tag_text
                )
                VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT(memory_id) DO UPDATE SET
                  search_text = excluded.search_text,
                  concept_text = excluded.concept_text,
                  tag_text = excluded.tag_text
                "#
            ),
            params![memory_id, content, concept_text, tag_text],
        )
        .await
        .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn upsert_concept_tx(
        &self,
        tx: &Transaction,
        concept: &MemoryConcept,
    ) -> Result<Option<i64>, PortError> {
        let label = concept.label.trim();
        if label.is_empty() {
            return Ok(None);
        }
        let normalized = normalize_label(label);
        let concepts = self.concepts_table_name();
        tx.execute(
            &format!(
                r#"
                INSERT INTO {concepts} (
                  canonical_label,
                  normalized_label,
                  loose_type
                )
                VALUES (?1, ?2, ?3)
                ON CONFLICT(normalized_label) DO UPDATE SET
                  canonical_label = {concepts}.canonical_label
                "#
            ),
            params![label, normalized.as_str(), concept.loose_type.as_deref()],
        )
        .await
        .map_err(map_libsql_error)?;
        let mut rows = tx
            .query(
                &format!("SELECT id FROM {concepts} WHERE normalized_label = ?1 LIMIT 1"),
                [normalized.as_str()],
            )
            .await
            .map_err(map_libsql_error)?;
        Ok(match rows.next().await.map_err(map_libsql_error)? {
            Some(row) => Some(row.get(0).map_err(map_libsql_error)?),
            None => None,
        })
    }

    async fn upsert_tag_tx(
        &self,
        tx: &Transaction,
        tag: &MemoryTag,
    ) -> Result<Option<i64>, PortError> {
        let label = tag.label.trim();
        if label.is_empty() {
            return Ok(None);
        }
        let namespace = if tag.namespace.trim().is_empty() {
            "operation"
        } else {
            tag.namespace.trim()
        };
        let normalized = normalize_label(label);
        let tags = self.tags_table_name();
        tx.execute(
            &format!(
                r#"
                INSERT INTO {tags} (
                  label,
                  normalized_label,
                  namespace
                )
                VALUES (?1, ?2, ?3)
                ON CONFLICT(namespace, normalized_label) DO UPDATE SET
                  label = {tags}.label
                "#
            ),
            params![label, normalized.as_str(), namespace],
        )
        .await
        .map_err(map_libsql_error)?;
        let mut rows = tx
            .query(
                &format!(
                    "SELECT id FROM {tags} WHERE namespace = ?1 AND normalized_label = ?2 LIMIT 1"
                ),
                params![namespace, normalized.as_str()],
            )
            .await
            .map_err(map_libsql_error)?;
        Ok(match rows.next().await.map_err(map_libsql_error)? {
            Some(row) => Some(row.get(0).map_err(map_libsql_error)?),
            None => None,
        })
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

    async fn memory_id_for_index(&self, index: &MemoryIndex) -> Result<Option<i64>, PortError> {
        let sql = format!(
            r#"
            SELECT id
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
        Ok(match rows.next().await.map_err(map_libsql_error)? {
            Some(row) => Some(row.get(0).map_err(map_libsql_error)?),
            None => None,
        })
    }

    async fn record_by_id(&self, id: i64) -> Result<Option<MemoryRecord>, PortError> {
        let sql = format!(
            r#"
            SELECT id, memory_index, content, kind, rank, occurred_at_ms, stored_at_ms,
                   affect_arousal, valence, emotion
            FROM {memories}
            WHERE id = ?1
              AND deleted_at_ms IS NULL
            LIMIT 1
            "#,
            memories = self.table_name,
        );
        let mut rows = self
            .conn
            .query(&sql, [id])
            .await
            .map_err(map_libsql_error)?;
        match rows.next().await.map_err(map_libsql_error)? {
            Some(row) => Ok(Some(self.row_to_record(&row).await?)),
            None => Ok(None),
        }
    }

    async fn upsert_link_tx(
        &self,
        tx: &Transaction,
        from_memory_id: i64,
        to_memory_id: i64,
        link: NewMemoryLink,
        updated_at_ms: i64,
    ) -> Result<(), PortError> {
        let links = self.links_table_name();
        tx.execute(
            &format!(
                r#"
                INSERT INTO {links} (
                  from_memory_id,
                  to_memory_id,
                  relation,
                  freeform_relation,
                  strength,
                  confidence,
                  updated_at_ms
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ON CONFLICT(
                  from_memory_id,
                  to_memory_id,
                  relation,
                  COALESCE(freeform_relation, '')
                ) DO UPDATE SET
                  strength = excluded.strength,
                  confidence = excluded.confidence,
                  updated_at_ms = excluded.updated_at_ms
                "#
            ),
            params![
                from_memory_id,
                to_memory_id,
                relation_to_i64(link.relation),
                link.freeform_relation.as_deref(),
                clamp_confidence(link.strength),
                clamp_confidence(link.confidence),
                updated_at_ms,
            ],
        )
        .await
        .map_err(map_libsql_error)?;
        Ok(())
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

    async fn hard_delete_memory_tx(
        &self,
        tx: &Transaction,
        memory_id: i64,
        _now: i64,
    ) -> Result<(), PortError> {
        for table in [
            self.memory_concepts_table_name(),
            self.memory_tags_table_name(),
            self.search_table_name(),
        ] {
            tx.execute(
                &format!("DELETE FROM {table} WHERE memory_id = ?1"),
                [memory_id],
            )
            .await
            .map_err(map_libsql_error)?;
        }
        let links = self.links_table_name();
        tx.execute(
            &format!("DELETE FROM {links} WHERE from_memory_id = ?1 OR to_memory_id = ?1"),
            [memory_id],
        )
        .await
        .map_err(map_libsql_error)?;

        let mut rows = tx
            .query(
                &format!(
                    r#"
                    SELECT table_name
                    FROM {PROFILE_REGISTRY_TABLE}
                    WHERE table_name LIKE ?1
                    "#
                ),
                [format!("{}_embeddings_%", self.table_name)],
            )
            .await
            .map_err(map_libsql_error)?;
        let mut embedding_tables = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            let table: String = row.get(0).map_err(map_libsql_error)?;
            embedding_tables.push(table);
        }
        drop(rows);
        for table in embedding_tables {
            validate_identifier("embedding table name", &table)?;
            tx.execute(
                &format!("DELETE FROM {table} WHERE memory_id = ?1"),
                [memory_id],
            )
            .await
            .map_err(map_libsql_error)?;
        }

        tx.execute(
            &format!("DELETE FROM {} WHERE id = ?1", self.table_name),
            [memory_id],
        )
        .await
        .map_err(map_libsql_error)?;
        Ok(())
    }

    async fn row_to_record(&self, row: &libsql::Row) -> Result<MemoryRecord, PortError> {
        let id: i64 = row.get(0).map_err(map_libsql_error)?;
        let index: String = row.get(1).map_err(map_libsql_error)?;
        let content: String = row.get(2).map_err(map_libsql_error)?;
        let kind: i64 = row.get(3).map_err(map_libsql_error)?;
        let rank: i64 = row.get(4).map_err(map_libsql_error)?;
        let occurred_at_ms: Option<i64> = row.get(5).map_err(map_libsql_error)?;
        let stored_at_ms: i64 = row.get(6).map_err(map_libsql_error)?;
        let affect_arousal: f64 = row.get(7).map_err(map_libsql_error)?;
        let valence: f64 = row.get(8).map_err(map_libsql_error)?;
        let emotion: String = row.get(9).map_err(map_libsql_error)?;
        Ok(MemoryRecord {
            index: MemoryIndex::new(index),
            content: MemoryContent::new(content),
            rank: rank_from_i64(rank)?,
            occurred_at: occurred_at_ms.and_then(chrono::DateTime::from_timestamp_millis),
            stored_at: chrono::DateTime::from_timestamp_millis(stored_at_ms).ok_or_else(|| {
                PortError::InvalidData(format!(
                    "invalid memory stored_at timestamp: {stored_at_ms}"
                ))
            })?,
            kind: kind_from_i64(kind)?,
            concepts: self.concepts_for_memory(id).await?,
            tags: self.tags_for_memory(id).await?,
            affect_arousal: clamp_confidence(affect_arousal as f32),
            valence: clamp_signed_unit(valence as f32),
            emotion,
        })
    }

    async fn concepts_for_memory(&self, memory_id: i64) -> Result<Vec<MemoryConcept>, PortError> {
        let concepts = self.concepts_table_name();
        let memory_concepts = self.memory_concepts_table_name();
        let sql = format!(
            r#"
            SELECT c.canonical_label, mc.mention_text, c.loose_type, mc.confidence
            FROM {memory_concepts} AS mc
            JOIN {concepts} AS c ON c.id = mc.concept_id
            WHERE mc.memory_id = ?1
            ORDER BY c.canonical_label ASC, mc.mention_text ASC
            "#,
        );
        let mut rows = self
            .conn
            .query(&sql, [memory_id])
            .await
            .map_err(map_libsql_error)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            let confidence: f64 = row.get(3).map_err(map_libsql_error)?;
            out.push(MemoryConcept {
                label: row.get(0).map_err(map_libsql_error)?,
                mention_text: row.get(1).map_err(map_libsql_error)?,
                loose_type: row.get(2).map_err(map_libsql_error)?,
                confidence: confidence as f32,
            });
        }
        Ok(out)
    }

    async fn tags_for_memory(&self, memory_id: i64) -> Result<Vec<MemoryTag>, PortError> {
        let tags = self.tags_table_name();
        let memory_tags = self.memory_tags_table_name();
        let sql = format!(
            r#"
            SELECT t.label, t.namespace, mt.confidence
            FROM {memory_tags} AS mt
            JOIN {tags} AS t ON t.id = mt.tag_id
            WHERE mt.memory_id = ?1
            ORDER BY t.namespace ASC, t.label ASC
            "#,
        );
        let mut rows = self
            .conn
            .query(&sql, [memory_id])
            .await
            .map_err(map_libsql_error)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            let confidence: f64 = row.get(2).map_err(map_libsql_error)?;
            out.push(MemoryTag {
                label: row.get(0).map_err(map_libsql_error)?,
                namespace: row.get(1).map_err(map_libsql_error)?,
                confidence: confidence as f32,
            });
        }
        Ok(out)
    }

    async fn put_indexed(
        &self,
        mem: IndexedMemory,
        source_ids_json: Option<String>,
    ) -> Result<MemoryRecord, PortError> {
        let embedding_json = self.embed_json(mem.content.as_str()).await?;
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let (memory_id, updated_at_ms) = self
            .upsert_memory_tx(
                &tx,
                &mem.index,
                mem.content.as_str(),
                mem.kind,
                mem.rank,
                mem.occurred_at,
                mem.stored_at,
                mem.affect_arousal,
                mem.valence,
                &mem.emotion,
                source_ids_json.as_deref(),
                now,
            )
            .await?;
        self.replace_sidecars_tx(
            &tx,
            memory_id,
            mem.content.as_str(),
            &mem.concepts,
            &mem.tags,
        )
        .await?;
        self.upsert_embedding_tx(&tx, memory_id, &embedding_json, now, updated_at_ms)
            .await?;
        tx.commit().await.map_err(map_libsql_error)?;
        self.get(&mem.index).await?.ok_or_else(|| {
            PortError::Backend(format!(
                "memory row was not found after put: {}",
                mem.index.as_str()
            ))
        })
    }
}

impl LibsqlPolicyStore {
    async fn from_connection(
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
        store.initialize_profile_schema().await?;
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

    async fn initialize_profile_schema(&self) -> Result<(), PortError> {
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
    async fn insert(
        &self,
        mem: NewMemory,
        stored_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<MemoryRecord, PortError> {
        let index = MemoryIndex::new(Uuid::now_v7().to_string());
        let embedding_json = self.embed_json(mem.content.as_str()).await?;
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let (memory_id, updated_at_ms) = self
            .upsert_memory_tx(
                &tx,
                &index,
                mem.content.as_str(),
                mem.kind,
                mem.rank,
                mem.occurred_at,
                stored_at,
                mem.affect_arousal,
                mem.valence,
                &mem.emotion,
                None,
                now,
            )
            .await?;
        self.replace_sidecars_tx(
            &tx,
            memory_id,
            mem.content.as_str(),
            &mem.concepts,
            &mem.tags,
        )
        .await?;
        self.upsert_embedding_tx(&tx, memory_id, &embedding_json, now, updated_at_ms)
            .await?;
        tx.commit().await.map_err(map_libsql_error)?;
        self.get(&index).await?.ok_or_else(|| {
            PortError::Backend(format!(
                "memory row was not found after insert: {}",
                index.as_str()
            ))
        })
    }

    async fn put(&self, mem: IndexedMemory) -> Result<MemoryRecord, PortError> {
        self.put_indexed(mem, None).await
    }

    async fn compact(
        &self,
        mem: NewMemory,
        sources: &[MemoryIndex],
        stored_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<MemoryRecord, PortError> {
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
                mem.kind,
                mem.rank,
                mem.occurred_at,
                stored_at,
                mem.affect_arousal,
                mem.valence,
                &mem.emotion,
                Some(&source_ids_json),
                now,
            )
            .await?;
        self.replace_sidecars_tx(
            &tx,
            memory_id,
            mem.content.as_str(),
            &mem.concepts,
            &mem.tags,
        )
        .await?;
        self.upsert_embedding_tx(&tx, memory_id, &embedding_json, now, updated_at_ms)
            .await?;
        self.soft_delete_many_tx(&tx, sources, now).await?;
        tx.commit().await.map_err(map_libsql_error)?;
        self.get(&index).await?.ok_or_else(|| {
            PortError::Backend(format!(
                "memory row was not found after compact: {}",
                index.as_str()
            ))
        })
    }

    async fn put_compacted(
        &self,
        mem: IndexedMemory,
        sources: &[MemoryIndex],
    ) -> Result<MemoryRecord, PortError> {
        let source_ids_json = source_ids_json(sources)?;
        let embedding_json = self.embed_json(mem.content.as_str()).await?;
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let (memory_id, updated_at_ms) = self
            .upsert_memory_tx(
                &tx,
                &mem.index,
                mem.content.as_str(),
                mem.kind,
                mem.rank,
                mem.occurred_at,
                mem.stored_at,
                mem.affect_arousal,
                mem.valence,
                &mem.emotion,
                Some(&source_ids_json),
                now,
            )
            .await?;
        self.replace_sidecars_tx(
            &tx,
            memory_id,
            mem.content.as_str(),
            &mem.concepts,
            &mem.tags,
        )
        .await?;
        self.upsert_embedding_tx(&tx, memory_id, &embedding_json, now, updated_at_ms)
            .await?;
        self.soft_delete_many_tx(&tx, sources, now).await?;
        tx.commit().await.map_err(map_libsql_error)?;
        self.get(&mem.index).await?.ok_or_else(|| {
            PortError::Backend(format!(
                "memory row was not found after put compacted: {}",
                mem.index.as_str()
            ))
        })
    }

    async fn get(&self, index: &MemoryIndex) -> Result<Option<MemoryRecord>, PortError> {
        let sql = format!(
            r#"
            SELECT id, memory_index, content, kind, rank, occurred_at_ms, stored_at_ms,
                   affect_arousal, valence, emotion
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
            Some(row) => Ok(Some(self.row_to_record(&row).await?)),
            None => Ok(None),
        }
    }

    async fn list_by_rank(&self, rank: MemoryRank) -> Result<Vec<MemoryRecord>, PortError> {
        let sql = format!(
            r#"
            SELECT id, memory_index, content, kind, rank, occurred_at_ms, stored_at_ms,
                   affect_arousal, valence, emotion
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
            out.push(self.row_to_record(&row).await?);
        }
        Ok(out)
    }

    async fn search(&self, q: &MemoryQuery) -> Result<Vec<MemoryRecord>, PortError> {
        if q.limit == 0 {
            return Ok(Vec::new());
        }

        let text = q.text.trim();
        let mut params = Vec::<Value>::new();
        let mut where_clauses = vec!["m.deleted_at_ms IS NULL".to_owned()];
        let embedding_param = if text.is_empty() {
            None
        } else {
            Some(push_query_param(
                &mut params,
                Value::Text(self.embed_json(text).await?),
            ))
        };

        if embedding_param.is_some() {
            where_clauses.push("e.content_updated_at_ms = m.updated_at_ms".to_owned());
        }

        let kind_placeholders = q
            .kinds
            .iter()
            .map(|kind| push_query_param(&mut params, Value::Integer(kind_to_i64(*kind))))
            .collect::<Vec<_>>();
        if !kind_placeholders.is_empty() {
            where_clauses.push(format!("m.kind IN ({})", kind_placeholders.join(", ")));
        }

        for concept in q
            .concepts
            .iter()
            .map(|concept| normalize_label(concept))
            .filter(|concept| !concept.is_empty())
        {
            let placeholder = push_query_param(&mut params, Value::Text(concept));
            where_clauses.push(format!(
                r#"
                EXISTS (
                  SELECT 1
                  FROM {memory_concepts} AS mc
                  JOIN {concepts} AS c ON c.id = mc.concept_id
                  WHERE mc.memory_id = m.id
                    AND c.normalized_label = {placeholder}
                )
                "#,
                memory_concepts = self.memory_concepts_table_name(),
                concepts = self.concepts_table_name(),
            ));
        }

        for (namespace, label) in q.tags.iter().filter_map(|tag| normalized_tag_filter(tag)) {
            let memory_tags = self.memory_tags_table_name();
            let tags = self.tags_table_name();
            match namespace {
                Some(namespace) => {
                    let namespace_placeholder =
                        push_query_param(&mut params, Value::Text(namespace));
                    let label_placeholder = push_query_param(&mut params, Value::Text(label));
                    where_clauses.push(format!(
                        r#"
                        EXISTS (
                          SELECT 1
                          FROM {memory_tags} AS mt
                          JOIN {tags} AS t ON t.id = mt.tag_id
                          WHERE mt.memory_id = m.id
                            AND LOWER(t.namespace) = {namespace_placeholder}
                            AND t.normalized_label = {label_placeholder}
                        )
                        "#,
                    ));
                }
                None => {
                    let label_placeholder = push_query_param(&mut params, Value::Text(label));
                    where_clauses.push(format!(
                        r#"
                        EXISTS (
                          SELECT 1
                          FROM {memory_tags} AS mt
                          JOIN {tags} AS t ON t.id = mt.tag_id
                          WHERE mt.memory_id = m.id
                            AND t.normalized_label = {label_placeholder}
                        )
                        "#,
                    ));
                }
            }
        }

        let join_clause = if embedding_param.is_some() {
            format!(
                "JOIN {embeddings} AS e ON e.memory_id = m.id",
                embeddings = self.embedding_table_name,
            )
        } else {
            String::new()
        };
        let order_clause = if let Some(placeholder) = embedding_param {
            format!("vector_distance_cos(e.embedding, vector32({placeholder})) ASC, m.id ASC")
        } else {
            "m.stored_at_ms DESC, m.id ASC".to_owned()
        };
        let limit_placeholder =
            push_query_param(&mut params, Value::Integer(limit_to_i64(q.limit)));
        let sql = format!(
            r#"
            SELECT m.id, m.memory_index, m.content, m.kind, m.rank, m.occurred_at_ms, m.stored_at_ms,
                   m.affect_arousal, m.valence, m.emotion
            FROM {memories} AS m
            {join_clause}
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT {limit_placeholder}
            "#,
            memories = self.table_name,
            where_clause = where_clauses.join("\n              AND "),
        );

        let mut rows = self
            .conn
            .query(&sql, params_from_iter(params))
            .await
            .map_err(map_libsql_error)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            out.push(self.row_to_record(&row).await?);
        }

        Ok(out)
    }

    async fn linked(&self, q: &LinkedMemoryQuery) -> Result<Vec<LinkedMemoryRecord>, PortError> {
        if q.limit == 0 || q.memory_indexes.is_empty() {
            return Ok(Vec::new());
        }
        let links = self.links_table_name();
        let memories = &self.table_name;
        let mut out = Vec::new();
        let mut seen =
            std::collections::HashSet::<(String, String, MemoryLinkRelation, String)>::new();
        for index in &q.memory_indexes {
            let root_id = self.memory_id_for_index(index).await?;
            let Some(root_id) = root_id else {
                continue;
            };
            let sql = format!(
                r#"
                SELECT
                  l.from_memory_id,
                  from_m.memory_index,
                  l.to_memory_id,
                  to_m.memory_index,
                  l.relation,
                  l.freeform_relation,
                  l.strength,
                  l.confidence,
                  l.updated_at_ms,
                  CASE WHEN l.from_memory_id = ?1 THEN l.to_memory_id ELSE l.from_memory_id END AS linked_id
                FROM {links} AS l
                JOIN {memories} AS from_m ON from_m.id = l.from_memory_id
                JOIN {memories} AS to_m ON to_m.id = l.to_memory_id
                WHERE (l.from_memory_id = ?1 OR l.to_memory_id = ?1)
                  AND from_m.deleted_at_ms IS NULL
                  AND to_m.deleted_at_ms IS NULL
                ORDER BY l.updated_at_ms DESC, l.id ASC
                "#,
            );
            let mut rows = self
                .conn
                .query(&sql, [root_id])
                .await
                .map_err(map_libsql_error)?;
            while let Some(row) = rows.next().await.map_err(map_libsql_error)? {
                let from_id: i64 = row.get(0).map_err(map_libsql_error)?;
                let from_index: String = row.get(1).map_err(map_libsql_error)?;
                let to_index: String = row.get(3).map_err(map_libsql_error)?;
                let relation = relation_from_i64(row.get(4).map_err(map_libsql_error)?)?;
                let freeform_relation: Option<String> = row.get(5).map_err(map_libsql_error)?;
                if !q.relation_filter.is_empty() && !q.relation_filter.contains(&relation) {
                    continue;
                }
                let outgoing = from_id == root_id;
                if !matches!(q.direction, MemoryLinkDirection::Both)
                    && !matches!(
                        (q.direction, outgoing),
                        (MemoryLinkDirection::Outgoing, true)
                            | (MemoryLinkDirection::Incoming, false)
                    )
                {
                    continue;
                }
                let key = (
                    from_index.clone(),
                    to_index.clone(),
                    relation,
                    freeform_relation.clone().unwrap_or_default(),
                );
                if !seen.insert(key) {
                    continue;
                }
                let linked_id: i64 = row.get(9).map_err(map_libsql_error)?;
                let Some(record) = self.record_by_id(linked_id).await? else {
                    continue;
                };
                let updated_at_ms: i64 = row.get(8).map_err(map_libsql_error)?;
                let strength: f64 = row.get(6).map_err(map_libsql_error)?;
                let confidence: f64 = row.get(7).map_err(map_libsql_error)?;
                out.push(LinkedMemoryRecord {
                    record,
                    link: MemoryLink {
                        from_memory: MemoryIndex::new(from_index),
                        to_memory: MemoryIndex::new(to_index),
                        relation,
                        freeform_relation,
                        strength: strength as f32,
                        confidence: confidence as f32,
                        updated_at: chrono::DateTime::from_timestamp_millis(updated_at_ms)
                            .ok_or_else(|| {
                                PortError::InvalidData(format!(
                                    "invalid memory link timestamp: {updated_at_ms}"
                                ))
                            })?,
                    },
                });
                if out.len() >= q.limit {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }

    async fn upsert_link(
        &self,
        link: NewMemoryLink,
        updated_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<MemoryLink, PortError> {
        let updated_at_ms = updated_at.timestamp_millis();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let from_id = self
            .memory_id_for_index_tx(&tx, &link.from_memory)
            .await?
            .ok_or_else(|| PortError::NotFound(link.from_memory.to_string()))?;
        let to_id = self
            .memory_id_for_index_tx(&tx, &link.to_memory)
            .await?
            .ok_or_else(|| PortError::NotFound(link.to_memory.to_string()))?;
        self.upsert_link_tx(&tx, from_id, to_id, link.clone(), updated_at_ms)
            .await?;
        tx.commit().await.map_err(map_libsql_error)?;
        Ok(MemoryLink {
            from_memory: link.from_memory,
            to_memory: link.to_memory,
            relation: link.relation,
            freeform_relation: link.freeform_relation,
            strength: clamp_confidence(link.strength),
            confidence: clamp_confidence(link.confidence),
            updated_at,
        })
    }

    async fn delete(&self, index: &MemoryIndex) -> Result<(), PortError> {
        let now = now_ms();
        let tx = self.conn.transaction().await.map_err(map_libsql_error)?;
        let Some(memory_id) = self.memory_id_for_index_tx(&tx, index).await? else {
            tx.commit().await.map_err(map_libsql_error)?;
            return Ok(());
        };
        self.hard_delete_memory_tx(&tx, memory_id, now).await?;
        tx.commit().await.map_err(map_libsql_error)?;
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

fn kind_to_i64(kind: MemoryKind) -> i64 {
    match kind {
        MemoryKind::Episode => 0,
        MemoryKind::Statement => 1,
        MemoryKind::Reflection => 2,
        MemoryKind::Hypothesis => 3,
        MemoryKind::Dream => 4,
        MemoryKind::Procedure => 5,
        MemoryKind::Plan => 6,
    }
}

fn kind_from_i64(value: i64) -> Result<MemoryKind, PortError> {
    match value {
        0 => Ok(MemoryKind::Episode),
        1 => Ok(MemoryKind::Statement),
        2 => Ok(MemoryKind::Reflection),
        3 => Ok(MemoryKind::Hypothesis),
        4 => Ok(MemoryKind::Dream),
        5 => Ok(MemoryKind::Procedure),
        6 => Ok(MemoryKind::Plan),
        _ => Err(PortError::InvalidData(format!(
            "invalid memory kind: {value}"
        ))),
    }
}

fn relation_to_i64(relation: MemoryLinkRelation) -> i64 {
    match relation {
        MemoryLinkRelation::Related => 0,
        MemoryLinkRelation::Supports => 1,
        MemoryLinkRelation::Contradicts => 2,
        MemoryLinkRelation::Updates => 3,
        MemoryLinkRelation::Corrects => 4,
        MemoryLinkRelation::DerivedFrom => 5,
    }
}

fn relation_from_i64(value: i64) -> Result<MemoryLinkRelation, PortError> {
    match value {
        0 => Ok(MemoryLinkRelation::Related),
        1 => Ok(MemoryLinkRelation::Supports),
        2 => Ok(MemoryLinkRelation::Contradicts),
        3 => Ok(MemoryLinkRelation::Updates),
        4 => Ok(MemoryLinkRelation::Corrects),
        5 => Ok(MemoryLinkRelation::DerivedFrom),
        _ => Err(PortError::InvalidData(format!(
            "invalid memory link relation: {value}"
        ))),
    }
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

fn normalize_label(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn push_query_param(params: &mut Vec<Value>, value: Value) -> String {
    params.push(value);
    format!("?{}", params.len())
}

fn normalized_tag_filter(value: &str) -> Option<(Option<String>, String)> {
    let normalized = normalize_label(value);
    if normalized.is_empty() {
        return None;
    }
    let Some((namespace, label)) = normalized.split_once(':') else {
        return Some((None, normalized));
    };
    let label = label.trim();
    if label.is_empty() {
        return None;
    }
    let namespace = namespace.trim();
    Some((
        (!namespace.is_empty()).then(|| namespace.to_owned()),
        label.to_owned(),
    ))
}

fn clamp_confidence(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn clamp_signed_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
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

fn hex_string(bytes: &[u8]) -> String {
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
    use crate::migrations::{
        BridgeMigration, CURRENT_RELEASED_MIGRATIONS, CURRENT_SCHEMA_MAJOR, CURRENT_SCHEMA_MINOR,
        CURRENT_SCHEMA_SNAPSHOT_SQL, DevMigration, MIGRATION_METADATA_DDL, MigrationBundle,
        ReleasedMigration, SCHEMA_DEV_TASK_TABLE, SCHEMA_FAMILY, SCHEMA_VERSION_TABLE,
        SchemaVersion, run_migration_bundle,
    };
    use chrono::TimeZone as _;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug)]
    struct TestEmbedder {
        dims: usize,
    }

    impl TestEmbedder {
        fn boxed(dims: usize) -> Box<dyn Embedder> {
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
        let policies = count_rows(&store, DEFAULT_POLICY_TABLE).await;
        let version = schema_version_for_conn(&store.conn).await;

        assert_eq!(profiles, 1);
        assert_eq!(memories, 0);
        assert_eq!(policies, 0);
        assert_eq!(
            version,
            SchemaVersion {
                major: CURRENT_SCHEMA_MAJOR,
                minor: CURRENT_SCHEMA_MINOR
            }
        );
    }

    #[tokio::test]
    async fn agent_store_uses_one_database_for_memory_and_policy() {
        let path = test_db_path();
        let agent = LibsqlAgentStore::connect(
            LibsqlAgentStoreConfig::local(path.clone(), 3, 3)
                .with_memory_active_profile(profile("memory", "v1", 3))
                .with_policy_active_profile(profile("policy", "v1", 3)),
            TestEmbedder::boxed(3),
            TestEmbedder::boxed(3),
        )
        .await
        .unwrap();

        let memory = agent.memory_store();
        let policy = agent.policy_store();
        assert!(table_exists(&memory.conn, DEFAULT_MEMORY_TABLE).await);
        assert!(table_exists(&policy.conn, DEFAULT_POLICY_TABLE).await);
        assert!(path.exists());
    }

    #[tokio::test]
    async fn connect_rejects_dimension_mismatch() {
        let result = LibsqlAgentStore::connect(
            LibsqlAgentStoreConfig::local(test_db_path(), 2, 3),
            TestEmbedder::boxed(3),
            TestEmbedder::boxed(3),
        )
        .await;

        assert!(matches!(result, Err(PortError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn insert_then_get_roundtrips_content_and_rank() {
        let store = store().await;
        let occurred_at = chrono::Utc.with_ymd_and_hms(2025, 5, 10, 0, 0, 0).unwrap();
        let stored_at = chrono::Utc.with_ymd_and_hms(2025, 5, 11, 1, 2, 3).unwrap();
        let id = store
            .insert(
                NewMemory {
                    affect_arousal: 0.72,
                    valence: -0.25,
                    emotion: "focused concern".to_owned(),
                    ..new_memory("alpha beta", MemoryRank::LongTerm, Some(occurred_at))
                },
                stored_at,
            )
            .await
            .unwrap()
            .index;

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.index, id);
        assert_eq!(got.content.as_str(), "alpha beta");
        assert_eq!(got.rank, MemoryRank::LongTerm);
        assert_eq!(got.occurred_at, Some(occurred_at));
        assert_eq!(got.stored_at, stored_at);
        assert_eq!(got.kind, MemoryKind::Statement);
        assert_eq!(got.affect_arousal, 0.72);
        assert_eq!(got.valence, -0.25);
        assert_eq!(got.emotion, "focused concern");
    }

    #[tokio::test]
    async fn insert_stores_sidecars_and_search_filters() {
        let store = store().await;
        let id = store
            .insert(
                NewMemory {
                    kind: MemoryKind::Episode,
                    concepts: vec![MemoryConcept {
                        label: "Ryo".to_string(),
                        mention_text: Some("Ryo".to_string()),
                        loose_type: Some("person".to_string()),
                        confidence: 0.9,
                    }],
                    tags: vec![MemoryTag {
                        label: "preference".to_string(),
                        namespace: "operation".to_string(),
                        confidence: 0.8,
                    }],
                    ..new_memory(
                        "Ryo said he prefers Kyoto coffee shops.",
                        MemoryRank::LongTerm,
                        None,
                    )
                },
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.kind, MemoryKind::Episode);
        assert_eq!(got.concepts[0].label, "Ryo");
        assert_eq!(got.tags[0].label, "preference");

        let hits = store
            .search(&MemoryQuery {
                text: String::new(),
                limit: 10,
                kinds: vec![MemoryKind::Episode],
                concepts: vec!["ryo".to_string()],
                tags: vec!["preference".to_string()],
            })
            .await
            .unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.index.clone()).collect::<Vec<_>>(),
            vec![id]
        );
    }

    #[tokio::test]
    async fn search_filters_are_applied_before_vector_limit() {
        let store = store().await;
        for i in 0..10 {
            store
                .put(indexed_memory(
                    MemoryIndex::new(format!("non-target-{i}")),
                    "alpha",
                    MemoryRank::LongTerm,
                ))
                .await
                .unwrap();
        }
        let target = MemoryIndex::new("target");
        store
            .put(IndexedMemory {
                concepts: vec![MemoryConcept::new("needle")],
                ..indexed_memory(target.clone(), "alpha", MemoryRank::LongTerm)
            })
            .await
            .unwrap();

        let hits = store
            .search(&MemoryQuery {
                text: "alpha".to_string(),
                limit: 1,
                kinds: Vec::new(),
                concepts: vec!["needle".to_string()],
                tags: Vec::new(),
            })
            .await
            .unwrap();

        assert_eq!(
            hits.iter().map(|hit| hit.index.clone()).collect::<Vec<_>>(),
            vec![target]
        );
    }

    #[tokio::test]
    async fn search_tag_namespace_filter_is_sql_side() {
        let store = store().await;
        for i in 0..10 {
            store
                .put(indexed_memory(
                    MemoryIndex::new(format!("non-target-tag-{i}")),
                    "alpha",
                    MemoryRank::LongTerm,
                ))
                .await
                .unwrap();
        }
        let target = MemoryIndex::new("target-tag");
        store
            .put(IndexedMemory {
                tags: vec![MemoryTag {
                    label: "preference".to_string(),
                    namespace: "operation".to_string(),
                    confidence: 1.0,
                }],
                ..indexed_memory(target.clone(), "alpha", MemoryRank::LongTerm)
            })
            .await
            .unwrap();

        let hits = store
            .search(&MemoryQuery {
                text: "alpha".to_string(),
                limit: 1,
                kinds: Vec::new(),
                concepts: Vec::new(),
                tags: vec!["operation:preference".to_string()],
            })
            .await
            .unwrap();

        assert_eq!(
            hits.iter().map(|hit| hit.index.clone()).collect::<Vec<_>>(),
            vec![target]
        );
    }

    #[tokio::test]
    async fn initial_dev_migration_contains_current_memory_columns() {
        let store = store().await;

        assert!(column_exists(&store, DEFAULT_MEMORY_TABLE, "occurred_at_ms").await);
        assert!(column_exists(&store, DEFAULT_MEMORY_TABLE, "stored_at_ms").await);
        assert!(column_exists(&store, DEFAULT_MEMORY_TABLE, "kind").await);
        assert!(column_exists(&store, DEFAULT_MEMORY_TABLE, "affect_arousal").await);
        assert!(column_exists(&store, DEFAULT_MEMORY_TABLE, "valence").await);
        assert!(column_exists(&store, DEFAULT_MEMORY_TABLE, "emotion").await);
        assert_eq!(dev_task_count(&store.conn, 0, 1).await, 1);
    }

    #[tokio::test]
    async fn dev_migrations_apply_in_order_and_skip_matching_checksums() {
        const DEV_A: &str = "CREATE TABLE IF NOT EXISTS dev_a(id INTEGER);";
        const DEV_B: &str = "CREATE TABLE IF NOT EXISTS dev_b(id INTEGER);";
        const DEV_TASKS: &[DevMigration] = &[
            DevMigration {
                major: 0,
                minor: 2,
                task_tag: "first-task",
                sql: DEV_A,
            },
            DevMigration {
                major: 0,
                minor: 2,
                task_tag: "second-task",
                sql: DEV_B,
            },
        ];
        let bundle = MigrationBundle {
            current: SchemaVersion { major: 0, minor: 1 },
            snapshot_sql: CURRENT_SCHEMA_SNAPSHOT_SQL,
            released: CURRENT_RELEASED_MIGRATIONS,
            dev: DEV_TASKS,
            bridge: None,
        };
        let conn = open_test_conn(test_db_path()).await;

        run_migration_bundle(&conn, bundle).await.unwrap();
        run_migration_bundle(&conn, bundle).await.unwrap();

        assert!(table_exists(&conn, "dev_a").await);
        assert!(table_exists(&conn, "dev_b").await);
        assert_eq!(dev_task_count(&conn, 0, 2).await, 2);
    }

    #[tokio::test]
    async fn dev_migration_checksum_drift_is_rejected() {
        const DEV_A: &str = "CREATE TABLE IF NOT EXISTS dev_checksum_a(id INTEGER);";
        const DEV_B: &str = "CREATE TABLE IF NOT EXISTS dev_checksum_b(id INTEGER);";
        const DEV_TASKS_A: &[DevMigration] = &[DevMigration {
            major: 0,
            minor: 2,
            task_tag: "checksum-task",
            sql: DEV_A,
        }];
        const DEV_TASKS_B: &[DevMigration] = &[DevMigration {
            major: 0,
            minor: 2,
            task_tag: "checksum-task",
            sql: DEV_B,
        }];
        let conn = open_test_conn(test_db_path()).await;
        let bundle_a = MigrationBundle {
            current: SchemaVersion { major: 0, minor: 1 },
            snapshot_sql: CURRENT_SCHEMA_SNAPSHOT_SQL,
            released: CURRENT_RELEASED_MIGRATIONS,
            dev: DEV_TASKS_A,
            bridge: None,
        };
        let bundle_b = MigrationBundle {
            dev: DEV_TASKS_B,
            ..bundle_a
        };

        run_migration_bundle(&conn, bundle_a).await.unwrap();
        let error = run_migration_bundle(&conn, bundle_b).await.unwrap_err();

        assert!(matches!(error, PortError::InvalidData(_)));
    }

    #[tokio::test]
    async fn bridge_migration_upgrades_previous_terminal_then_current_minor() {
        const BRIDGE_SQL: &str = "CREATE TABLE IF NOT EXISTS bridge_marker(id INTEGER);";
        const RELEASE_SQL: &str = "CREATE TABLE IF NOT EXISTS release_marker(id INTEGER);";
        const RELEASED: &[ReleasedMigration] = &[ReleasedMigration {
            from: SchemaVersion { major: 1, minor: 0 },
            to: SchemaVersion { major: 1, minor: 1 },
            sql: RELEASE_SQL,
        }];
        let bundle = MigrationBundle {
            current: SchemaVersion { major: 1, minor: 1 },
            snapshot_sql: "CREATE TABLE IF NOT EXISTS current_marker(id INTEGER);",
            released: RELEASED,
            dev: &[],
            bridge: Some(BridgeMigration {
                from: SchemaVersion { major: 0, minor: 1 },
                to: SchemaVersion { major: 1, minor: 0 },
                sql: BRIDGE_SQL,
            }),
        };
        let conn = open_test_conn(test_db_path()).await;
        seed_schema_version(&conn, SchemaVersion { major: 0, minor: 1 }).await;

        run_migration_bundle(&conn, bundle).await.unwrap();

        assert_eq!(schema_version_for_conn(&conn).await, bundle.current);
        assert_eq!(
            bridge_audit_for_conn(&conn).await,
            Some(SchemaVersion { major: 0, minor: 1 })
        );
        assert!(table_exists(&conn, "bridge_marker").await);
        assert!(table_exists(&conn, "release_marker").await);
    }

    #[tokio::test]
    async fn bridge_migration_rejects_non_terminal_old_major() {
        const RELEASED: &[ReleasedMigration] = &[];
        let bundle = MigrationBundle {
            current: SchemaVersion { major: 1, minor: 0 },
            snapshot_sql: "CREATE TABLE IF NOT EXISTS current_marker(id INTEGER);",
            released: RELEASED,
            dev: &[],
            bridge: Some(BridgeMigration {
                from: SchemaVersion { major: 0, minor: 1 },
                to: SchemaVersion { major: 1, minor: 0 },
                sql: "CREATE TABLE IF NOT EXISTS bridge_marker(id INTEGER);",
            }),
        };
        let conn = open_test_conn(test_db_path()).await;
        seed_schema_version(&conn, SchemaVersion { major: 0, minor: 0 }).await;

        let error = run_migration_bundle(&conn, bundle).await.unwrap_err();

        assert!(matches!(error, PortError::InvalidData(_)));
    }

    #[tokio::test]
    async fn released_migration_after_dev_tasks_clears_folded_task_tags() {
        const DEV_SQL: &str = "CREATE TABLE IF NOT EXISTS folded_dev(id INTEGER);";
        const RELEASE_SQL: &str = "CREATE TABLE IF NOT EXISTS folded_dev(id INTEGER);";
        const DEV_TASKS: &[DevMigration] = &[DevMigration {
            major: 0,
            minor: 2,
            task_tag: "folded-task",
            sql: DEV_SQL,
        }];
        const RELEASED: &[ReleasedMigration] = &[
            ReleasedMigration {
                from: SchemaVersion { major: 0, minor: 0 },
                to: SchemaVersion { major: 0, minor: 1 },
                sql: CURRENT_SCHEMA_SNAPSHOT_SQL,
            },
            ReleasedMigration {
                from: SchemaVersion { major: 0, minor: 1 },
                to: SchemaVersion { major: 0, minor: 2 },
                sql: RELEASE_SQL,
            },
        ];
        let dev_bundle = MigrationBundle {
            current: SchemaVersion { major: 0, minor: 1 },
            snapshot_sql: CURRENT_SCHEMA_SNAPSHOT_SQL,
            released: CURRENT_RELEASED_MIGRATIONS,
            dev: DEV_TASKS,
            bridge: None,
        };
        let release_bundle = MigrationBundle {
            current: SchemaVersion { major: 0, minor: 2 },
            snapshot_sql: CURRENT_SCHEMA_SNAPSHOT_SQL,
            released: RELEASED,
            dev: &[],
            bridge: None,
        };
        let conn = open_test_conn(test_db_path()).await;

        run_migration_bundle(&conn, dev_bundle).await.unwrap();
        assert_eq!(dev_task_count(&conn, 0, 2).await, 1);
        run_migration_bundle(&conn, release_bundle).await.unwrap();

        assert_eq!(
            schema_version_for_conn(&conn).await,
            SchemaVersion { major: 0, minor: 2 }
        );
        assert_eq!(dev_task_count(&conn, 0, 2).await, 0);
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
            .put(indexed_memory(
                MemoryIndex::new("identity-b"),
                "second",
                MemoryRank::Identity,
            ))
            .await
            .unwrap();
        store
            .put(indexed_memory(
                MemoryIndex::new("identity-a"),
                "first",
                MemoryRank::Identity,
            ))
            .await
            .unwrap();
        store
            .put(indexed_memory(
                MemoryIndex::new("ordinary"),
                "ordinary",
                MemoryRank::Permanent,
            ))
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
            .put(indexed_memory(id.clone(), "alpha", MemoryRank::ShortTerm))
            .await
            .unwrap();
        store.delete(&id).await.unwrap();
        assert!(store.get(&id).await.unwrap().is_none());

        store
            .put(indexed_memory(id.clone(), "beta", MemoryRank::Permanent))
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
            .insert(
                new_memory("alpha", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;
        store
            .insert(
                new_memory("beta", MemoryRank::LongTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap();
        store
            .insert(
                new_memory("gamma", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap();

        let hits = store.search(&memory_query("alpha", 1)).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, alpha);
    }

    #[tokio::test]
    async fn search_returns_empty_for_zero_limit() {
        let store = store().await;
        store
            .insert(
                new_memory("alpha", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap();

        let hits = store.search(&memory_query("alpha", 0)).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn delete_hides_rows_and_is_idempotent() {
        let store = store().await;
        let id = store
            .insert(
                new_memory("alpha", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;

        store.delete(&id).await.unwrap();
        store.delete(&id).await.unwrap();

        assert!(store.get(&id).await.unwrap().is_none());
        let hits = store.search(&memory_query("alpha", 10)).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn hard_delete_removes_sidecars_links_search_and_embeddings() {
        let store = store().await;
        let first = store
            .insert(
                NewMemory {
                    concepts: vec![MemoryConcept::new("alpha-project")],
                    tags: vec![MemoryTag::operational("follow-up")],
                    ..new_memory("alpha", MemoryRank::ShortTerm, None)
                },
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;
        let second = store
            .insert(
                new_memory("beta", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;
        store
            .upsert_link(
                NewMemoryLink {
                    from_memory: second.clone(),
                    to_memory: first.clone(),
                    relation: MemoryLinkRelation::Updates,
                    freeform_relation: None,
                    strength: 0.7,
                    confidence: 0.6,
                },
                chrono::Utc::now(),
            )
            .await
            .unwrap();
        let first_row_id = memory_row_id(&store, &first).await;

        store.delete(&first).await.unwrap();
        store.delete(&first).await.unwrap();

        assert!(store.get(&first).await.unwrap().is_none());
        assert_eq!(
            count_rows_where(
                &store,
                &store.memory_concepts_table_name(),
                "memory_id",
                first_row_id
            )
            .await,
            0
        );
        assert_eq!(
            count_rows_where(
                &store,
                &store.memory_tags_table_name(),
                "memory_id",
                first_row_id
            )
            .await,
            0
        );
        assert_eq!(
            count_rows_where(
                &store,
                &store.search_table_name(),
                "memory_id",
                first_row_id
            )
            .await,
            0
        );
        assert_eq!(
            count_rows_where(
                &store,
                store.embedding_table_name(),
                "memory_id",
                first_row_id
            )
            .await,
            0
        );
        assert!(
            store
                .linked(&LinkedMemoryQuery::around(vec![second], 8))
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .search(&MemoryQuery {
                    text: String::new(),
                    limit: 10,
                    kinds: Vec::new(),
                    concepts: vec!["alpha-project".to_string()],
                    tags: Vec::new(),
                })
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn linked_dedup_keeps_distinct_freeform_relations() {
        let store = store().await;
        let from = MemoryIndex::new("from");
        let to = MemoryIndex::new("to");
        let updated_at = chrono::Utc.with_ymd_and_hms(2025, 6, 1, 12, 0, 0).unwrap();
        store
            .put(indexed_memory(from.clone(), "alpha", MemoryRank::LongTerm))
            .await
            .unwrap();
        store
            .put(indexed_memory(to.clone(), "beta", MemoryRank::LongTerm))
            .await
            .unwrap();

        for freeform_relation in ["part-a", "part-b"] {
            let link = store
                .upsert_link(
                    NewMemoryLink {
                        from_memory: from.clone(),
                        to_memory: to.clone(),
                        relation: MemoryLinkRelation::Related,
                        freeform_relation: Some(freeform_relation.to_string()),
                        strength: 0.9,
                        confidence: 0.8,
                    },
                    updated_at,
                )
                .await
                .unwrap();
            assert_eq!(link.updated_at, updated_at);
        }

        let linked = store
            .linked(&LinkedMemoryQuery::around(vec![from], 8))
            .await
            .unwrap();
        let mut freeform_relations = linked
            .into_iter()
            .map(|item| item.link.freeform_relation.unwrap())
            .collect::<Vec<_>>();
        freeform_relations.sort();

        assert_eq!(freeform_relations, vec!["part-a", "part-b"]);
    }

    #[tokio::test]
    async fn compact_inserts_summary_and_hides_sources() {
        let store = store().await;
        let first = store
            .insert(
                new_memory("alpha", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;
        let second = store
            .insert(
                new_memory("beta", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;

        let merged = store
            .compact(
                NewMemory {
                    kind: MemoryKind::Reflection,
                    concepts: vec![MemoryConcept::new("alpha")],
                    tags: vec![MemoryTag::operational("compaction-summary")],
                    ..new_memory("alpha beta", MemoryRank::LongTerm, None)
                },
                &[first.clone(), second.clone()],
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;

        assert!(store.get(&first).await.unwrap().is_none());
        assert!(store.get(&second).await.unwrap().is_none());
        let got = store.get(&merged).await.unwrap().unwrap();
        assert_eq!(got.content.as_str(), "alpha beta");
        assert_eq!(got.rank, MemoryRank::LongTerm);
        assert_eq!(got.kind, MemoryKind::Reflection);

        let parsed = source_ids_for(&store, &merged).await;
        assert_eq!(parsed, vec![first.to_string(), second.to_string()]);
        let linked = store
            .linked(&LinkedMemoryQuery {
                memory_indexes: vec![merged.clone()],
                relation_filter: vec![MemoryLinkRelation::DerivedFrom],
                direction: MemoryLinkDirection::Outgoing,
                limit: 8,
            })
            .await
            .unwrap();
        assert!(linked.is_empty());
    }

    #[tokio::test]
    async fn put_compacted_mirrors_replica_flow() {
        let store = store().await;
        let first = MemoryIndex::new("first");
        let second = MemoryIndex::new("second");
        store
            .put(indexed_memory(
                first.clone(),
                "alpha",
                MemoryRank::ShortTerm,
            ))
            .await
            .unwrap();
        store
            .put(indexed_memory(
                second.clone(),
                "beta",
                MemoryRank::ShortTerm,
            ))
            .await
            .unwrap();

        let merged = MemoryIndex::new("merged");
        store
            .put_compacted(
                IndexedMemory {
                    kind: MemoryKind::Reflection,
                    tags: vec![MemoryTag::operational("compaction-summary")],
                    ..indexed_memory(merged.clone(), "alpha beta", MemoryRank::LongTerm)
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
        let store = LibsqlAgentStore::connect(
            LibsqlAgentStoreConfig::local(test_db_path(), 3, 3),
            Box::new(WrongDimEmbedder) as Box<dyn Embedder>,
            TestEmbedder::boxed(3),
        )
        .await
        .unwrap()
        .memory_store();

        let error = store
            .insert(
                new_memory("alpha", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
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
            .insert(
                new_memory("alpha", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;

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
            .insert(
                new_memory("alpha", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;

        let store_b = store_for_profile(path, profile("test", "b", 2)).await;
        store_b
            .put(indexed_memory(
                MemoryIndex::new("beta-only-b"),
                "beta",
                MemoryRank::ShortTerm,
            ))
            .await
            .unwrap();

        let hits_a = store_a.search(&memory_query("beta", 10)).await.unwrap();
        assert_eq!(hits_a.len(), 1);
        assert_eq!(hits_a[0].index, alpha);

        let hits_b = store_b.search(&memory_query("beta", 10)).await.unwrap();
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].index, MemoryIndex::new("beta-only-b"));
    }

    #[tokio::test]
    async fn profile_embeddings_become_stale_until_backfilled() {
        let path = test_db_path();
        let store_a = store_for_profile(path.clone(), profile("test", "a", 3)).await;
        let id = store_a
            .insert(
                new_memory("alpha", MemoryRank::ShortTerm, None),
                chrono::Utc::now(),
            )
            .await
            .unwrap()
            .index;

        let store_b = store_for_profile(path, profile("test", "b", 2)).await;
        assert_eq!(store_b.backfill_active_profile(10).await.unwrap(), 1);
        assert_eq!(
            store_b
                .search(&memory_query("alpha", 10))
                .await
                .unwrap()
                .len(),
            1
        );

        store_a
            .put(indexed_memory(id, "beta", MemoryRank::ShortTerm))
            .await
            .unwrap();

        assert!(
            store_b
                .search(&memory_query("beta", 10))
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(store_b.backfill_active_profile(10).await.unwrap(), 1);
        assert_eq!(
            store_b
                .search(&memory_query("beta", 10))
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

        let result = LibsqlAgentStore::connect(
            LibsqlAgentStoreConfig::local(path, profile.dimensions, profile.dimensions)
                .with_memory_active_profile(profile),
            TestEmbedder::boxed(3),
            TestEmbedder::boxed(3),
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
                      kind,
                      rank,
                      stored_at_ms,
                      created_at_ms,
                      updated_at_ms
                    )
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                    "#
                ),
                params![
                    id.as_str(),
                    "bad",
                    kind_to_i64(MemoryKind::Statement),
                    99_i64,
                    now_ms(),
                    now_ms(),
                    now_ms()
                ],
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
        LibsqlAgentStore::connect(
            LibsqlAgentStoreConfig::local(test_db_path(), 3, 3),
            TestEmbedder::boxed(3),
            TestEmbedder::boxed(3),
        )
        .await
        .unwrap()
        .memory_store()
    }

    fn new_memory(
        content: &str,
        rank: MemoryRank,
        occurred_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> NewMemory {
        NewMemory {
            content: MemoryContent::new(content),
            rank,
            occurred_at,
            kind: MemoryKind::Statement,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        }
    }

    fn indexed_memory(index: MemoryIndex, content: &str, rank: MemoryRank) -> IndexedMemory {
        IndexedMemory {
            index,
            content: MemoryContent::new(content),
            rank,
            occurred_at: None,
            stored_at: chrono::Utc::now(),
            kind: MemoryKind::Statement,
            concepts: Vec::new(),
            tags: Vec::new(),
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
        }
    }

    fn memory_query(text: &str, limit: usize) -> MemoryQuery {
        MemoryQuery::text(text, limit)
    }

    async fn policy_store() -> LibsqlPolicyStore {
        LibsqlAgentStore::connect(
            LibsqlAgentStoreConfig::local(test_db_path(), 3, 3),
            TestEmbedder::boxed(3),
            TestEmbedder::boxed(3),
        )
        .await
        .unwrap()
        .policy_store()
    }

    async fn store_for_profile(path: PathBuf, profile: EmbeddingProfile) -> LibsqlMemoryStore {
        let dims = profile.dimensions;
        LibsqlAgentStore::connect(
            LibsqlAgentStoreConfig::local(path, dims, dims).with_memory_active_profile(profile),
            TestEmbedder::boxed(dims),
            TestEmbedder::boxed(dims),
        )
        .await
        .unwrap()
        .memory_store()
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

    async fn count_rows_where(
        store: &LibsqlMemoryStore,
        table: &str,
        column: &str,
        value: i64,
    ) -> i64 {
        validate_identifier("test table", table).unwrap();
        validate_identifier("test column", column).unwrap();
        let mut rows = store
            .conn
            .query(
                &format!("SELECT COUNT(*) FROM {table} WHERE {column} = ?1"),
                [value],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        row.get(0).unwrap()
    }

    async fn memory_row_id(store: &LibsqlMemoryStore, index: &MemoryIndex) -> i64 {
        let mut rows = store
            .conn
            .query(
                &format!("SELECT id FROM {DEFAULT_MEMORY_TABLE} WHERE memory_index = ?1 LIMIT 1"),
                [index.as_str()],
            )
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

    async fn table_exists(conn: &Connection, table: &str) -> bool {
        let mut rows = conn
            .query(
                r#"
                SELECT 1
                FROM sqlite_schema
                WHERE type = 'table'
                  AND name = ?1
                LIMIT 1
                "#,
                [table],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().is_some()
    }

    async fn schema_version_for_conn(conn: &Connection) -> SchemaVersion {
        let mut rows = conn
            .query(
                &format!(
                    r#"
                    SELECT major, minor
                    FROM {SCHEMA_VERSION_TABLE}
                    WHERE schema_family = ?1
                    LIMIT 1
                    "#
                ),
                [SCHEMA_FAMILY],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        SchemaVersion {
            major: row.get(0).unwrap(),
            minor: row.get(1).unwrap(),
        }
    }

    async fn bridge_audit_for_conn(conn: &Connection) -> Option<SchemaVersion> {
        let mut rows = conn
            .query(
                &format!(
                    r#"
                    SELECT bridge_from_major, bridge_from_minor
                    FROM {SCHEMA_VERSION_TABLE}
                    WHERE schema_family = ?1
                    LIMIT 1
                    "#
                ),
                [SCHEMA_FAMILY],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let major: Option<i64> = row.get(0).unwrap();
        let minor: Option<i64> = row.get(1).unwrap();
        major
            .zip(minor)
            .map(|(major, minor)| SchemaVersion { major, minor })
    }

    async fn dev_task_count(conn: &Connection, major: i64, minor: i64) -> i64 {
        let mut rows = conn
            .query(
                &format!(
                    r#"
                    SELECT COUNT(*)
                    FROM {SCHEMA_DEV_TASK_TABLE}
                    WHERE schema_family = ?1
                      AND major = ?2
                      AND minor = ?3
                    "#
                ),
                params![SCHEMA_FAMILY, major, minor],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        row.get(0).unwrap()
    }

    async fn open_test_conn(path: PathBuf) -> Connection {
        let database = libsql::Builder::new_local(path).build().await.unwrap();
        database.connect().unwrap()
    }

    async fn seed_schema_version(conn: &Connection, version: SchemaVersion) {
        conn.execute_batch(MIGRATION_METADATA_DDL).await.unwrap();
        conn.execute(
            &format!(
                r#"
                INSERT INTO {SCHEMA_VERSION_TABLE} (
                  schema_family,
                  major,
                  minor,
                  updated_at_ms
                )
                VALUES (?1, ?2, ?3, ?4)
                "#
            ),
            params![SCHEMA_FAMILY, version.major, version.minor, now_ms()],
        )
        .await
        .unwrap();
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
        dir.join("agent.db")
    }
}
