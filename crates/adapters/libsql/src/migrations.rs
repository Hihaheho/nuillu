use std::collections::BTreeSet;

use libsql::{Connection, Transaction, TransactionBehavior, params};
use nuillu_module::ports::PortError;
use sha2::{Digest, Sha256};

pub(crate) const SCHEMA_FAMILY: &str = "agent";
pub(crate) const SCHEMA_VERSION_TABLE: &str = "nuillu_schema_version";
pub(crate) const SCHEMA_DEV_TASK_TABLE: &str = "nuillu_schema_dev_tasks";

pub(crate) const MIGRATION_METADATA_DDL: &str = include_str!("../migrations/current/metadata.sql");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SchemaVersion {
    pub(crate) major: i64,
    pub(crate) minor: i64,
}

impl SchemaVersion {
    fn label(self) -> String {
        format!("v{}.{}", self.major, self.minor)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ReleasedMigration {
    pub(crate) from: SchemaVersion,
    pub(crate) to: SchemaVersion,
    pub(crate) sql: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct DevMigration {
    pub(crate) major: i64,
    pub(crate) minor: i64,
    pub(crate) task_tag: &'static str,
    pub(crate) sql: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct BridgeMigration {
    pub(crate) from: SchemaVersion,
    pub(crate) to: SchemaVersion,
    pub(crate) sql: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct MigrationBundle {
    pub(crate) current: SchemaVersion,
    pub(crate) snapshot_sql: &'static str,
    pub(crate) released: &'static [ReleasedMigration],
    pub(crate) dev: &'static [DevMigration],
    pub(crate) bridge: Option<BridgeMigration>,
}

include!(concat!(env!("OUT_DIR"), "/libsql_migrations.rs"));

const CURRENT_MIGRATION_BUNDLE: MigrationBundle = MigrationBundle {
    current: SchemaVersion {
        major: CURRENT_SCHEMA_MAJOR,
        minor: CURRENT_SCHEMA_MINOR,
    },
    snapshot_sql: CURRENT_SCHEMA_SNAPSHOT_SQL,
    released: CURRENT_RELEASED_MIGRATIONS,
    dev: CURRENT_DEV_MIGRATIONS,
    bridge: CURRENT_BRIDGE_MIGRATION,
};

pub(crate) async fn run_agent_migrations(conn: &Connection) -> Result<(), PortError> {
    run_migration_bundle(conn, CURRENT_MIGRATION_BUNDLE).await
}

pub(crate) async fn run_migration_bundle(
    conn: &Connection,
    bundle: MigrationBundle,
) -> Result<(), PortError> {
    validate_migration_bundle(bundle)?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .await
        .map_err(map_libsql_error)?;
    tx.execute_batch(MIGRATION_METADATA_DDL)
        .await
        .map_err(map_libsql_error)?;

    let version = read_schema_version_tx(&tx).await?;
    let mut version = match version {
        Some(version) => version,
        None => {
            tx.execute_batch(bundle.snapshot_sql)
                .await
                .map_err(map_libsql_error)?;
            set_schema_version_tx(&tx, bundle.current, None).await?;
            clear_released_dev_tasks_tx(&tx, bundle.current).await?;
            apply_dev_migrations_tx(&tx, bundle, bundle.current).await?;
            tx.commit().await.map_err(map_libsql_error)?;
            return Ok(());
        }
    };

    if let Some(bridge) = bundle.bridge
        && version == bridge.from
    {
        tx.execute_batch(bridge.sql)
            .await
            .map_err(map_libsql_error)?;
        set_schema_version_tx(&tx, bridge.to, Some(bridge.from)).await?;
        version = bridge.to;
    }

    if version.major != bundle.current.major {
        let required = bundle
            .bridge
            .map(|bridge| bridge.from.label())
            .unwrap_or_else(|| bundle.current.label());
        return Err(PortError::InvalidData(format!(
            "libSQL schema {} cannot be opened by {}; migrate to {required} before running this binary",
            version.label(),
            bundle.current.label()
        )));
    }

    if version.minor > bundle.current.minor {
        return Err(PortError::InvalidData(format!(
            "libSQL schema {} is newer than this binary supports ({})",
            version.label(),
            bundle.current.label()
        )));
    }

    version = apply_released_migrations_tx(&tx, bundle, version).await?;

    clear_released_dev_tasks_tx(&tx, bundle.current).await?;
    apply_dev_migrations_tx(&tx, bundle, version).await?;
    tx.commit().await.map_err(map_libsql_error)
}

fn validate_migration_bundle(bundle: MigrationBundle) -> Result<(), PortError> {
    let mut seen_releases = BTreeSet::new();
    for migration in bundle.released {
        if migration.from == migration.to {
            return Err(PortError::InvalidData(format!(
                "released migration {} is self-referential",
                migration.to.label()
            )));
        }
        if !seen_releases.insert((migration.from.major, migration.from.minor)) {
            return Err(PortError::InvalidData(format!(
                "duplicate released migration source {}",
                migration.from.label()
            )));
        }
    }

    let mut seen_tasks = BTreeSet::new();
    for task in bundle.dev {
        validate_task_tag(task.task_tag)?;
        if !seen_tasks.insert((task.major, task.minor, task.task_tag)) {
            return Err(PortError::InvalidData(format!(
                "duplicate dev migration task v{}.{}.{}",
                task.major, task.minor, task.task_tag
            )));
        }
    }
    Ok(())
}

async fn read_schema_version_tx(tx: &Transaction) -> Result<Option<SchemaVersion>, PortError> {
    let mut rows = tx
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
        .map_err(map_libsql_error)?;
    Ok(match rows.next().await.map_err(map_libsql_error)? {
        Some(row) => Some(SchemaVersion {
            major: row.get(0).map_err(map_libsql_error)?,
            minor: row.get(1).map_err(map_libsql_error)?,
        }),
        None => None,
    })
}

async fn set_schema_version_tx(
    tx: &Transaction,
    version: SchemaVersion,
    bridge_from: Option<SchemaVersion>,
) -> Result<(), PortError> {
    tx.execute(
        &format!(
            r#"
            INSERT INTO {SCHEMA_VERSION_TABLE} (
              schema_family,
              major,
              minor,
              updated_at_ms,
              bridge_from_major,
              bridge_from_minor
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(schema_family) DO UPDATE SET
              major = excluded.major,
              minor = excluded.minor,
              updated_at_ms = excluded.updated_at_ms,
              bridge_from_major = COALESCE(
                excluded.bridge_from_major,
                {SCHEMA_VERSION_TABLE}.bridge_from_major
              ),
              bridge_from_minor = COALESCE(
                excluded.bridge_from_minor,
                {SCHEMA_VERSION_TABLE}.bridge_from_minor
              )
            "#
        ),
        params![
            SCHEMA_FAMILY,
            version.major,
            version.minor,
            now_ms(),
            bridge_from.map(|version| version.major),
            bridge_from.map(|version| version.minor),
        ],
    )
    .await
    .map_err(map_libsql_error)?;
    Ok(())
}

async fn apply_released_migrations_tx(
    tx: &Transaction,
    bundle: MigrationBundle,
    mut version: SchemaVersion,
) -> Result<SchemaVersion, PortError> {
    while version != bundle.current {
        let Some(migration) = bundle
            .released
            .iter()
            .find(|migration| migration.from == version)
        else {
            return Err(PortError::InvalidData(format!(
                "libSQL schema has no migration path from {} to {}",
                version.label(),
                bundle.current.label()
            )));
        };
        if migration.to.major != bundle.current.major || migration.to.minor > bundle.current.minor {
            return Err(PortError::InvalidData(format!(
                "released migration {} points outside supported schema {}",
                migration.to.label(),
                bundle.current.label()
            )));
        }
        tx.execute_batch(migration.sql)
            .await
            .map_err(map_libsql_error)?;
        set_schema_version_tx(tx, migration.to, None).await?;
        clear_released_dev_tasks_tx(tx, migration.to).await?;
        version = migration.to;
    }
    Ok(version)
}

async fn clear_released_dev_tasks_tx(
    tx: &Transaction,
    version: SchemaVersion,
) -> Result<(), PortError> {
    tx.execute(
        &format!(
            r#"
            DELETE FROM {SCHEMA_DEV_TASK_TABLE}
            WHERE schema_family = ?1
              AND major = ?2
              AND minor <= ?3
            "#
        ),
        params![SCHEMA_FAMILY, version.major, version.minor],
    )
    .await
    .map_err(map_libsql_error)?;
    Ok(())
}

async fn apply_dev_migrations_tx(
    tx: &Transaction,
    bundle: MigrationBundle,
    released_version: SchemaVersion,
) -> Result<(), PortError> {
    for task in bundle.dev {
        if task.major != released_version.major || task.minor <= released_version.minor {
            continue;
        }
        let checksum = sha256_hex(task.sql.as_bytes());
        let mut rows = tx
            .query(
                &format!(
                    r#"
                    SELECT checksum
                    FROM {SCHEMA_DEV_TASK_TABLE}
                    WHERE schema_family = ?1
                      AND major = ?2
                      AND minor = ?3
                      AND task_tag = ?4
                    LIMIT 1
                    "#
                ),
                params![SCHEMA_FAMILY, task.major, task.minor, task.task_tag],
            )
            .await
            .map_err(map_libsql_error)?;
        if let Some(row) = rows.next().await.map_err(map_libsql_error)? {
            let applied_checksum: String = row.get(0).map_err(map_libsql_error)?;
            if applied_checksum != checksum {
                return Err(PortError::InvalidData(format!(
                    "dev migration v{}.{}.{} checksum changed after it was applied",
                    task.major, task.minor, task.task_tag
                )));
            }
            continue;
        }
        drop(rows);

        tx.execute_batch(task.sql).await.map_err(map_libsql_error)?;
        tx.execute(
            &format!(
                r#"
                INSERT INTO {SCHEMA_DEV_TASK_TABLE} (
                  schema_family,
                  major,
                  minor,
                  task_tag,
                  checksum,
                  applied_at_ms
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                "#
            ),
            params![
                SCHEMA_FAMILY,
                task.major,
                task.minor,
                task.task_tag,
                checksum.as_str(),
                now_ms(),
            ],
        )
        .await
        .map_err(map_libsql_error)?;
    }
    Ok(())
}

fn validate_task_tag(task_tag: &str) -> Result<(), PortError> {
    let mut chars = task_tag.chars();
    let Some(first) = chars.next() else {
        return Err(PortError::InvalidInput(
            "migration task tag must not be empty".into(),
        ));
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return Err(PortError::InvalidInput(format!(
            "migration task tag must start with a lowercase ASCII letter or digit: {task_tag}"
        )));
    }
    if !chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-') {
        return Err(PortError::InvalidInput(format!(
            "migration task tag must use lowercase ASCII letters, digits, or hyphens: {task_tag}"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_string(&digest)
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
