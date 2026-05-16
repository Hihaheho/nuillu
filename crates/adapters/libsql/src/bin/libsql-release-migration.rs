use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use eure::FromEure;
use libsql::Connection;

const SCHEMA_FAMILY: &str = "agent";
const METADATA_SQL: &str = include_str!("../../migrations/current/metadata.sql");
const SNAPSHOT_SQL: &str = include_str!("../../migrations/current/snapshot.sql");

#[derive(Debug)]
struct Args {
    current_dir: PathBuf,
    major: Option<i64>,
    minor: Option<i64>,
    bridge_from_major: Option<i64>,
    bridge_from_minor: Option<i64>,
    dry_run: bool,
}

#[derive(Debug, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct DevManifest {
    schema_family: String,
    next_major: i64,
    next_minor: i64,
    #[eure(default)]
    tasks: Vec<DevTask>,
}

#[derive(Debug, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct DevTask {
    tag: String,
    file: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let manifest = read_dev_manifest(&args.current_dir.join("dev.eure"))?;
    let major = args.major.unwrap_or(manifest.next_major);
    let minor = args.minor.unwrap_or(manifest.next_minor);
    if major != manifest.next_major || minor != manifest.next_minor {
        return Err(format!(
            "requested v{major}.{minor}, but dev.eure points at v{}.{}",
            manifest.next_major, manifest.next_minor
        )
        .into());
    }

    validate_manifest(&manifest, major, minor)?;
    let folded_sql = fold_task_sql(&args.current_dir, major, minor, &manifest)?;
    verify_idempotent(&args.current_dir, &manifest).await?;

    if args.dry_run {
        println!(
            "validated {} task migration(s) for v{major}.{minor}",
            manifest.tasks.len()
        );
        return Ok(());
    }

    let released_file = args
        .current_dir
        .join("released")
        .join(format!("v{major}.{minor}.sql"));
    fs::create_dir_all(released_file.parent().expect("released file has parent"))?;
    fs::write(&released_file, folded_sql.as_bytes())?;
    let released_manifest = args.current_dir.join("released.eure");
    if minor == 0 {
        write_major_zero_release_manifest(&released_manifest, major)?;
    } else {
        append_release_manifest(
            &released_manifest,
            &format!("v{major}.{minor}"),
            &format!("released/v{major}.{minor}.sql"),
        )?;
        update_release_manifest_current(&released_manifest, major, minor)?;
    }
    regenerate_snapshot(&args.current_dir, &manifest).await?;
    write_empty_next_dev_manifest(&args.current_dir.join("dev.eure"), major, minor + 1)?;

    if minor == 0 {
        match (args.bridge_from_major, args.bridge_from_minor) {
            (Some(from_major), Some(from_minor)) => {
                let bridge_dir = bridge_dir_for_current(&args.current_dir)?;
                fs::create_dir_all(&bridge_dir)?;
                let bridge_path = bridge_dir.join("previous-to-current.sql");
                let bridge_sql = format!(
                    "-- Bridge migration from v{from_major}.{from_minor} to v{major}.0.\n{}",
                    fs::read_to_string(&released_file)?
                );
                fs::write(bridge_path, bridge_sql)?;
                fs::write(
                    bridge_dir.join("bridge.eure"),
                    format!(
                        "schema-family = \"{SCHEMA_FAMILY}\"\nenabled = true\nfrom-major = {from_major}\nfrom-minor = {from_minor}\nto-major = {major}\nto-minor = 0\nfile = \"previous-to-current.sql\"\n"
                    ),
                )?;
            }
            _ => {
                return Err(
                    "major release v*.0 requires --bridge-from-major and --bridge-from-minor"
                        .into(),
                );
            }
        }
    }

    println!(
        "released v{major}.{minor} from {} task migration(s)",
        manifest.tasks.len()
    );
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut current_dir = default_current_dir();
    let mut major = None;
    let mut minor = None;
    let mut bridge_from_major = None;
    let mut bridge_from_minor = None;
    let mut dry_run = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--current-dir" => current_dir = PathBuf::from(required_value(&mut args, &arg)?),
            "--major" => major = Some(required_value(&mut args, &arg)?.parse()?),
            "--minor" => minor = Some(required_value(&mut args, &arg)?.parse()?),
            "--bridge-from-major" => {
                bridge_from_major = Some(required_value(&mut args, &arg)?.parse()?);
            }
            "--bridge-from-minor" => {
                bridge_from_minor = Some(required_value(&mut args, &arg)?.parse()?);
            }
            "--dry-run" => dry_run = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    Ok(Args {
        current_dir,
        major,
        minor,
        bridge_from_major,
        bridge_from_minor,
        dry_run,
    })
}

fn required_value(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value").into())
}

fn default_current_dir() -> PathBuf {
    let repo_relative = PathBuf::from("crates/adapters/libsql/migrations/current");
    if repo_relative.exists() {
        return repo_relative;
    }
    PathBuf::from("migrations/current")
}

fn bridge_dir_for_current(current_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    Ok(current_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("current migration dir has no bridge sibling")?
        .join("bridge"))
}

fn print_help() {
    println!(
        "Usage: libsql-release-migration [--current-dir DIR] [--major N] [--minor N] [--dry-run]"
    );
}

fn read_dev_manifest(path: &Path) -> Result<DevManifest, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let manifest: DevManifest = eure::parse_content(&text, path.to_path_buf())
        .map_err(|message| format!("failed to parse {}: {message}", path.display()))?;
    if manifest.schema_family != SCHEMA_FAMILY {
        return Err(format!(
            "{} has schema-family={:?}; expected {SCHEMA_FAMILY:?}",
            path.display(),
            manifest.schema_family
        )
        .into());
    }
    Ok(manifest)
}

fn validate_manifest(manifest: &DevManifest, major: i64, minor: i64) -> Result<(), Box<dyn Error>> {
    let mut seen = BTreeSet::new();
    for task in &manifest.tasks {
        validate_task_tag(&task.tag)?;
        if !seen.insert(task.tag.as_str()) {
            return Err(format!("duplicate task tag: {}", task.tag).into());
        }
        let expected = format!("v{major}.{minor}.{}.sql", task.tag);
        let file_name = Path::new(&task.file)
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("invalid task file path: {}", task.file))?;
        if file_name != expected {
            return Err(format!("task {} must use file name {expected}", task.tag).into());
        }
    }
    Ok(())
}

fn validate_task_tag(task_tag: &str) -> Result<(), Box<dyn Error>> {
    let mut chars = task_tag.chars();
    let Some(first) = chars.next() else {
        return Err("task tag must not be empty".into());
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return Err(format!("invalid task tag: {task_tag}").into());
    }
    if !chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-') {
        return Err(format!("invalid task tag: {task_tag}").into());
    }
    Ok(())
}

fn fold_task_sql(
    current_dir: &Path,
    major: i64,
    minor: i64,
    manifest: &DevManifest,
) -> Result<String, Box<dyn Error>> {
    let mut out = format!("-- Released migration v{major}.{minor}.\n");
    for task in &manifest.tasks {
        let path = current_dir.join(&task.file);
        let sql = fs::read_to_string(&path)?;
        out.push_str("\n-- ");
        out.push_str(&task.tag);
        out.push('\n');
        out.push_str(sql.trim());
        out.push('\n');
    }
    Ok(out)
}

async fn verify_idempotent(
    current_dir: &Path,
    manifest: &DevManifest,
) -> Result<(), Box<dyn Error>> {
    let conn = scratch_connection().await?;
    conn.execute_batch(METADATA_SQL).await?;
    conn.execute_batch(SNAPSHOT_SQL).await?;
    conn.execute_batch(METADATA_SQL).await?;
    conn.execute_batch(SNAPSHOT_SQL).await?;
    for task in &manifest.tasks {
        let sql = fs::read_to_string(current_dir.join(&task.file))?;
        conn.execute_batch(&sql).await?;
        conn.execute_batch(&sql).await?;
    }
    Ok(())
}

async fn regenerate_snapshot(
    current_dir: &Path,
    manifest: &DevManifest,
) -> Result<(), Box<dyn Error>> {
    let conn = scratch_connection().await?;
    conn.execute_batch(METADATA_SQL).await?;
    conn.execute_batch(SNAPSHOT_SQL).await?;
    for task in &manifest.tasks {
        conn.execute_batch(&fs::read_to_string(current_dir.join(&task.file))?)
            .await?;
    }

    let mut rows = conn
        .query(
            r#"
            SELECT sql
            FROM sqlite_schema
            WHERE sql IS NOT NULL
              AND name NOT LIKE 'sqlite_%'
              AND name NOT LIKE 'nuillu_schema_%'
            ORDER BY
              CASE type
                WHEN 'table' THEN 0
                WHEN 'index' THEN 1
                WHEN 'view' THEN 2
                WHEN 'trigger' THEN 3
                ELSE 4
              END,
              name
            "#,
            (),
        )
        .await?;
    let mut snapshot = String::new();
    while let Some(row) = rows.next().await? {
        let sql: String = row.get(0)?;
        snapshot.push_str(sql.trim_end_matches(';'));
        snapshot.push_str(";\n\n");
    }
    fs::write(current_dir.join("snapshot.sql"), snapshot)?;
    Ok(())
}

async fn scratch_connection() -> Result<Connection, Box<dyn Error>> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let dir = std::env::current_dir()?.join(".tmp").join(format!(
        "libsql-release-migration-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&dir)?;
    let path = dir.join("scratch.db");
    let database = libsql::Builder::new_local(path).build().await?;
    Ok(database.connect()?)
}

fn append_release_manifest(path: &Path, version: &str, file: &str) -> Result<(), Box<dyn Error>> {
    let mut text = fs::read_to_string(path)?;
    if text.contains(&format!("version = \"{version}\"")) {
        return Err(format!("{version} already exists in {}", path.display()).into());
    }
    text.push_str(&format!(
        "\n@ releases[] {{\n  version = \"{version}\"\n  file = \"{file}\"\n}}\n"
    ));
    fs::write(path, text)?;
    Ok(())
}

fn write_major_zero_release_manifest(path: &Path, major: i64) -> Result<(), Box<dyn Error>> {
    fs::write(
        path,
        format!(
            "schema-family = \"{SCHEMA_FAMILY}\"\ncurrent-major = {major}\ncurrent-minor = 0\n"
        ),
    )?;
    Ok(())
}

fn update_release_manifest_current(
    path: &Path,
    major: i64,
    minor: i64,
) -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut saw_major = false;
    let mut saw_minor = false;
    let mut out = String::new();
    for line in text.lines() {
        if line.starts_with("current-major") {
            out.push_str(&format!("current-major = {major}\n"));
            saw_major = true;
        } else if line.starts_with("current-minor") {
            out.push_str(&format!("current-minor = {minor}\n"));
            saw_minor = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !saw_major || !saw_minor {
        return Err("released.eure must contain current-major and current-minor".into());
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_empty_next_dev_manifest(
    path: &Path,
    major: i64,
    next_minor: i64,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        path,
        format!(
            "schema-family = \"{SCHEMA_FAMILY}\"\nnext-major = {major}\nnext-minor = {next_minor}\n"
        ),
    )?;
    Ok(())
}
