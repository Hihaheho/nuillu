use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use eure::{FromEure, report::IntoErrorReports};

const SCHEMA_FAMILY: &str = "agent";

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let current_dir = manifest_dir.join("migrations/current");
    let bridge_dir = manifest_dir.join("migrations/bridge");
    rerun_if_changed(&current_dir);
    rerun_if_changed(&bridge_dir);

    let released_path = current_dir.join("released.eure");
    let dev_path = current_dir.join("dev.eure");
    let bridge_path = bridge_dir.join("bridge.eure");
    rerun_if_changed(&released_path);
    rerun_if_changed(&dev_path);
    rerun_if_changed(&bridge_path);

    let released: ReleasedManifest = read_eure(&released_path)?;
    let dev: DevManifest = read_eure(&dev_path)?;
    let bridge = read_bridge_manifest(&bridge_path)?;

    validate_schema_family(&released_path, Some(released.schema_family.as_str()))?;
    validate_schema_family(&dev_path, Some(dev.schema_family.as_str()))?;
    validate_schema_family(&bridge_path, bridge.schema_family.as_deref())?;

    let current_major = released.current_major;
    let current_minor = released.current_minor;
    let releases = release_entries(&released)?;
    let dev_tasks = dev_entries(&dev)?;
    let bridge_entry = bridge_entry(&bridge)?;

    for release in &releases {
        rerun_if_changed(&current_dir.join(&release.file));
    }
    for task in &dev_tasks {
        rerun_if_changed(&current_dir.join(&task.file));
    }
    if let Some(bridge) = &bridge_entry {
        rerun_if_changed(&bridge_dir.join(&bridge.file));
    }

    let mut out = String::new();
    out.push_str(&format!(
        "pub(crate) const CURRENT_SCHEMA_MAJOR: i64 = {current_major};\n"
    ));
    out.push_str(&format!(
        "pub(crate) const CURRENT_SCHEMA_MINOR: i64 = {current_minor};\n"
    ));
    out.push_str(
        "pub(crate) const CURRENT_SCHEMA_SNAPSHOT_SQL: &str = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/migrations/current/snapshot.sql\"));\n",
    );
    out.push_str("pub(crate) const CURRENT_RELEASED_MIGRATIONS: &[ReleasedMigration] = &[\n");
    for release in &releases {
        out.push_str(&format!(
            "    ReleasedMigration {{ from: SchemaVersion {{ major: {}, minor: {} }}, to: SchemaVersion {{ major: {}, minor: {} }}, sql: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/migrations/current/{}\")) }},\n",
            release.from_major,
            release.from_minor,
            release.to_major,
            release.to_minor,
            release.file
        ));
    }
    out.push_str("];\n");

    out.push_str("const CURRENT_DEV_MIGRATIONS: &[DevMigration] = &[\n");
    for task in &dev_tasks {
        out.push_str(&format!(
            "    DevMigration {{ major: {}, minor: {}, task_tag: {:?}, sql: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/migrations/current/{}\")) }},\n",
            task.major, task.minor, task.tag, task.file
        ));
    }
    out.push_str("];\n");

    if let Some(bridge) = bridge_entry {
        out.push_str(&format!(
            "const CURRENT_BRIDGE_MIGRATION: Option<BridgeMigration> = Some(BridgeMigration {{ from: SchemaVersion {{ major: {}, minor: {} }}, to: SchemaVersion {{ major: {}, minor: {} }}, sql: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/migrations/bridge/{}\")) }});\n",
            bridge.from_major,
            bridge.from_minor,
            bridge.to_major,
            bridge.to_minor,
            bridge.file
        ));
    } else {
        out.push_str("const CURRENT_BRIDGE_MIGRATION: Option<BridgeMigration> = None;\n");
    }

    fs::write(
        PathBuf::from(std::env::var("OUT_DIR")?).join("libsql_migrations.rs"),
        out,
    )?;
    Ok(())
}

fn rerun_if_changed(path: &Path) {
    println!("cargo::rerun-if-changed={}", path.display());
}

fn read_eure<T>(path: &Path) -> Result<T, Box<dyn Error>>
where
    T: for<'a> FromEure<'a> + Send + Sync + 'static,
    for<'a> <T as FromEure<'a>>::Error: IntoErrorReports,
{
    let text = fs::read_to_string(path)?;
    eure::parse_content(&text, path.to_path_buf())
        .map_err(|message| format!("failed to parse {}: {message}", path.display()).into())
}

fn read_bridge_manifest(path: &Path) -> Result<BridgeManifest, Box<dyn Error>> {
    if path.exists() {
        read_eure(path)
    } else {
        Ok(BridgeManifest::disabled())
    }
}

fn validate_schema_family(path: &Path, schema_family: Option<&str>) -> Result<(), Box<dyn Error>> {
    if let Some(schema_family) = schema_family
        && schema_family != SCHEMA_FAMILY
    {
        return Err(format!(
            "{} has schema-family={schema_family:?}; expected {SCHEMA_FAMILY:?}",
            path.display()
        )
        .into());
    }
    Ok(())
}

#[derive(Debug, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct ReleasedManifest {
    schema_family: String,
    current_major: i64,
    current_minor: i64,
    #[eure(default)]
    releases: Vec<ReleaseSpec>,
}

#[derive(Debug, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct ReleaseSpec {
    version: String,
    file: String,
}

#[derive(Debug)]
struct ReleaseEntry {
    from_major: i64,
    from_minor: i64,
    to_major: i64,
    to_minor: i64,
    file: String,
}

fn release_entries(manifest: &ReleasedManifest) -> Result<Vec<ReleaseEntry>, Box<dyn Error>> {
    if manifest.releases.is_empty() {
        if manifest.current_minor == 0 {
            return Ok(Vec::new());
        }
        return Err("released.eure must contain @ releases[] entries".into());
    }

    let mut seen_versions = BTreeSet::new();
    let mut entries = Vec::new();
    let mut from_major = manifest.current_major;
    let mut from_minor = 0;
    for release in &manifest.releases {
        if !seen_versions.insert(release.version.as_str()) {
            return Err(format!("duplicate released migration {}", release.version).into());
        }
        let (to_major, to_minor) = parse_version(&release.version)?;
        if to_major != manifest.current_major {
            return Err(format!(
                "release {} is not in current major {}",
                release.version, manifest.current_major
            )
            .into());
        }
        if to_minor == 0 {
            return Err(format!(
                "{} should be represented by the major bridge/snapshot, not @ releases[]",
                release.version
            )
            .into());
        }
        entries.push(ReleaseEntry {
            from_major,
            from_minor,
            to_major,
            to_minor,
            file: release.file.clone(),
        });
        from_major = to_major;
        from_minor = to_minor;
    }
    if from_minor != manifest.current_minor {
        return Err(format!(
            "released.eure current-minor={}, but last release is v{from_major}.{from_minor}",
            manifest.current_minor
        )
        .into());
    }
    Ok(entries)
}

#[derive(Debug, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct DevManifest {
    schema_family: String,
    next_major: i64,
    next_minor: i64,
    #[eure(default)]
    tasks: Vec<TaskSpec>,
}

#[derive(Debug, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct TaskSpec {
    tag: String,
    file: String,
}

#[derive(Debug)]
struct DevEntry {
    major: i64,
    minor: i64,
    tag: String,
    file: String,
}

fn dev_entries(manifest: &DevManifest) -> Result<Vec<DevEntry>, Box<dyn Error>> {
    let mut seen_tags = BTreeSet::new();
    let mut entries = Vec::new();
    for task in &manifest.tasks {
        if !seen_tags.insert(task.tag.as_str()) {
            return Err(format!("duplicate dev migration task {}", task.tag).into());
        }
        entries.push(DevEntry {
            major: manifest.next_major,
            minor: manifest.next_minor,
            tag: task.tag.clone(),
            file: task.file.clone(),
        });
    }
    Ok(entries)
}

#[derive(Debug, FromEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
struct BridgeManifest {
    #[eure(default)]
    schema_family: Option<String>,
    #[eure(default)]
    enabled: bool,
    #[eure(default)]
    from_major: Option<i64>,
    #[eure(default)]
    from_minor: Option<i64>,
    #[eure(default)]
    to_major: Option<i64>,
    #[eure(default)]
    to_minor: Option<i64>,
    #[eure(default)]
    file: Option<String>,
}

impl BridgeManifest {
    fn disabled() -> Self {
        Self {
            schema_family: None,
            enabled: false,
            from_major: None,
            from_minor: None,
            to_major: None,
            to_minor: None,
            file: None,
        }
    }
}

#[derive(Debug)]
struct BridgeEntry {
    from_major: i64,
    from_minor: i64,
    to_major: i64,
    to_minor: i64,
    file: String,
}

fn bridge_entry(manifest: &BridgeManifest) -> Result<Option<BridgeEntry>, Box<dyn Error>> {
    if !manifest.enabled {
        return Ok(None);
    }
    Ok(Some(BridgeEntry {
        from_major: required_bridge_field(manifest.from_major, "from-major")?,
        from_minor: required_bridge_field(manifest.from_minor, "from-minor")?,
        to_major: required_bridge_field(manifest.to_major, "to-major")?,
        to_minor: required_bridge_field(manifest.to_minor, "to-minor")?,
        file: manifest
            .file
            .clone()
            .ok_or("bridge.eure missing file when enabled = true")?,
    }))
}

fn required_bridge_field(value: Option<i64>, name: &str) -> Result<i64, Box<dyn Error>> {
    value.ok_or_else(|| format!("bridge.eure missing {name} when enabled = true").into())
}

fn parse_version(version: &str) -> Result<(i64, i64), Box<dyn Error>> {
    let Some(rest) = version.strip_prefix('v') else {
        return Err(format!("version must start with v: {version}").into());
    };
    let Some((major, minor)) = rest.split_once('.') else {
        return Err(format!("version must be v<major>.<minor>: {version}").into());
    };
    Ok((major.parse()?, minor.parse()?))
}
