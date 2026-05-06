use std::path::PathBuf;

use clap::Parser;
use nuillu_eval::{discover_case_files, normalize_text_block, parse_case_file};

#[derive(Debug, Parser)]
#[command(
    name = "nuillu-eval",
    about = "Validate and list data-driven nuillu eval cases"
)]
struct Args {
    /// Eure file or directory to load recursively.
    #[arg(long, default_value = "eval-cases")]
    cases: PathBuf,

    /// Print parsed case metadata as JSON.
    #[arg(long)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let case_paths = discover_case_files(&args.cases)?;

    if case_paths.is_empty() {
        anyhow::bail!("no .eure files found under {}", args.cases.display());
    }

    let mut entries = Vec::with_capacity(case_paths.len());
    for path in case_paths {
        let case = parse_case_file(&path)?;
        entries.push(serde_json::json!({
            "path": path.display().to_string(),
            "id": case.id.unwrap_or_else(|| {
                path.with_extension("")
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "case".to_string())
            }),
            "description": case.description.as_ref().map(|text| normalize_text_block(&text.content)),
            "checks": case.checks.len(),
        }));
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        for entry in &entries {
            println!(
                "{} checks={} {}",
                entry["id"].as_str().unwrap_or("case"),
                entry["checks"].as_u64().unwrap_or(0),
                entry["path"].as_str().unwrap_or("")
            );
        }
    }

    Ok(())
}
