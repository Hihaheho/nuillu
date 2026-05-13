use std::{collections::BTreeMap, fs, path::PathBuf};

use anyhow::Context as _;
use nuillu_visualizer_protocol::{AmbientSensoryRowView, ModuleSettingsView};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModuleSettingsFile {
    modules: Vec<ModuleSettingsView>,
}

#[derive(Debug)]
pub(super) struct ModuleSettingsState {
    path: PathBuf,
    modules: BTreeMap<String, ModuleSettingsView>,
}

impl ModuleSettingsState {
    pub(super) fn load(path: PathBuf) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self {
                path,
                modules: BTreeMap::new(),
            });
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read module settings from {}", path.display()))?;
        let file: ModuleSettingsFile = serde_json::from_str(&text)
            .with_context(|| format!("parse module settings from {}", path.display()))?;
        Ok(Self {
            path,
            modules: file
                .modules
                .into_iter()
                .map(|settings| (settings.module.clone(), settings))
                .collect(),
        })
    }

    pub(super) fn save(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create module settings dir {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(&ModuleSettingsFile {
            modules: self.modules.values().cloned().collect(),
        })?;
        fs::write(&self.path, text)
            .with_context(|| format!("write module settings to {}", self.path.display()))
    }

    pub(super) fn upsert(&mut self, settings: ModuleSettingsView) {
        self.modules.insert(settings.module.clone(), settings);
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &ModuleSettingsView> {
        self.modules.values()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AmbientRowsFile {
    rows: Vec<AmbientSensoryRowView>,
}

#[derive(Debug)]
pub(super) struct AmbientRows {
    path: PathBuf,
    pub(super) rows: Vec<AmbientSensoryRowView>,
}

impl AmbientRows {
    pub(super) fn load(path: PathBuf) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self {
                path,
                rows: Vec::new(),
            });
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read ambient sensory rows from {}", path.display()))?;
        let file: AmbientRowsFile = serde_json::from_str(&text)
            .with_context(|| format!("parse ambient sensory rows from {}", path.display()))?;
        Ok(Self {
            path,
            rows: file.rows,
        })
    }

    pub(super) fn save(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create ambient state dir {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(&AmbientRowsFile {
            rows: self.rows.clone(),
        })?;
        fs::write(&self.path, text)
            .with_context(|| format!("write ambient sensory rows to {}", self.path.display()))
    }

    pub(super) fn create(&mut self, modality: String, content: String, disabled: bool) {
        let id = self.next_id();
        self.rows.push(AmbientSensoryRowView {
            id,
            modality,
            content,
            disabled,
        });
    }

    pub(super) fn update(&mut self, row: AmbientSensoryRowView) {
        if let Some(existing) = self.rows.iter_mut().find(|existing| existing.id == row.id) {
            *existing = row;
        }
    }

    pub(super) fn remove(&mut self, row_id: &str) {
        self.rows.retain(|row| row.id != row_id);
    }

    fn next_id(&self) -> String {
        let mut index = self.rows.len().saturating_add(1);
        loop {
            let id = format!("ambient-{index}");
            if self.rows.iter().all(|row| row.id != id) {
                return id;
            }
            index = index.saturating_add(1);
        }
    }
}
