use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use nuillu_module::{ActionAffordance, Participant};
use nuillu_visualizer_protocol::{
    AmbientSensoryRowView, DerivedAmbientSensoryRowView, EditableSceneStateView,
    ModuleSettingsView, SceneAtmosphereRowView, SceneObjectRowView, ScenePersonRowView,
    SceneRowKind, SceneRowView, SceneSoundRowView, SceneStateView, derive_scene_ambient,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModuleSettingsFile {
    modules: Vec<ModuleSettingsView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActionAffordancesFile {
    affordances: Vec<ActionAffordance>,
}

#[derive(Debug)]
pub(super) struct ActionAffordanceState {
    path: PathBuf,
    affordances: BTreeMap<String, ActionAffordance>,
}

impl ActionAffordanceState {
    pub(super) fn load(path: PathBuf) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::from_affordances(path, Vec::new()));
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read action affordances from {}", path.display()))?;
        let file: ActionAffordancesFile = serde_json::from_str(&text)
            .with_context(|| format!("parse action affordances from {}", path.display()))?;
        Ok(Self::from_affordances(path, file.affordances))
    }

    fn from_affordances(path: PathBuf, affordances: Vec<ActionAffordance>) -> Self {
        Self {
            path,
            affordances: affordances
                .into_iter()
                .map(|affordance| (affordance.id.clone(), affordance))
                .collect(),
        }
    }

    pub(super) fn save(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create action affordance dir {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(&ActionAffordancesFile {
            affordances: self.affordances(),
        })?;
        fs::write(&self.path, text)
            .with_context(|| format!("write action affordances to {}", self.path.display()))
    }

    pub(super) fn replace(&mut self, affordances: Vec<ActionAffordance>) {
        self.affordances = affordances
            .into_iter()
            .map(|affordance| (affordance.id.clone(), affordance))
            .collect();
    }

    pub(super) fn affordances(&self) -> Vec<ActionAffordance> {
        self.affordances.values().cloned().collect()
    }
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

fn load_legacy_ambient_rows(path: &Path) -> anyhow::Result<Vec<AmbientSensoryRowView>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("read ambient sensory rows from {}", path.display()))?;
    let file: AmbientRowsFile = serde_json::from_str(&text)
        .with_context(|| format!("parse ambient sensory rows from {}", path.display()))?;
    Ok(file.rows)
}

#[derive(Debug)]
pub(super) struct SceneState {
    path: PathBuf,
    people: Vec<ScenePersonRowView>,
    objects: Vec<SceneObjectRowView>,
    sounds: Vec<SceneSoundRowView>,
    atmosphere: Vec<SceneAtmosphereRowView>,
}

impl SceneState {
    pub(super) fn load(
        path: PathBuf,
        legacy_ambient_path: &Path,
        seed_participants: &[String],
    ) -> anyhow::Result<Self> {
        if path.exists() {
            let text = fs::read_to_string(&path)
                .with_context(|| format!("read scene state from {}", path.display()))?;
            let file: EditableSceneStateView = serde_json::from_str(&text)
                .with_context(|| format!("parse scene state from {}", path.display()))?;
            return Ok(Self::from_file(path, file));
        }

        let mut state = Self::from_file(path, EditableSceneStateView::default());
        for name in seed_participants {
            if !name.trim().is_empty() {
                let id = state.next_id(SceneRowKind::Person);
                state.people.push(ScenePersonRowView {
                    id,
                    name: name.trim().to_string(),
                    direction: String::new(),
                    distance: String::new(),
                    state: String::new(),
                });
            }
        }
        if legacy_ambient_path.exists() {
            for row in load_legacy_ambient_rows(legacy_ambient_path)?
                .into_iter()
                .filter(|row| !row.disabled && !row.content.trim().is_empty())
            {
                let id = state.next_id(SceneRowKind::Atmosphere);
                state.atmosphere.push(SceneAtmosphereRowView {
                    id,
                    aspect: "other".to_string(),
                    description: legacy_atmosphere_description(&row),
                });
            }
        }
        Ok(state)
    }

    fn from_file(path: PathBuf, file: EditableSceneStateView) -> Self {
        Self {
            path,
            people: file.people,
            objects: file.objects,
            sounds: file.sounds,
            atmosphere: file.atmosphere,
        }
    }

    pub(super) fn save(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create scene state dir {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(&self.editable_view())?;
        fs::write(&self.path, text)
            .with_context(|| format!("write scene state to {}", self.path.display()))
    }

    pub(super) fn view(&self) -> SceneStateView {
        self.editable_view().into_scene_state()
    }

    pub(super) fn replace(&mut self, state: EditableSceneStateView) {
        self.people = state.people;
        self.objects = state.objects;
        self.sounds = state.sounds;
        self.atmosphere = state.atmosphere;
    }

    pub(super) fn participants(&self) -> Vec<Participant> {
        self.people
            .iter()
            .filter_map(|row| {
                let name = row.name.trim();
                if name.is_empty() {
                    None
                } else {
                    Some(Participant::new(name))
                }
            })
            .collect()
    }

    pub(super) fn find_person(&self, row_id: &str) -> Option<&ScenePersonRowView> {
        self.people.iter().find(|row| row.id == row_id)
    }

    pub(super) fn create(&mut self, kind: SceneRowKind) {
        match kind {
            SceneRowKind::Person => {
                let id = self.next_id(kind);
                self.people.push(ScenePersonRowView {
                    id,
                    name: String::new(),
                    direction: String::new(),
                    distance: String::new(),
                    state: String::new(),
                });
            }
            SceneRowKind::Object => {
                let id = self.next_id(kind);
                self.objects.push(SceneObjectRowView {
                    id,
                    name: String::new(),
                    direction: String::new(),
                    distance: String::new(),
                    visual_description: String::new(),
                    sound_description: String::new(),
                });
            }
            SceneRowKind::Sound => {
                let id = self.next_id(kind);
                self.sounds.push(SceneSoundRowView {
                    id,
                    direction: String::new(),
                    distance: String::new(),
                    description: String::new(),
                });
            }
            SceneRowKind::Atmosphere => {
                let id = self.next_id(kind);
                self.atmosphere.push(SceneAtmosphereRowView {
                    id,
                    aspect: "light".to_string(),
                    description: String::new(),
                });
            }
        }
    }

    pub(super) fn update(&mut self, row: SceneRowView) {
        match row {
            SceneRowView::Person(row) => update_row(&mut self.people, row, |row| &row.id),
            SceneRowView::Object(row) => update_row(&mut self.objects, row, |row| &row.id),
            SceneRowView::Sound(row) => update_row(&mut self.sounds, row, |row| &row.id),
            SceneRowView::Atmosphere(row) => {
                update_row(&mut self.atmosphere, row, |row| &row.id);
            }
        }
    }

    pub(super) fn remove(&mut self, kind: SceneRowKind, row_id: &str) {
        match kind {
            SceneRowKind::Person => self.people.retain(|row| row.id != row_id),
            SceneRowKind::Object => self.objects.retain(|row| row.id != row_id),
            SceneRowKind::Sound => self.sounds.retain(|row| row.id != row_id),
            SceneRowKind::Atmosphere => self.atmosphere.retain(|row| row.id != row_id),
        }
    }

    pub(super) fn create_legacy_ambient(
        &mut self,
        modality: String,
        content: String,
        disabled: bool,
    ) {
        if disabled || content.trim().is_empty() {
            return;
        }
        let id = self.next_id(SceneRowKind::Atmosphere);
        self.atmosphere.push(SceneAtmosphereRowView {
            id,
            aspect: legacy_modality_aspect(&modality),
            description: content,
        });
    }

    pub(super) fn update_legacy_ambient(&mut self, row: AmbientSensoryRowView) {
        if row.disabled {
            self.remove(SceneRowKind::Atmosphere, &row.id);
            return;
        }
        let next = SceneAtmosphereRowView {
            id: row.id,
            aspect: legacy_modality_aspect(&row.modality),
            description: row.content,
        };
        if self
            .atmosphere
            .iter()
            .any(|existing| existing.id == next.id)
        {
            update_row(&mut self.atmosphere, next, |row| &row.id);
        } else {
            self.atmosphere.push(next);
        }
    }

    pub(super) fn remove_legacy_ambient(&mut self, row_id: &str) {
        self.remove(SceneRowKind::Atmosphere, row_id);
    }

    pub(super) fn derived_ambient(&self) -> Vec<DerivedAmbientSensoryRowView> {
        derive_scene_ambient(&self.editable_view())
    }

    fn editable_view(&self) -> EditableSceneStateView {
        EditableSceneStateView {
            people: self.people.clone(),
            objects: self.objects.clone(),
            sounds: self.sounds.clone(),
            atmosphere: self.atmosphere.clone(),
        }
    }

    fn next_id(&self, kind: SceneRowKind) -> String {
        let prefix = match kind {
            SceneRowKind::Person => "person",
            SceneRowKind::Object => "object",
            SceneRowKind::Sound => "sound",
            SceneRowKind::Atmosphere => "atmosphere",
        };
        let mut index = self.row_count(kind).saturating_add(1);
        loop {
            let id = format!("{prefix}-{index}");
            if !self.has_id(kind, &id) {
                return id;
            }
            index = index.saturating_add(1);
        }
    }

    fn row_count(&self, kind: SceneRowKind) -> usize {
        match kind {
            SceneRowKind::Person => self.people.len(),
            SceneRowKind::Object => self.objects.len(),
            SceneRowKind::Sound => self.sounds.len(),
            SceneRowKind::Atmosphere => self.atmosphere.len(),
        }
    }

    fn has_id(&self, kind: SceneRowKind, id: &str) -> bool {
        match kind {
            SceneRowKind::Person => self.people.iter().any(|row| row.id == id),
            SceneRowKind::Object => self.objects.iter().any(|row| row.id == id),
            SceneRowKind::Sound => self.sounds.iter().any(|row| row.id == id),
            SceneRowKind::Atmosphere => self.atmosphere.iter().any(|row| row.id == id),
        }
    }
}

fn update_row<T>(rows: &mut [T], row: T, id: impl Fn(&T) -> &str) {
    if let Some(existing) = rows.iter_mut().find(|existing| id(existing) == id(&row)) {
        *existing = row;
    }
}

fn legacy_atmosphere_description(row: &AmbientSensoryRowView) -> String {
    if row.modality.trim().is_empty() {
        row.content.clone()
    } else {
        format!("{}: {}", row.modality.trim(), row.content.trim())
    }
}

fn legacy_modality_aspect(modality: &str) -> String {
    match modality.trim().to_ascii_lowercase().as_str() {
        "vision" | "sight" | "visual" => "light".to_string(),
        "smell" | "olfaction" => "smell".to_string(),
        "touch" | "temperature" => "temperature".to_string(),
        _ => "other".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_affordance_state_loads_empty_and_round_trips() {
        let path = PathBuf::from(format!(
            ".tmp/action-affordances-{}.json",
            uuid::Uuid::now_v7()
        ));

        let mut state = ActionAffordanceState::load(path.clone()).unwrap();
        assert_eq!(
            state
                .affordances()
                .into_iter()
                .map(|affordance| affordance.id)
                .collect::<Vec<_>>(),
            Vec::<String>::new()
        );

        state.replace(vec![ActionAffordance {
            id: "clock".to_string(),
            label: "Clock".to_string(),
            description: "Check the current time.".to_string(),
            use_when: "when time matters".to_string(),
            effect: "The host reports the current time as sensory input.".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }]);
        state.save().unwrap();

        let reloaded = ActionAffordanceState::load(path).unwrap();
        assert_eq!(
            reloaded
                .affordances()
                .into_iter()
                .map(|affordance| affordance.id)
                .collect::<Vec<_>>(),
            vec!["clock".to_string()]
        );
    }

    #[test]
    fn scene_state_derives_people_objects_sounds_and_atmosphere() {
        let state = SceneState::from_file(
            PathBuf::from(".tmp/test-scene-state.json"),
            EditableSceneStateView {
                people: vec![ScenePersonRowView {
                    id: "person-1".to_string(),
                    name: "Pibi".to_string(),
                    direction: "front".to_string(),
                    distance: "2m".to_string(),
                    state: "watching Nui".to_string(),
                }],
                objects: vec![SceneObjectRowView {
                    id: "object-1".to_string(),
                    name: "bowl".to_string(),
                    direction: "left".to_string(),
                    distance: String::new(),
                    visual_description: "red food bowl".to_string(),
                    sound_description: "soft rattling".to_string(),
                }],
                sounds: vec![SceneSoundRowView {
                    id: "sound-1".to_string(),
                    direction: "behind".to_string(),
                    distance: "far".to_string(),
                    description: "rain tapping".to_string(),
                }],
                atmosphere: vec![SceneAtmosphereRowView {
                    id: "atmosphere-1".to_string(),
                    aspect: "smell".to_string(),
                    description: "wet stone smell".to_string(),
                }],
            },
        );

        assert_eq!(
            state.derived_ambient(),
            vec![
                DerivedAmbientSensoryRowView {
                    id: "scene:person:person-1".to_string(),
                    modality: "vision".to_string(),
                    content: "Pibi is present at front, 2m away; watching Nui.".to_string(),
                },
                DerivedAmbientSensoryRowView {
                    id: "scene:object:object-1:visual".to_string(),
                    modality: "vision".to_string(),
                    content: "bowl is visible at left; red food bowl.".to_string(),
                },
                DerivedAmbientSensoryRowView {
                    id: "scene:object:object-1:sound".to_string(),
                    modality: "audition".to_string(),
                    content: "bowl is making sound at left; soft rattling.".to_string(),
                },
                DerivedAmbientSensoryRowView {
                    id: "scene:sound:sound-1".to_string(),
                    modality: "audition".to_string(),
                    content: "A sound is present from behind, far away; rain tapping.".to_string(),
                },
                DerivedAmbientSensoryRowView {
                    id: "scene:atmosphere:atmosphere-1".to_string(),
                    modality: "smell".to_string(),
                    content: "smell: wet stone smell".to_string(),
                },
            ]
        );
    }

    #[test]
    fn scene_state_participants_skip_empty_names() {
        let state = SceneState::from_file(
            PathBuf::from(".tmp/test-scene-state.json"),
            EditableSceneStateView {
                people: vec![
                    ScenePersonRowView {
                        id: "person-1".to_string(),
                        name: "Pibi".to_string(),
                        direction: String::new(),
                        distance: String::new(),
                        state: String::new(),
                    },
                    ScenePersonRowView {
                        id: "person-2".to_string(),
                        name: " ".to_string(),
                        direction: String::new(),
                        distance: String::new(),
                        state: String::new(),
                    },
                ],
                ..EditableSceneStateView::default()
            },
        );

        assert_eq!(
            state
                .participants()
                .into_iter()
                .map(|participant| participant.name)
                .collect::<Vec<_>>(),
            vec!["Pibi".to_string()]
        );
    }

    #[test]
    fn scene_state_load_imports_legacy_ambient_once_and_persists() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmp")
            .join(format!("scene-state-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create test state dir");
        let legacy_path = root.join("ambient-sensory.json");
        let scene_path = root.join("scene-state.json");
        fs::write(
            &legacy_path,
            serde_json::to_string_pretty(&AmbientRowsFile {
                rows: vec![AmbientSensoryRowView {
                    id: "ambient-1".to_string(),
                    modality: "smell".to_string(),
                    content: "wet stone smell".to_string(),
                    disabled: false,
                }],
            })
            .expect("serialize legacy rows"),
        )
        .expect("write legacy rows");

        let state = SceneState::load(scene_path.clone(), &legacy_path, &["Pibi".to_string()])
            .expect("load imported scene state");

        assert_eq!(state.view().people[0].name, "Pibi");
        assert_eq!(
            state.view().atmosphere[0].description,
            "smell: wet stone smell"
        );

        state.save().expect("save imported scene state");
        let loaded = SceneState::load(scene_path, &legacy_path, &["Koro".to_string()])
            .expect("reload scene state");

        assert_eq!(
            loaded
                .view()
                .people
                .into_iter()
                .map(|row| row.name)
                .collect::<Vec<_>>(),
            vec!["Pibi".to_string()]
        );
        assert_eq!(loaded.view().atmosphere.len(), 1);
    }
}
