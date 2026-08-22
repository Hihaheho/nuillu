use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use async_trait::async_trait;
use nuillu_module::{ActionAffordance, Participant};
use nuillu_visualizer_protocol::{
    AmbientSensoryRowView, DerivedAmbientSensoryRowView, EditableSceneStateView,
    ModuleSettingsView, SceneAtmosphereRowView, SceneObjectRowView, ScenePersonRowView,
    SceneRowKind, SceneRowView, SceneSoundRowView, SceneStateView, derive_scene_ambient,
};
use serde::{Deserialize, Serialize};

use crate::ports::ServerStatePort;

const SCENE_STATE_FILE: &str = "scene-state.json";
const LEGACY_AMBIENT_FILE: &str = "ambient-sensory.json";
const MODULE_SETTINGS_FILE: &str = "module-settings.json";
const ACTION_AFFORDANCES_FILE: &str = "action-affordances.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModuleSettingsFile {
    modules: Vec<ModuleSettingsView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActionAffordancesFile {
    affordances: Vec<ActionAffordance>,
}

#[derive(Debug, Clone)]
pub struct FileServerStatePort {
    state_dir: PathBuf,
}

impl FileServerStatePort {
    pub fn new(state_dir: impl Into<PathBuf>) -> Self {
        Self {
            state_dir: state_dir.into(),
        }
    }

    fn path(&self, file: &str) -> PathBuf {
        self.state_dir.join(file)
    }
}

#[async_trait(?Send)]
impl ServerStatePort for FileServerStatePort {
    async fn load_scene(
        &self,
        seed_participants: &[String],
    ) -> anyhow::Result<EditableSceneStateView> {
        let path = self.path(SCENE_STATE_FILE);
        if path.exists() {
            let text = fs::read_to_string(&path)
                .with_context(|| format!("read scene state from {}", path.display()))?;
            return serde_json::from_str(&text)
                .with_context(|| format!("parse scene state from {}", path.display()));
        }

        let mut state = initial_scene_state(seed_participants);
        let legacy_path = self.path(LEGACY_AMBIENT_FILE);
        for row in load_legacy_ambient_rows(&legacy_path)?
            .into_iter()
            .filter(|row| !row.disabled && !row.content.trim().is_empty())
        {
            let id = next_scene_id(&state, SceneRowKind::Atmosphere);
            state.atmosphere.push(SceneAtmosphereRowView {
                id,
                aspect: "other".to_string(),
                description: legacy_atmosphere_description(&row),
            });
        }
        Ok(state)
    }

    async fn save_scene(&self, state: &EditableSceneStateView) -> anyhow::Result<()> {
        write_json(self.path(SCENE_STATE_FILE), state, "scene state")
    }

    async fn load_module_settings(&self) -> anyhow::Result<Vec<ModuleSettingsView>> {
        let path = self.path(MODULE_SETTINGS_FILE);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read module settings from {}", path.display()))?;
        let file: ModuleSettingsFile = serde_json::from_str(&text)
            .with_context(|| format!("parse module settings from {}", path.display()))?;
        Ok(file.modules)
    }

    async fn save_module_settings(&self, settings: &[ModuleSettingsView]) -> anyhow::Result<()> {
        write_json(
            self.path(MODULE_SETTINGS_FILE),
            &ModuleSettingsFile {
                modules: settings.to_vec(),
            },
            "module settings",
        )
    }

    async fn load_action_affordances(&self) -> anyhow::Result<Vec<ActionAffordance>> {
        let path = self.path(ACTION_AFFORDANCES_FILE);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read action affordances from {}", path.display()))?;
        let file: ActionAffordancesFile = serde_json::from_str(&text)
            .with_context(|| format!("parse action affordances from {}", path.display()))?;
        Ok(file.affordances)
    }

    async fn save_action_affordances(
        &self,
        affordances: &[ActionAffordance],
    ) -> anyhow::Result<()> {
        write_json(
            self.path(ACTION_AFFORDANCES_FILE),
            &ActionAffordancesFile {
                affordances: affordances.to_vec(),
            },
            "action affordances",
        )
    }
}

#[derive(Debug, Default)]
pub struct InMemoryServerStatePort {
    scene: RefCell<Option<EditableSceneStateView>>,
    module_settings: RefCell<Vec<ModuleSettingsView>>,
    action_affordances: RefCell<Vec<ActionAffordance>>,
}

impl InMemoryServerStatePort {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait(?Send)]
impl ServerStatePort for InMemoryServerStatePort {
    async fn load_scene(
        &self,
        seed_participants: &[String],
    ) -> anyhow::Result<EditableSceneStateView> {
        Ok(self
            .scene
            .borrow()
            .clone()
            .unwrap_or_else(|| initial_scene_state(seed_participants)))
    }

    async fn save_scene(&self, state: &EditableSceneStateView) -> anyhow::Result<()> {
        self.scene.replace(Some(state.clone()));
        Ok(())
    }

    async fn load_module_settings(&self) -> anyhow::Result<Vec<ModuleSettingsView>> {
        Ok(self.module_settings.borrow().clone())
    }

    async fn save_module_settings(&self, settings: &[ModuleSettingsView]) -> anyhow::Result<()> {
        self.module_settings.replace(settings.to_vec());
        Ok(())
    }

    async fn load_action_affordances(&self) -> anyhow::Result<Vec<ActionAffordance>> {
        Ok(self.action_affordances.borrow().clone())
    }

    async fn save_action_affordances(
        &self,
        affordances: &[ActionAffordance],
    ) -> anyhow::Result<()> {
        self.action_affordances.replace(affordances.to_vec());
        Ok(())
    }
}

fn write_json(path: PathBuf, value: &impl Serialize, label: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create {label} dir {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(value)?;
    fs::write(&path, text).with_context(|| format!("write {label} to {}", path.display()))
}

fn initial_scene_state(seed_participants: &[String]) -> EditableSceneStateView {
    EditableSceneStateView {
        people: seed_participants
            .iter()
            .filter_map(|name| {
                let name = name.trim();
                (!name.is_empty()).then(|| name.to_owned())
            })
            .enumerate()
            .map(|(index, name)| ScenePersonRowView {
                id: format!("person-{}", index + 1),
                name,
                direction: String::new(),
                distance: String::new(),
                state: String::new(),
            })
            .collect(),
        ..EditableSceneStateView::default()
    }
}

fn next_scene_id(state: &EditableSceneStateView, kind: SceneRowKind) -> String {
    SceneState::from_file(state.clone()).next_id(kind)
}

#[derive(Debug)]
pub(super) struct ActionAffordanceState {
    affordances: BTreeMap<String, ActionAffordance>,
}

impl ActionAffordanceState {
    pub(super) async fn load(port: &dyn ServerStatePort) -> anyhow::Result<Self> {
        Ok(Self::from_affordances(
            port.load_action_affordances().await?,
        ))
    }

    fn from_affordances(affordances: Vec<ActionAffordance>) -> Self {
        Self {
            affordances: affordances
                .into_iter()
                .map(|affordance| (affordance.id.clone(), affordance))
                .collect(),
        }
    }

    pub(super) async fn save(&self, port: &dyn ServerStatePort) -> anyhow::Result<()> {
        port.save_action_affordances(&self.affordances()).await
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
    modules: BTreeMap<String, ModuleSettingsView>,
}

impl ModuleSettingsState {
    pub(super) async fn load(port: &dyn ServerStatePort) -> anyhow::Result<Self> {
        Ok(Self {
            modules: port
                .load_module_settings()
                .await?
                .into_iter()
                .map(|settings| (settings.module.clone(), settings))
                .collect(),
        })
    }

    pub(super) async fn save(&self, port: &dyn ServerStatePort) -> anyhow::Result<()> {
        port.save_module_settings(&self.modules.values().cloned().collect::<Vec<_>>())
            .await
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
    people: Vec<ScenePersonRowView>,
    objects: Vec<SceneObjectRowView>,
    sounds: Vec<SceneSoundRowView>,
    atmosphere: Vec<SceneAtmosphereRowView>,
}

impl SceneState {
    pub(super) async fn load(
        port: &dyn ServerStatePort,
        seed_participants: &[String],
    ) -> anyhow::Result<Self> {
        Ok(Self::from_file(port.load_scene(seed_participants).await?))
    }

    fn from_file(file: EditableSceneStateView) -> Self {
        Self {
            people: file.people,
            objects: file.objects,
            sounds: file.sounds,
            atmosphere: file.atmosphere,
        }
    }

    pub(super) async fn save(&self, port: &dyn ServerStatePort) -> anyhow::Result<()> {
        port.save_scene(&self.editable_view()).await
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

    #[tokio::test(flavor = "current_thread")]
    async fn in_memory_state_port_round_trips_typed_state() {
        let port = InMemoryServerStatePort::new();
        let scene = port.load_scene(&["Pibi".to_string()]).await.unwrap();
        let affordance = ActionAffordance {
            id: "clock".to_string(),
            label: "Clock".to_string(),
            description: "Check the current time.".to_string(),
            use_when: "when time matters".to_string(),
            effect: "The host reports the current time.".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };

        port.save_scene(&scene).await.unwrap();
        port.save_action_affordances(std::slice::from_ref(&affordance))
            .await
            .unwrap();

        assert_eq!(port.load_scene(&["Koro".to_string()]).await.unwrap(), scene);
        assert_eq!(
            port.load_action_affordances().await.unwrap(),
            vec![affordance]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn action_affordance_state_loads_empty_and_round_trips() {
        let root = PathBuf::from(format!(".tmp/action-affordances-{}", uuid::Uuid::now_v7()));
        let port = FileServerStatePort::new(root);

        let mut state = ActionAffordanceState::load(&port).await.unwrap();
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
        state.save(&port).await.unwrap();

        let reloaded = ActionAffordanceState::load(&port).await.unwrap();
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
        let state = SceneState::from_file(EditableSceneStateView {
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
        });

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
        let state = SceneState::from_file(EditableSceneStateView {
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
        });

        assert_eq!(
            state
                .participants()
                .into_iter()
                .map(|participant| participant.name)
                .collect::<Vec<_>>(),
            vec!["Pibi".to_string()]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scene_state_load_imports_legacy_ambient_once_and_persists() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmp")
            .join(format!("scene-state-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create test state dir");
        let legacy_path = root.join("ambient-sensory.json");
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

        let port = FileServerStatePort::new(root);
        let state = SceneState::load(&port, &["Pibi".to_string()])
            .await
            .expect("load imported scene state");

        assert_eq!(state.view().people[0].name, "Pibi");
        assert_eq!(
            state.view().atmosphere[0].description,
            "smell: wet stone smell"
        );

        state.save(&port).await.expect("save imported scene state");
        let loaded = SceneState::load(&port, &["Koro".to_string()])
            .await
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
