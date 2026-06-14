use std::collections::{BTreeMap, BTreeSet};

use crate::i18n::EguiI18nExt as _;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleFilterState {
    default_selected: bool,
    overrides: BTreeMap<String, bool>,
}

impl Default for ModuleFilterState {
    fn default() -> Self {
        Self {
            default_selected: true,
            overrides: BTreeMap::new(),
        }
    }
}

impl ModuleFilterState {
    pub fn is_selected(&self, module: &str) -> bool {
        self.overrides
            .get(module)
            .copied()
            .unwrap_or(self.default_selected)
    }

    pub fn set_selected(&mut self, module: String, selected: bool) {
        if selected == self.default_selected {
            self.overrides.remove(&module);
        } else {
            self.overrides.insert(module, selected);
        }
    }

    pub fn select_all(&mut self) {
        self.default_selected = true;
        self.overrides.clear();
    }

    pub fn deselect_all(&mut self) {
        self.default_selected = false;
        self.overrides.clear();
    }
}

pub fn render_module_filter(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    state: &mut ModuleFilterState,
    modules: impl IntoIterator<Item = impl AsRef<str>>,
) {
    let modules = normalized_modules(modules);
    let selected_count = modules
        .iter()
        .filter(|module| state.is_selected(module))
        .count();
    let title = ui.ctx().tr_args(
        "module-filter-title",
        &[
            ("selected", selected_count.into()),
            ("total", modules.len().into()),
        ],
    );
    egui::CollapsingHeader::new(title)
        .id_salt(id_salt)
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button(ui.ctx().tr("module-filter-select-all")).clicked() {
                    state.select_all();
                }
                if ui
                    .button(ui.ctx().tr("module-filter-deselect-all"))
                    .clicked()
                {
                    state.deselect_all();
                }
            });
            ui.add_space(4.0);
            if modules.is_empty() {
                ui.label(ui.ctx().tr("module-filter-empty"));
                return;
            }
            for module in modules {
                let mut selected = state.is_selected(&module);
                if ui.checkbox(&mut selected, &module).changed() {
                    state.set_selected(module, selected);
                }
            }
        });
}

pub fn normalized_modules(modules: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    modules
        .into_iter()
        .map(|module| module.as_ref().to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_filter_defaults_to_selected_and_bulk_actions_set_future_default() {
        let mut filter = ModuleFilterState::default();

        assert!(filter.is_selected("sensory"));
        filter.deselect_all();
        assert!(!filter.is_selected("sensory"));
        assert!(!filter.is_selected("future-module"));

        filter.set_selected("sensory".to_string(), true);
        assert!(filter.is_selected("sensory"));
        assert!(!filter.is_selected("future-module"));

        filter.select_all();
        assert!(filter.is_selected("sensory"));
        assert!(filter.is_selected("future-module"));
    }

    #[test]
    fn normalized_modules_sorts_and_deduplicates() {
        assert_eq!(
            normalized_modules(["memory", "sensory", "memory"]),
            vec!["memory".to_string(), "sensory".to_string()]
        );
    }
}
