use egui::{Order, Pos2, Vec2, Window};
use egui_hooks::UseHookExt as _;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct WindowState {
    open: bool,
    pos: [f32; 2],
    size: [f32; 2],
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            open: true,
            pos: [24.0, 80.0],
            size: [520.0, 420.0],
        }
    }
}

pub struct PersistedWindow<'a> {
    id: &'a str,
    title: &'a str,
    default_pos: [f32; 2],
    default_size: [f32; 2],
    open_override: Option<bool>,
}

impl<'a> PersistedWindow<'a> {
    pub fn new(id: &'a str, title: &'a str) -> Self {
        Self {
            id,
            title,
            default_pos: [24.0, 80.0],
            default_size: [520.0, 420.0],
            open_override: None,
        }
    }

    pub fn default_pos(mut self, x: f32, y: f32) -> Self {
        self.default_pos = [x, y];
        self
    }

    pub fn default_size(mut self, width: f32, height: f32) -> Self {
        self.default_size = [width, height];
        self
    }

    pub fn open_requested(self, open_requested: bool) -> Self {
        if open_requested {
            self.open_override(Some(true))
        } else {
            self
        }
    }

    pub fn open_override(mut self, open_override: Option<bool>) -> Self {
        self.open_override = open_override;
        self
    }

    pub fn show(self, ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) -> bool {
        ui.push_id(self.id, |ui| {
            let state = ui.use_persisted_state(
                || WindowState {
                    pos: self.default_pos,
                    size: self.default_size,
                    ..WindowState::default()
                },
                (),
            );
            let effective_state = if let Some(open) = self.open_override {
                WindowState { open, ..*state }
            } else {
                *state
            };
            if !effective_state.open {
                if self.open_override.is_some() {
                    state.set_next(effective_state);
                }
                return false;
            }

            let mut open = effective_state.open;
            let mut window = Window::new(self.title)
                .id(egui::Id::new(self.id))
                .open(&mut open)
                .default_pos(Pos2::new(effective_state.pos[0], effective_state.pos[1]))
                .default_size(Vec2::new(effective_state.size[0], effective_state.size[1]));
            if self.open_override == Some(true) {
                window = window.order(Order::Foreground);
            }
            let response = window.show(ui.ctx(), |ui| {
                ui.push_id(self.id, add_contents);
            });

            if let Some(response) = response {
                let rect = response.response.rect;
                state.set_next(WindowState {
                    open,
                    pos: [rect.min.x, rect.min.y],
                    size: [rect.width(), rect.height()],
                });
            } else {
                state.set_next(WindowState {
                    open,
                    ..effective_state
                });
            }
            open
        })
        .inner
    }
}
