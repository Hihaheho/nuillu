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
    open_requested: bool,
}

impl<'a> PersistedWindow<'a> {
    pub fn new(id: &'a str, title: &'a str) -> Self {
        Self {
            id,
            title,
            default_pos: [24.0, 80.0],
            default_size: [520.0, 420.0],
            open_requested: false,
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

    pub fn open_requested(mut self, open_requested: bool) -> Self {
        self.open_requested = open_requested;
        self
    }

    pub fn show(self, ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
        ui.push_id(self.id, |ui| {
            let state = ui.use_persisted_state(
                || WindowState {
                    pos: self.default_pos,
                    size: self.default_size,
                    ..WindowState::default()
                },
                (),
            );
            let effective_state = if self.open_requested {
                WindowState {
                    open: true,
                    ..*state
                }
            } else {
                *state
            };
            if !effective_state.open {
                return;
            }

            let mut open = effective_state.open;
            let mut window = Window::new(self.title)
                .id(egui::Id::new(self.id))
                .open(&mut open)
                .default_pos(Pos2::new(effective_state.pos[0], effective_state.pos[1]))
                .default_size(Vec2::new(effective_state.size[0], effective_state.size[1]));
            if self.open_requested {
                window = window.order(Order::Foreground);
            }
            let response = window.show(ui.ctx(), add_contents);

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
        });
    }
}
