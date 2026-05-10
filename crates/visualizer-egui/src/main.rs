use eframe::egui;
use egui::{Pos2, Vec2, Window};
use egui_hooks::UseHookExt;

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui App",
        native_options,
        Box::new(|cc| Ok(Box::new(MyEguiApp::new(cc)))),
    )
    .unwrap();
}

#[derive(Default)]
struct MyEguiApp {}

impl MyEguiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_global_style.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        Self::default()
    }
}

impl eframe::App for MyEguiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.push_id("chat", |ui| {
            ChatWindow {}.ui(ui);
        });
    }
}

pub struct ChatWindow {}

impl ChatWindow {
    fn ui(self, ui: &mut egui::Ui) {
        let pos = ui.use_persisted_state(|| Pos2::new(0.0, 0.0), ());
        if let Some(res) = Window::new("Chat")
            .current_pos(*pos)
            .movable(true)
            .show(ui.ctx(), |ui| {})
        {
            let delta = res.response.drag_delta();
            pos.set_next(Pos2::new(pos.x + delta.x, pos.y + delta.y));
        }
    }
}
