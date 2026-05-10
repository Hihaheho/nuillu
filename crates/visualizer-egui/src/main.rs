use std::sync::mpsc;

use nuillu_visualizer_egui::{VisualizerApp, VisualizerChannels};

fn main() -> eframe::Result<()> {
    let (_event_tx, event_rx) = mpsc::channel();
    let (command_tx, _command_rx) = mpsc::channel();
    eframe::run_native(
        "Nuillu Visualizer",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            Ok(Box::new(VisualizerApp::new(
                cc,
                VisualizerChannels {
                    events: event_rx,
                    commands: command_tx,
                },
            )))
        }),
    )
}
