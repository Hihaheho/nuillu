use std::sync::mpsc;
use std::thread;

use anyhow::Context as _;
use nuillu_visualizer_egui::{VisualizerApp, VisualizerChannels, VisualizerCommand, eframe};
use tokio::runtime::Builder;

use crate::{RunnerConfig, RunnerError, RunnerHooks, VisualizerHook, run_suite_with_hooks};

pub fn run_suite_with_visualizer(config: RunnerConfig) -> anyhow::Result<()> {
    let (event_tx, event_rx) = mpsc::channel();
    let (command_tx, command_rx) = mpsc::channel();
    let shutdown_tx = command_tx.clone();
    let eval_thread = thread::spawn(move || -> Result<(), RunnerError> {
        if !wait_for_start_command(&command_rx) {
            return Ok(());
        }
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|source| RunnerError::Driver {
                path: config.cases_root.clone(),
                message: source.to_string(),
            })?;
        let mut hooks = RunnerHooks::with_visualizer(VisualizerHook::new(event_tx, command_rx));
        runtime.block_on(run_suite_with_hooks(&config, &mut hooks))?;
        Ok(())
    });

    eframe::run_native(
        "Nuillu Visualizer",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            Ok(Box::new(VisualizerApp::new(
                cc,
                VisualizerChannels {
                    events: event_rx,
                    commands: command_tx,
                    start_suite_from_ui: true,
                },
            )))
        }),
    )
    .context("run egui visualizer")?;
    let _ = shutdown_tx.send(nuillu_visualizer_egui::VisualizerCommand::Shutdown);

    match eval_thread.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error.into()),
        Err(_) => anyhow::bail!("eval GUI thread panicked"),
    }
}

fn wait_for_start_command(commands: &mpsc::Receiver<VisualizerCommand>) -> bool {
    loop {
        match commands.recv() {
            Ok(VisualizerCommand::StartSuite) => return true,
            Ok(VisualizerCommand::Shutdown) | Err(_) => return false,
            Ok(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visualizer_wait_for_start_command_returns_true_for_start_suite() {
        let (tx, rx) = mpsc::channel();
        tx.send(VisualizerCommand::StartSuite).unwrap();

        assert!(wait_for_start_command(&rx));
    }

    #[test]
    fn visualizer_wait_for_start_command_returns_false_for_shutdown() {
        let (tx, rx) = mpsc::channel();
        tx.send(VisualizerCommand::Shutdown).unwrap();

        assert!(!wait_for_start_command(&rx));
    }
}
