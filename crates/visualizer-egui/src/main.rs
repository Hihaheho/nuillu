use std::sync::mpsc;

use nuillu_visualizer_egui::{VisualizerApp, VisualizerChannels};
use nuillu_visualizer_protocol::{VisualizerClientMessage, VisualizerClientPort};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(std::env::args().skip(1))?;
    let remote = args.host.is_some();
    let (server_messages, client_messages) = if let Some(host) = args.host {
        let port = VisualizerClientPort::connect(host.as_str())?;
        port.send(VisualizerClientMessage::hello())?;
        port.into_channels()
    } else {
        let (_server_tx, server_rx) = mpsc::channel();
        let (client_tx, _client_rx) = mpsc::channel();
        (server_rx, client_tx)
    };
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_active(true)
            .with_visible(true),
        ..eframe::NativeOptions::default()
    };
    eframe::run_native(
        "Nuillu Visualizer",
        native_options,
        Box::new(|cc| {
            Ok(Box::new(VisualizerApp::new(
                cc,
                VisualizerChannels {
                    server_messages,
                    client_messages,
                    remote,
                },
            )))
        }),
    )?;
    Ok(())
}

#[derive(Debug, Default)]
struct Args {
    host: Option<String>,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, std::io::Error> {
        let mut parsed = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--host" => {
                    let Some(host) = args.next() else {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "--host requires an address",
                        ));
                    };
                    parsed.host = Some(host);
                }
                "-h" | "--help" => {
                    println!("Usage: nuillu-visualizer-gui [--host HOST:PORT]");
                    std::process::exit(0);
                }
                other => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("unknown argument: {other}"),
                    ));
                }
            }
        }
        Ok(parsed)
    }
}
