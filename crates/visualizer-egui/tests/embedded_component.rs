use nuillu_visualizer_egui::Locale;
use nuillu_visualizer_egui::{
    AgentActionInvocationRequest, BlackboardSnapshot, Visualizer, VisualizerClientMessage,
    VisualizerCommand, VisualizerConfig, VisualizerEvent, VisualizerServerMessage, VisualizerTabId,
    VisualizerUiResources, blackboard, egui,
};

fn embedded_config() -> VisualizerConfig {
    VisualizerConfig::default()
}

fn open_tab(visualizer: &mut Visualizer, id: &str) {
    visualizer.apply_server_message(VisualizerServerMessage::Event {
        event: VisualizerEvent::OpenTab {
            tab_id: VisualizerTabId::new(id),
            title: "Embedded".to_string(),
        },
    });
    visualizer.apply_server_message(VisualizerServerMessage::Event {
        event: VisualizerEvent::AgentActionInvocationRequested {
            tab_id: VisualizerTabId::new(id),
            request: AgentActionInvocationRequest {
                invocation_id: format!("{id}-invocation"),
                action_id: "embedded-test-action".to_string(),
                arguments: serde_json::Value::Null,
            },
        },
    });
}

#[test]
fn root_component_renders_two_namespaced_instances_without_a_transport() {
    let ctx = egui::Context::default();
    let mut first = Visualizer::with_config(egui::Id::new("first"), embedded_config());
    let mut second = Visualizer::with_config(egui::Id::new("second"), embedded_config());
    open_tab(&mut first, "shared-tab-id");
    open_tab(&mut second, "shared-tab-id");

    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1600.0, 1200.0),
        )),
        time: Some(0.0),
        ..egui::RawInput::default()
    };
    let mut first_messages = Vec::new();
    let mut second_messages = Vec::new();
    let _ = ctx.run_ui(input, |ui| {
        first_messages.extend(first.show(ui).into_messages());
        second_messages.extend(second.show(ui).into_messages());
    });

    for messages in [first_messages, second_messages] {
        assert!(
            messages.iter().any(|message| matches!(
                message,
                VisualizerClientMessage::Command {
                    command: VisualizerCommand::CompleteAgentActionInvocation { tab_id, .. },
                } if tab_id.as_str() == "shared-tab-id"
            )),
            "unexpected component messages: {:?}",
            messages
        );
    }
}

#[test]
fn leaf_component_has_a_safe_default_i18n_context() {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 600.0),
        )),
        ..egui::RawInput::default()
    };

    let _ = ctx.run_ui(input, |ui| {
        blackboard::ui(ui, &BlackboardSnapshot::default());
    });
}

#[test]
fn host_ftl_combines_with_embedded_translations() {
    let resources = VisualizerUiResources::builder()
        .add_ftl(
            Locale::JaJp,
            "host-title = { menu-theme-light } + host\nmenu-zoom = Host override",
        )
        .build()
        .expect("host translations load");

    assert_eq!(
        resources.translate(Locale::JaJp, "host-title"),
        "ライト + host"
    );
    assert_eq!(
        resources.translate(Locale::JaJp, "menu-zoom"),
        "Host override"
    );
    assert_eq!(
        resources.translate(Locale::JaJp, "i18n-fallback-probe"),
        "English fallback"
    );
}
