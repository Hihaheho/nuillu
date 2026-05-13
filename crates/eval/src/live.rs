use std::{fs, net::TcpListener, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context as _;
use nuillu_agent::{AgentEventLoopConfig, run as run_agent};
use nuillu_blackboard::BlackboardCommand;
use nuillu_memory::{MemoryQuery, MemoryRecord, MemoryStore};
use nuillu_module::{
    AmbientSensoryEntry, Participant, SensoryInput, SensoryInputMailbox, SensoryModality,
};
use nuillu_query_agentic::NoopFileSearchProvider;
use nuillu_types::{MemoryRank, ModuleId};
use nuillu_visualizer_protocol::{
    AmbientSensoryRowView, MemoryRecordView, ModuleSettingsView, TabStatus,
    VisualizerClientMessage, VisualizerCommand, VisualizerEvent, VisualizerServerMessage,
    VisualizerServerPort, VisualizerTabId, memory_page_from_records,
};
use serde::{Deserialize, Serialize};
use tokio::{runtime::Builder, task::LocalSet};

use crate::{
    EvalLimits, EvalModule, RunnerConfig, VisualizerHook,
    cases::DEFAULT_FULL_AGENT_MODULES,
    gui::{
        accept_visualizer_connection, spawn_visualizer_gui, wait_for_visualizer_exit_with_context,
    },
    runner::{
        LiveReporter, ReplicaHardCap, action_module_ids, apply_visualizer_module_settings,
        build_eval_environment, emit_visualizer_blackboard_snapshot, emit_visualizer_memory_page,
        eval_registry, full_agent_allocation,
    },
};

const LIVE_TAB_ID: &str = "live";
const LIVE_TITLE: &str = "nuillu-server";
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone)]
pub struct LiveServerConfig {
    pub runner: RunnerConfig,
    pub state_dir: PathBuf,
    pub disabled_modules: Vec<EvalModule>,
    pub participants: Vec<String>,
}

pub fn run_live_with_visualizer(config: LiveServerConfig) -> anyhow::Result<()> {
    fs::create_dir_all(&config.state_dir)
        .with_context(|| format!("create state dir {}", config.state_dir.display()))?;
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("bind visualizer RPC listener")?;
    let addr = listener
        .local_addr()
        .context("read visualizer RPC listener address")?;
    eprintln!("visualizer RPC listening on {addr}");
    listener
        .set_nonblocking(true)
        .context("set visualizer RPC listener nonblocking")?;
    let mut child = spawn_visualizer_gui(&addr.to_string())?;
    eprintln!("visualizer process started pid={}", child.id());
    let stream = accept_visualizer_connection(&listener, &mut child)?;
    eprintln!("visualizer RPC connected");
    let port = VisualizerServerPort::from_stream(stream).context("open visualizer RPC port")?;
    port.send(VisualizerServerMessage::hello())
        .context("send visualizer protocol hello")?;
    let _ = port.recv();

    let tab_id = VisualizerTabId::new(LIVE_TAB_ID.to_string());
    port.send(VisualizerServerMessage::event(VisualizerEvent::OpenTab {
        tab_id: tab_id.clone(),
        title: LIVE_TITLE.to_string(),
    }))
    .context("open live tab")?;
    port.send(VisualizerServerMessage::event(
        VisualizerEvent::SetTabStatus {
            tab_id,
            status: TabStatus::Running,
        },
    ))
    .context("set live tab status")?;

    let (command_rx, event_tx) = port.into_channels();
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build live tokio runtime")?;
    let local = LocalSet::new();
    let mut hook = VisualizerHook::new(event_tx, command_rx);
    let result = runtime.block_on(local.run_until(run_live(config, &mut hook)));
    if let Err(error) = &result {
        eprintln!("nuillu-server runtime failed: {error:#}");
        let tab_id = VisualizerTabId::new(LIVE_TAB_ID.to_string());
        hook.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("nuillu-server runtime failed: {error:#}"),
        });
        hook.send_event(VisualizerEvent::SetTabStatus {
            tab_id,
            status: TabStatus::Invalid,
        });
    }
    let wait_context = if result.is_ok() {
        "nuillu-server runtime stopped"
    } else {
        "nuillu-server runtime failed"
    };
    wait_for_visualizer_exit_with_context(child, wait_context);
    result
}

async fn run_live(config: LiveServerConfig, visualizer: &mut VisualizerHook) -> anyhow::Result<()> {
    let tab_id = VisualizerTabId::new(LIVE_TAB_ID.to_string());
    let mut ambient = AmbientRows::load(config.state_dir.join("ambient-sensory.json"))?;
    let mut module_settings =
        LiveModuleSettings::load(config.state_dir.join("module-settings.json"))?;
    visualizer.send_event(VisualizerEvent::AmbientSensoryRows {
        tab_id: tab_id.clone(),
        rows: ambient.rows.clone(),
    });

    let modules = DEFAULT_FULL_AGENT_MODULES.to_vec();
    let allocation = full_agent_allocation(&EvalLimits::default(), &modules);
    let reporter = LiveReporter::new_with_log_context(
        &config.runner.run_id,
        &config.state_dir,
        "nuillu-server",
        "tab",
    )?;
    let env = build_eval_environment(
        &config.state_dir,
        &config.runner,
        allocation,
        None,
        action_module_ids(&modules),
        Arc::new(NoopFileSearchProvider),
        None,
        LIVE_TAB_ID,
        &reporter,
        Some(visualizer.event_sender()),
    )
    .await?;
    env.caps
        .scene()
        .set(config.participants.iter().map(Participant::new));
    for module in &config.disabled_modules {
        env.blackboard
            .apply(BlackboardCommand::SetModuleForcedDisabled {
                module: module.module_id(),
                disabled: true,
            })
            .await;
    }

    emit_visualizer_blackboard_snapshot(LIVE_TAB_ID, &env.blackboard, Some(visualizer)).await;
    emit_visualizer_memory_page(
        LIVE_TAB_ID,
        visualizer,
        &env.blackboard,
        env.memory.as_ref(),
        0,
        25,
    )
    .await;

    let sensory = env.caps.host_io().sensory_input_mailbox();
    let allocation_updates = env.caps.host_io().allocation_updated_mailbox();
    let mut restart_count = 0_u64;
    loop {
        let allocated = eval_registry(
            &modules,
            &env.memory_caps,
            &env.policy_caps,
            &env.file_search,
            &env.utterance_sink,
            ReplicaHardCap::V1Max,
        )
        .build(&env.caps)
        .await?;
        apply_persisted_module_settings(
            &module_settings,
            visualizer,
            &tab_id,
            &env.blackboard,
            &allocation_updates,
        )
        .await;

        let result = run_agent(
            allocated,
            AgentEventLoopConfig {
                idle_threshold: Duration::from_secs(1),
                activate_retries: 2,
            },
            drive_live_until_shutdown(
                visualizer,
                &tab_id,
                &mut ambient,
                &mut module_settings,
                &sensory,
                &allocation_updates,
                &env,
            ),
        )
        .await;

        match result {
            Ok(()) if visualizer.shutdown_requested() => break,
            Ok(()) => {
                restart_count = restart_count.saturating_add(1);
                let message = format!(
                    "agent runtime ended without a GUI shutdown; restarting attempt={restart_count}"
                );
                eprintln!("nuillu-server {message}");
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message,
                });
            }
            Err(error) => {
                restart_count = restart_count.saturating_add(1);
                let message =
                    format!("agent runtime error; restarting attempt={restart_count}: {error}");
                eprintln!("nuillu-server {message}");
                visualizer.send_event(VisualizerEvent::Log {
                    tab_id: tab_id.clone(),
                    message,
                });
            }
        }

        if visualizer.shutdown_requested() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    visualizer.send_event(VisualizerEvent::SetTabStatus {
        tab_id,
        status: TabStatus::Stopped,
    });
    Ok(())
}

async fn drive_live_until_shutdown(
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    ambient: &mut AmbientRows,
    module_settings: &mut LiveModuleSettings,
    sensory: &SensoryInputMailbox,
    allocation_updates: &nuillu_module::AllocationUpdatedMailbox,
    env: &crate::runner::EvalEnvironment,
) {
    publish_ambient_snapshot(ambient, sensory, visualizer, tab_id, env.clock.as_ref()).await;
    loop {
        if visualizer.shutdown_requested() {
            break;
        }
        while let Some(message) = visualizer.try_recv_command() {
            if handle_live_visualizer_message(
                message,
                visualizer,
                tab_id,
                ambient,
                module_settings,
                sensory,
                allocation_updates,
                env,
            )
            .await
            {
                break;
            }
        }
        if visualizer.shutdown_requested() {
            break;
        }
        emit_visualizer_blackboard_snapshot(LIVE_TAB_ID, &env.blackboard, Some(visualizer)).await;
        tokio::time::sleep(SNAPSHOT_INTERVAL).await;
    }
}

async fn handle_live_visualizer_message(
    message: VisualizerClientMessage,
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    ambient: &mut AmbientRows,
    module_settings: &mut LiveModuleSettings,
    sensory: &SensoryInputMailbox,
    allocation_updates: &nuillu_module::AllocationUpdatedMailbox,
    env: &crate::runner::EvalEnvironment,
) -> bool {
    let command = match message {
        VisualizerClientMessage::Hello { .. } => return false,
        VisualizerClientMessage::InvokeAction { .. } => return false,
        VisualizerClientMessage::Command { command } => command,
    };
    match command {
        VisualizerCommand::Shutdown => {
            visualizer.request_shutdown();
            true
        }
        VisualizerCommand::SendSensoryInput {
            tab_id: command_tab,
            input,
        } if command_tab == *tab_id => {
            let body = SensoryInput::Observed {
                modality: SensoryModality::parse(input.modality),
                direction: None,
                content: input.content,
                observed_at: env.clock.now(),
            };
            let _ = sensory.publish(body.clone()).await;
            visualizer.send_event(VisualizerEvent::SensoryInput {
                tab_id: tab_id.clone(),
                input: body,
            });
            false
        }
        VisualizerCommand::CreateAmbientSensoryRow {
            tab_id: command_tab,
            modality,
            content,
            disabled,
        } if command_tab == *tab_id => {
            ambient.create(modality, content, disabled);
            persist_and_emit_ambient(ambient, visualizer, tab_id, sensory, env.clock.as_ref())
                .await;
            false
        }
        VisualizerCommand::UpdateAmbientSensoryRow {
            tab_id: command_tab,
            row,
        } if command_tab == *tab_id => {
            ambient.update(row);
            persist_and_emit_ambient(ambient, visualizer, tab_id, sensory, env.clock.as_ref())
                .await;
            false
        }
        VisualizerCommand::RemoveAmbientSensoryRow {
            tab_id: command_tab,
            row_id,
        } if command_tab == *tab_id => {
            ambient.remove(&row_id);
            persist_and_emit_ambient(ambient, visualizer, tab_id, sensory, env.clock.as_ref())
                .await;
            false
        }
        VisualizerCommand::SetModuleDisabled {
            tab_id: command_tab,
            module,
            disabled,
        } if command_tab == *tab_id => {
            match ModuleId::new(module.clone()) {
                Ok(module_id) => {
                    env.blackboard
                        .apply(BlackboardCommand::SetModuleForcedDisabled {
                            module: module_id,
                            disabled,
                        })
                        .await;
                }
                Err(_) => {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: format!("invalid module id: {module}"),
                    });
                }
            }
            false
        }
        VisualizerCommand::SetModuleSettings {
            tab_id: command_tab,
            settings,
        } if command_tab == *tab_id => {
            if apply_visualizer_module_settings(
                tab_id,
                visualizer,
                &env.blackboard,
                allocation_updates,
                settings.clone(),
            )
            .await
            {
                module_settings.upsert(settings);
                if let Err(error) = module_settings.save() {
                    visualizer.send_event(VisualizerEvent::Log {
                        tab_id: tab_id.clone(),
                        message: format!("failed to save module settings: {error}"),
                    });
                }
            }
            false
        }
        VisualizerCommand::QueryMemory {
            tab_id: command_tab,
            query,
            limit,
        } if command_tab == *tab_id => {
            let records = env
                .memory
                .search(&MemoryQuery {
                    text: query.clone(),
                    limit,
                })
                .await
                .map(|records| records.into_iter().map(memory_record_view).collect())
                .unwrap_or_default();
            visualizer.send_event(VisualizerEvent::MemoryQueryResult {
                tab_id: tab_id.clone(),
                query,
                records,
            });
            false
        }
        VisualizerCommand::ListMemories {
            tab_id: command_tab,
            page,
            per_page,
        } if command_tab == *tab_id => {
            let records = list_all_memories(env.memory.as_ref()).await;
            visualizer.send_event(VisualizerEvent::MemoryPage {
                tab_id: tab_id.clone(),
                page: memory_page_from_records(&records, page, per_page),
            });
            false
        }
        _ => false,
    }
}

async fn persist_and_emit_ambient(
    ambient: &AmbientRows,
    visualizer: &mut VisualizerHook,
    tab_id: &VisualizerTabId,
    sensory: &SensoryInputMailbox,
    clock: &dyn nuillu_module::ports::Clock,
) {
    if let Err(error) = ambient.save() {
        visualizer.send_event(VisualizerEvent::Log {
            tab_id: tab_id.clone(),
            message: format!("failed to save ambient sensory rows: {error}"),
        });
    }
    visualizer.send_event(VisualizerEvent::AmbientSensoryRows {
        tab_id: tab_id.clone(),
        rows: ambient.rows.clone(),
    });
    publish_ambient_snapshot(ambient, sensory, visualizer, tab_id, clock).await;
}

async fn publish_ambient_snapshot(
    ambient: &AmbientRows,
    sensory: &SensoryInputMailbox,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    clock: &dyn nuillu_module::ports::Clock,
) {
    let entries = ambient
        .rows
        .iter()
        .filter(|row| !row.disabled && !row.content.trim().is_empty())
        .map(|row| AmbientSensoryEntry {
            id: row.id.clone(),
            modality: SensoryModality::parse(&row.modality),
            content: row.content.clone(),
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return;
    }
    let body = SensoryInput::AmbientSnapshot {
        entries,
        observed_at: clock.now(),
    };
    let _ = sensory.publish(body.clone()).await;
    visualizer.send_event(VisualizerEvent::SensoryInput {
        tab_id: tab_id.clone(),
        input: body,
    });
}

async fn apply_persisted_module_settings(
    settings: &LiveModuleSettings,
    visualizer: &VisualizerHook,
    tab_id: &VisualizerTabId,
    blackboard: &nuillu_blackboard::Blackboard,
    allocation_updates: &nuillu_module::AllocationUpdatedMailbox,
) {
    for setting in settings.iter() {
        apply_visualizer_module_settings(
            tab_id,
            visualizer,
            blackboard,
            allocation_updates,
            setting.clone(),
        )
        .await;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModuleSettingsFile {
    modules: Vec<ModuleSettingsView>,
}

#[derive(Debug)]
struct LiveModuleSettings {
    path: PathBuf,
    modules: std::collections::BTreeMap<String, ModuleSettingsView>,
}

impl LiveModuleSettings {
    fn load(path: PathBuf) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self {
                path,
                modules: std::collections::BTreeMap::new(),
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

    fn save(&self) -> anyhow::Result<()> {
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

    fn upsert(&mut self, settings: ModuleSettingsView) {
        self.modules.insert(settings.module.clone(), settings);
    }

    fn iter(&self) -> impl Iterator<Item = &ModuleSettingsView> {
        self.modules.values()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AmbientRowsFile {
    rows: Vec<AmbientSensoryRowView>,
}

#[derive(Debug)]
struct AmbientRows {
    path: PathBuf,
    rows: Vec<AmbientSensoryRowView>,
}

impl AmbientRows {
    fn load(path: PathBuf) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self {
                path,
                rows: Vec::new(),
            });
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read ambient sensory rows from {}", path.display()))?;
        let file: AmbientRowsFile = serde_json::from_str(&text)
            .with_context(|| format!("parse ambient sensory rows from {}", path.display()))?;
        Ok(Self {
            path,
            rows: file.rows,
        })
    }

    fn save(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create ambient state dir {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(&AmbientRowsFile {
            rows: self.rows.clone(),
        })?;
        fs::write(&self.path, text)
            .with_context(|| format!("write ambient sensory rows to {}", self.path.display()))
    }

    fn create(&mut self, modality: String, content: String, disabled: bool) {
        let id = self.next_id();
        self.rows.push(AmbientSensoryRowView {
            id,
            modality,
            content,
            disabled,
        });
    }

    fn update(&mut self, row: AmbientSensoryRowView) {
        if let Some(existing) = self.rows.iter_mut().find(|existing| existing.id == row.id) {
            *existing = row;
        }
    }

    fn remove(&mut self, row_id: &str) {
        self.rows.retain(|row| row.id != row_id);
    }

    fn next_id(&self) -> String {
        let mut index = self.rows.len().saturating_add(1);
        loop {
            let id = format!("ambient-{index}");
            if self.rows.iter().all(|row| row.id != id) {
                return id;
            }
            index = index.saturating_add(1);
        }
    }
}

async fn list_all_memories(memory: &dyn MemoryStore) -> Vec<MemoryRecordView> {
    let mut out = Vec::new();
    for rank in [
        MemoryRank::Identity,
        MemoryRank::Permanent,
        MemoryRank::LongTerm,
        MemoryRank::MidTerm,
        MemoryRank::ShortTerm,
    ] {
        if let Ok(records) = memory.list_by_rank(rank).await {
            out.extend(records.into_iter().map(memory_record_view));
        }
    }
    out.sort_by(|left, right| {
        right
            .occurred_at
            .cmp(&left.occurred_at)
            .then_with(|| left.index.cmp(&right.index))
    });
    out
}

fn memory_record_view(record: MemoryRecord) -> MemoryRecordView {
    MemoryRecordView {
        index: record.index.as_str().to_string(),
        rank: format!("{:?}", record.rank),
        occurred_at: record.occurred_at,
        content: record.content.as_str().to_string(),
    }
}
