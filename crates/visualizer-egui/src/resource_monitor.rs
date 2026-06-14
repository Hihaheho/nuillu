use std::collections::{BTreeMap, BTreeSet, VecDeque};

use egui::Color32;
use egui_plot::{Line, Plot, PlotPoint, PlotPoints};
use nuillu_module::RuntimeEvent;

use crate::{
    AllocationView, BlackboardSnapshot, InteroceptionView, ModulePolicyView,
    i18n::{EguiI18nExt as _, I18nArg},
    module_filter,
    module_filter::ModuleFilterState,
};

const MAX_HISTORY_SECS: f64 = 15.0 * 60.0;
const MODULE_PLOT_HEIGHT: f32 = 220.0;
const INTEROCEPTION_PLOT_HEIGHT: f32 = 190.0;
const THROUGHPUT_PLOT_HEIGHT: f32 = 190.0;

#[derive(Debug, Clone)]
pub struct ResourceMonitorState {
    snapshots: VecDeque<ResourceSnapshotSample>,
    buckets: VecDeque<RuntimeEventBucket>,
    window: ResourceMonitorWindow,
}

impl Default for ResourceMonitorState {
    fn default() -> Self {
        Self {
            snapshots: VecDeque::new(),
            buckets: VecDeque::new(),
            window: ResourceMonitorWindow::FiveMinutes,
        }
    }
}

impl ResourceMonitorState {
    pub fn record_snapshot_at(&mut self, snapshot: &BlackboardSnapshot, now_secs: f64) {
        let capacities = module_capacities(&snapshot.module_policies);
        let allocation = snapshot
            .allocation
            .iter()
            .map(|view| {
                (
                    view.module.clone(),
                    ModuleResourceSample::from_view(view, capacities.get(view.module.as_str())),
                )
            })
            .collect();
        self.snapshots.push_back(ResourceSnapshotSample {
            at_secs: now_secs,
            allocation,
            interoception: snapshot.interoception.clone(),
        });
        self.prune(now_secs);
    }

    pub fn record_runtime_event_at(&mut self, event: &RuntimeEvent, now_secs: f64) {
        let module = runtime_event_module(event);
        let second = now_secs.floor() as i64;
        if self
            .buckets
            .back()
            .is_none_or(|bucket| bucket.second != second)
        {
            self.buckets.push_back(RuntimeEventBucket {
                second,
                modules: BTreeMap::new(),
            });
        }
        let bucket = self
            .buckets
            .back_mut()
            .expect("runtime event bucket exists after push");
        let counts = bucket.modules.entry(module).or_default();
        match event {
            RuntimeEvent::LlmAccessed { .. } => counts.llm_accessed += 1,
            RuntimeEvent::LlmCompleted { .. } => counts.llm_completed += 1,
            RuntimeEvent::ModuleBatchThrottled { .. } => counts.throttles += 1,
            RuntimeEvent::ModuleBatchReady { .. } => counts.batch_ready += 1,
            RuntimeEvent::ModuleActivationCompleted { succeeded, .. } => {
                if *succeeded {
                    counts.activations_completed += 1;
                } else {
                    counts.activations_failed += 1;
                }
            }
            RuntimeEvent::ModuleActivationAttemptFailed { .. } => {
                counts.activations_failed += 1;
            }
            RuntimeEvent::SessionCompactionStarted { .. } => counts.compactions_started += 1,
            RuntimeEvent::SessionCompactionCompleted { .. } => counts.compactions_completed += 1,
            RuntimeEvent::SessionCompactionFailed { .. } => counts.compactions_failed += 1,
            RuntimeEvent::MemoUpdated { .. }
            | RuntimeEvent::ModuleTaskFailed { .. }
            | RuntimeEvent::ModuleWarning { .. } => {}
        }
        self.prune(now_secs);
    }

    pub fn module_names(&self) -> Vec<String> {
        self.module_name_set().into_iter().collect()
    }

    fn module_name_set(&self) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for sample in &self.snapshots {
            names.extend(sample.allocation.keys().cloned());
        }
        for bucket in &self.buckets {
            names.extend(bucket.modules.keys().cloned());
        }
        names
    }

    fn prune(&mut self, now_secs: f64) {
        let cutoff = now_secs - MAX_HISTORY_SECS;
        while self
            .snapshots
            .front()
            .is_some_and(|sample| sample.at_secs < cutoff)
        {
            self.snapshots.pop_front();
        }
        while self
            .buckets
            .front()
            .is_some_and(|bucket| (bucket.second as f64) < cutoff)
        {
            self.buckets.pop_front();
        }
    }

    fn recent_snapshot_count(&self, now_secs: f64) -> usize {
        let cutoff = now_secs - self.window.secs();
        self.snapshots
            .iter()
            .filter(|sample| sample.at_secs >= cutoff)
            .count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceMonitorWindow {
    OneMinute,
    FiveMinutes,
    FifteenMinutes,
}

impl ResourceMonitorWindow {
    fn secs(self) -> f64 {
        match self {
            Self::OneMinute => 60.0,
            Self::FiveMinutes => 5.0 * 60.0,
            Self::FifteenMinutes => 15.0 * 60.0,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::OneMinute => "1m",
            Self::FiveMinutes => "5m",
            Self::FifteenMinutes => "15m",
        }
    }
}

const WINDOWS: [ResourceMonitorWindow; 3] = [
    ResourceMonitorWindow::OneMinute,
    ResourceMonitorWindow::FiveMinutes,
    ResourceMonitorWindow::FifteenMinutes,
];

#[derive(Debug, Clone, PartialEq)]
struct ResourceSnapshotSample {
    at_secs: f64,
    allocation: BTreeMap<String, ModuleResourceSample>,
    interoception: InteroceptionView,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ModuleResourceSample {
    activation_ratio: f64,
    active_replicas: u8,
    bpm: Option<f64>,
    period_ms: Option<u64>,
    replica_capacity: u8,
}

impl ModuleResourceSample {
    fn from_view(view: &AllocationView, capacity: Option<&u8>) -> Self {
        Self {
            activation_ratio: view.activation_ratio,
            active_replicas: view.active_replicas,
            bpm: view.bpm,
            period_ms: view.period_ms,
            replica_capacity: capacity.copied().unwrap_or(view.active_replicas.max(1)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RuntimeEventBucket {
    second: i64,
    modules: BTreeMap<String, ModuleRuntimeCounts>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ModuleRuntimeCounts {
    batch_ready: u32,
    activations_completed: u32,
    activations_failed: u32,
    throttles: u32,
    llm_accessed: u32,
    llm_completed: u32,
    compactions_started: u32,
    compactions_completed: u32,
    compactions_failed: u32,
}

impl ModuleRuntimeCounts {
    fn add(self, other: Self) -> Self {
        Self {
            batch_ready: self.batch_ready + other.batch_ready,
            activations_completed: self.activations_completed + other.activations_completed,
            activations_failed: self.activations_failed + other.activations_failed,
            throttles: self.throttles + other.throttles,
            llm_accessed: self.llm_accessed + other.llm_accessed,
            llm_completed: self.llm_completed + other.llm_completed,
            compactions_started: self.compactions_started + other.compactions_started,
            compactions_completed: self.compactions_completed + other.compactions_completed,
            compactions_failed: self.compactions_failed + other.compactions_failed,
        }
    }

    fn compactions(self) -> u32 {
        self.compactions_started + self.compactions_completed + self.compactions_failed
    }
}

pub fn ui(
    ui: &mut egui::Ui,
    state: &mut ResourceMonitorState,
    filter: &mut ModuleFilterState,
    modules: &[String],
    now_secs: f64,
) {
    state.prune(now_secs);

    ui.horizontal_wrapped(|ui| {
        ui.heading(ui.ctx().tr("resource-monitor-heading"));
        ui.label(ui.ctx().tr_args(
            "resource-monitor-samples",
            &[("count", state.recent_snapshot_count(now_secs).into())],
        ));
        ui.label(
            ui.ctx()
                .tr_args("modules-count", &[("count", modules.len().into())]),
        );
        ui.separator();
        ui.label(ui.ctx().tr("resource-monitor-window"));
        for window in WINDOWS {
            ui.selectable_value(&mut state.window, window, window.label());
        }
    });
    ui.separator();

    module_filter::render_module_filter(ui, "resource-monitor-module-filter", filter, modules);
    ui.add_space(4.0);

    let selected_modules = modules
        .iter()
        .filter(|module| filter.is_selected(module))
        .cloned()
        .collect::<Vec<_>>();
    if selected_modules.is_empty() {
        ui.label(ui.ctx().tr("resource-monitor-no-modules-selected"));
        return;
    }

    if state.snapshots.is_empty() && state.buckets.is_empty() {
        ui.label(ui.ctx().tr("resource-monitor-empty"));
        return;
    }

    egui::ScrollArea::vertical()
        .id_salt("resource-monitor-scroll")
        .show(ui, |ui| {
            render_module_activity_plot(ui, state, &selected_modules, now_secs);
            ui.add_space(8.0);
            render_interoception_plot(ui, state, now_secs);
            ui.add_space(8.0);
            render_throughput_plot(ui, state, &selected_modules, now_secs);
            ui.add_space(4.0);
            render_latest_snapshot(ui, state);
        });
}

fn render_module_activity_plot(
    ui: &mut egui::Ui,
    state: &ResourceMonitorState,
    selected_modules: &[String],
    now_secs: f64,
) {
    ui.strong(ui.ctx().tr("resource-monitor-allocation-activity"));
    ui.label(ui.ctx().tr("resource-monitor-allocation-activity-help"));
    let activity_scale = module_activity_load_scale(state, selected_modules, now_secs);
    Plot::new("resource-monitor-module-activity")
        .height(MODULE_PLOT_HEIGHT)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_axis_zoom_drag(false)
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .default_x_bounds(-state.window.secs(), 0.0)
        .default_y_bounds(0.0, 1.2)
        .label_formatter(plot_hover_label)
        .show(ui, |plot_ui| {
            plot_ui.set_plot_bounds_x(-state.window.secs()..=0.0);
            plot_ui.set_plot_bounds_y(0.0..=1.2);
            for (index, module) in selected_modules.iter().enumerate() {
                let color = module_color(index);
                let activation = module_activation_points(state, module, now_secs);
                if !activation.is_empty() {
                    plot_ui.line(
                        Line::new(format!("{module} allocation"), activation)
                            .color(color)
                            .width(1.4),
                    );
                }

                let activity = module_activity_load_points(state, module, now_secs, activity_scale);
                if !activity.is_empty() {
                    plot_ui.line(
                        Line::new(format!("{module} runtime load"), activity)
                            .color(translucent(color))
                            .width(0.9),
                    );
                }
            }
        });
}

pub fn render_interoception_plot(ui: &mut egui::Ui, state: &ResourceMonitorState, now_secs: f64) {
    ui.strong(ui.ctx().tr("resource-monitor-interoception"));
    ui.label(ui.ctx().tr("resource-monitor-interoception-help"));
    Plot::new("resource-monitor-interoception")
        .height(INTEROCEPTION_PLOT_HEIGHT)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_axis_zoom_drag(false)
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .default_x_bounds(-state.window.secs(), 0.0)
        .default_y_bounds(0.0, 1.0)
        .label_formatter(plot_hover_label)
        .show(ui, |plot_ui| {
            plot_ui.set_plot_bounds_x(-state.window.secs()..=0.0);
            plot_ui.set_plot_bounds_y(0.0..=1.0);
            let lines = [
                (
                    "wake arousal",
                    Color32::from_rgb(64, 160, 224),
                    interoception_points(state, now_secs, |view| f64::from(view.wake_arousal)),
                ),
                (
                    "nrem pressure",
                    Color32::from_rgb(160, 132, 220),
                    interoception_points(state, now_secs, |view| f64::from(view.nrem_pressure)),
                ),
                (
                    "rem pressure",
                    Color32::from_rgb(224, 128, 96),
                    interoception_points(state, now_secs, |view| f64::from(view.rem_pressure)),
                ),
                (
                    "affect arousal",
                    Color32::from_rgb(232, 184, 64),
                    interoception_points(state, now_secs, |view| f64::from(view.affect_arousal)),
                ),
                (
                    "valence norm",
                    Color32::from_rgb(88, 184, 128),
                    interoception_points(state, now_secs, |view| {
                        (f64::from(view.valence) + 1.0) / 2.0
                    }),
                ),
            ];
            for (name, color, points) in lines {
                if !points.is_empty() {
                    plot_ui.line(Line::new(name, points).color(color).width(1.5));
                }
            }
        });
}

fn render_throughput_plot(
    ui: &mut egui::Ui,
    state: &ResourceMonitorState,
    selected_modules: &[String],
    now_secs: f64,
) {
    let y_max = throughput_y_max(state, selected_modules, now_secs);
    ui.strong(ui.ctx().tr("resource-monitor-throughput"));
    ui.label(ui.ctx().tr("resource-monitor-throughput-help"));
    Plot::new("resource-monitor-throughput")
        .height(THROUGHPUT_PLOT_HEIGHT)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_axis_zoom_drag(false)
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .default_x_bounds(-state.window.secs(), 0.0)
        .default_y_bounds(0.0, y_max)
        .label_formatter(plot_hover_label)
        .show(ui, |plot_ui| {
            plot_ui.set_plot_bounds_x(-state.window.secs()..=0.0);
            plot_ui.set_plot_bounds_y(0.0..=y_max);
            let lines = [
                (
                    "batches",
                    Color32::from_rgb(72, 150, 220),
                    aggregate_event_points(state, selected_modules, now_secs, |counts| {
                        counts.batch_ready
                    }),
                ),
                (
                    "activations",
                    Color32::from_rgb(92, 184, 124),
                    aggregate_event_points(state, selected_modules, now_secs, |counts| {
                        counts.activations_completed
                    }),
                ),
                (
                    "failed activations",
                    Color32::from_rgb(224, 80, 72),
                    aggregate_event_points(state, selected_modules, now_secs, |counts| {
                        counts.activations_failed
                    }),
                ),
                (
                    "throttles",
                    Color32::from_rgb(224, 168, 64),
                    aggregate_event_points(state, selected_modules, now_secs, |counts| {
                        counts.throttles
                    }),
                ),
                (
                    "llm access",
                    Color32::from_rgb(156, 112, 220),
                    aggregate_event_points(state, selected_modules, now_secs, |counts| {
                        counts.llm_accessed + counts.llm_completed
                    }),
                ),
                (
                    "compaction",
                    Color32::from_rgb(120, 168, 176),
                    aggregate_event_points(state, selected_modules, now_secs, |counts| {
                        counts.compactions()
                    }),
                ),
            ];
            for (name, color, points) in lines {
                if !points.is_empty() {
                    plot_ui.line(Line::new(name, points).color(color).width(1.5));
                }
            }
        });
}

fn render_latest_snapshot(ui: &mut egui::Ui, state: &ResourceMonitorState) {
    let Some(sample) = state.snapshots.back() else {
        return;
    };
    let interoception = &sample.interoception;
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.strong(ui.ctx().tr("resource-monitor-latest"));
        ui.label(ui.ctx().tr_args(
            "resource-monitor-mode",
            &[("mode", I18nArg::from(interoception.mode.as_str()))],
        ));
        ui.label(ui.ctx().tr_args(
            "resource-monitor-wake",
            &[("value", format!("{:.2}", interoception.wake_arousal).into())],
        ));
        ui.label(ui.ctx().tr_args(
            "resource-monitor-nrem",
            &[(
                "value",
                format!("{:.2}", interoception.nrem_pressure).into(),
            )],
        ));
        ui.label(ui.ctx().tr_args(
            "resource-monitor-rem",
            &[("value", format!("{:.2}", interoception.rem_pressure).into())],
        ));
        ui.label(ui.ctx().tr_args(
            "resource-monitor-affect",
            &[(
                "value",
                format!("{:.2}", interoception.affect_arousal).into(),
            )],
        ));
        ui.label(ui.ctx().tr_args(
            "resource-monitor-valence",
            &[("value", format!("{:.2}", interoception.valence).into())],
        ));
        if !interoception.emotion.is_empty() {
            ui.label(ui.ctx().tr_args(
                "resource-monitor-emotion",
                &[("emotion", I18nArg::from(interoception.emotion.as_str()))],
            ));
        }
    });

    let active_modules = sample
        .allocation
        .iter()
        .filter(|(_, allocation)| allocation.active_replicas > 0)
        .map(|(module, allocation)| {
            let bpm = allocation
                .bpm
                .map(|bpm| format!("{bpm:.1} bpm"))
                .unwrap_or_else(|| "- bpm".to_string());
            let period = allocation
                .period_ms
                .map(|ms| format!("{ms} ms"))
                .unwrap_or_else(|| "- ms".to_string());
            format!(
                "{} {:.2} {}/{} {} {}",
                module,
                allocation.activation_ratio,
                allocation.active_replicas,
                allocation.replica_capacity,
                bpm,
                period
            )
        })
        .collect::<Vec<_>>();
    if !active_modules.is_empty() {
        ui.label(ui.ctx().tr_args(
            "resource-monitor-active-modules",
            &[("modules", active_modules.join(", ").into())],
        ));
    }
}

fn plot_hover_label(name: &str, value: &PlotPoint) -> String {
    if name.is_empty() {
        String::new()
    } else {
        format!("{name}\n{:.1}s ago  {:.3}", -value.x, value.y)
    }
}

fn module_activation_points(
    state: &ResourceMonitorState,
    module: &str,
    now_secs: f64,
) -> PlotPoints<'static> {
    let cutoff = now_secs - state.window.secs();
    state
        .snapshots
        .iter()
        .filter(|sample| sample.at_secs >= cutoff)
        .filter_map(|sample| {
            sample
                .allocation
                .get(module)
                .map(|allocation| [sample.at_secs - now_secs, allocation.activation_ratio])
        })
        .collect()
}

fn module_activity_load_scale(
    state: &ResourceMonitorState,
    selected_modules: &[String],
    now_secs: f64,
) -> f64 {
    let cutoff = now_secs - state.window.secs();
    let selected = selected_modules
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let max_count = state
        .buckets
        .iter()
        .filter(|bucket| (bucket.second as f64) >= cutoff)
        .flat_map(|bucket| {
            bucket
                .modules
                .iter()
                .filter(|(module, _)| selected.contains(module.as_str()))
                .map(|(_, counts)| module_activity_load_count(*counts))
        })
        .max()
        .unwrap_or(1);
    f64::from(max_count.max(1))
}

fn module_activity_load_points(
    state: &ResourceMonitorState,
    module: &str,
    now_secs: f64,
    scale: f64,
) -> PlotPoints<'static> {
    let cutoff = now_secs - state.window.secs();
    let counts_by_second = state
        .buckets
        .iter()
        .filter(|bucket| (bucket.second as f64) >= cutoff)
        .filter_map(|bucket| {
            bucket
                .modules
                .get(module)
                .copied()
                .map(|counts| (bucket.second, module_activity_load_count(counts)))
        })
        .filter(|(_, count)| *count > 0)
        .collect::<BTreeMap<_, _>>();
    if counts_by_second.is_empty() {
        return Vec::<[f64; 2]>::new().into();
    }

    let start = cutoff.floor() as i64;
    let end = now_secs.floor() as i64;
    let scale = scale.max(1.0);
    (start..=end)
        .map(|second| {
            let count = counts_by_second.get(&second).copied().unwrap_or_default();
            [second as f64 + 0.5 - now_secs, f64::from(count) / scale]
        })
        .collect()
}

fn module_activity_load_count(counts: ModuleRuntimeCounts) -> u32 {
    counts.batch_ready + counts.activations_completed + counts.activations_failed
}

fn translucent(color: Color32) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 130)
}

fn interoception_points(
    state: &ResourceMonitorState,
    now_secs: f64,
    value: impl Fn(&InteroceptionView) -> f64,
) -> PlotPoints<'static> {
    let cutoff = now_secs - state.window.secs();
    state
        .snapshots
        .iter()
        .filter(|sample| sample.at_secs >= cutoff)
        .map(|sample| [sample.at_secs - now_secs, value(&sample.interoception)])
        .collect()
}

fn throughput_y_max(
    state: &ResourceMonitorState,
    selected_modules: &[String],
    now_secs: f64,
) -> f64 {
    let cutoff = now_secs - state.window.secs();
    let selected = selected_modules
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let max_count = state
        .buckets
        .iter()
        .filter(|bucket| (bucket.second as f64) >= cutoff)
        .map(|bucket| {
            bucket
                .modules
                .iter()
                .filter(|(module, _)| selected.contains(module.as_str()))
                .fold(ModuleRuntimeCounts::default(), |acc, (_, counts)| {
                    acc.add(*counts)
                })
        })
        .fold(1, |max_count, counts| {
            max_count
                .max(counts.batch_ready)
                .max(counts.activations_completed)
                .max(counts.activations_failed)
                .max(counts.throttles)
                .max(counts.llm_accessed + counts.llm_completed)
                .max(counts.compactions())
        });
    f64::from(max_count) * 1.15
}

fn aggregate_event_points(
    state: &ResourceMonitorState,
    selected_modules: &[String],
    now_secs: f64,
    value: impl Fn(ModuleRuntimeCounts) -> u32,
) -> PlotPoints<'static> {
    let cutoff = now_secs - state.window.secs();
    let selected = selected_modules
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    state
        .buckets
        .iter()
        .filter(|bucket| (bucket.second as f64) >= cutoff)
        .map(|bucket| {
            let counts = bucket
                .modules
                .iter()
                .filter(|(module, _)| selected.contains(module.as_str()))
                .fold(ModuleRuntimeCounts::default(), |acc, (_, counts)| {
                    acc.add(*counts)
                });
            [
                bucket.second as f64 + 0.5 - now_secs,
                f64::from(value(counts)),
            ]
        })
        .collect()
}

fn module_capacities(policies: &[ModulePolicyView]) -> BTreeMap<&str, u8> {
    policies
        .iter()
        .map(|policy| (policy.module.as_str(), policy.replica_capacity))
        .collect()
}

fn runtime_event_module(event: &RuntimeEvent) -> String {
    match event {
        RuntimeEvent::LlmAccessed { owner, .. }
        | RuntimeEvent::LlmCompleted { owner, .. }
        | RuntimeEvent::MemoUpdated { owner, .. }
        | RuntimeEvent::ModuleBatchThrottled { owner, .. }
        | RuntimeEvent::ModuleBatchReady { owner, .. }
        | RuntimeEvent::ModuleActivationCompleted { owner, .. }
        | RuntimeEvent::ModuleActivationAttemptFailed { owner, .. }
        | RuntimeEvent::ModuleTaskFailed { owner, .. }
        | RuntimeEvent::ModuleWarning { owner, .. }
        | RuntimeEvent::SessionCompactionStarted { owner, .. }
        | RuntimeEvent::SessionCompactionCompleted { owner, .. }
        | RuntimeEvent::SessionCompactionFailed { owner, .. } => owner.module.as_str().to_owned(),
    }
}

fn module_color(index: usize) -> Color32 {
    const PALETTE: [Color32; 12] = [
        Color32::from_rgb(64, 152, 224),
        Color32::from_rgb(224, 120, 84),
        Color32::from_rgb(88, 176, 112),
        Color32::from_rgb(184, 128, 216),
        Color32::from_rgb(224, 184, 72),
        Color32::from_rgb(88, 184, 184),
        Color32::from_rgb(208, 96, 128),
        Color32::from_rgb(144, 168, 88),
        Color32::from_rgb(120, 128, 232),
        Color32::from_rgb(232, 144, 64),
        Color32::from_rgb(112, 200, 152),
        Color32::from_rgb(184, 152, 112),
    ];
    PALETTE[index % PALETTE.len()]
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use nuillu_types::{ModuleId, ModuleInstanceId, ReplicaIndex};

    use super::*;

    #[test]
    fn records_snapshot_samples_and_modules() {
        let mut state = ResourceMonitorState::default();
        state.record_snapshot_at(
            &snapshot_with_allocation("sensory", 0.5, 1, 2, 12.0, 5_000),
            1.0,
        );

        assert_eq!(state.snapshots.len(), 1);
        assert_eq!(state.module_names(), vec!["sensory".to_string()]);
        assert_eq!(
            state.snapshots[0].allocation["sensory"],
            ModuleResourceSample {
                activation_ratio: 0.5,
                active_replicas: 1,
                bpm: Some(12.0),
                period_ms: Some(5_000),
                replica_capacity: 2,
            }
        );
    }

    #[test]
    fn buckets_runtime_events_by_second_and_module() {
        let mut state = ResourceMonitorState::default();
        let owner = owner("sensory");
        state.record_runtime_event_at(
            &RuntimeEvent::ModuleBatchReady {
                sequence: 1,
                owner: owner.clone(),
                batch_type: "Batch".to_string(),
                batch_debug: "Batch".to_string(),
            },
            2.2,
        );
        state.record_runtime_event_at(
            &RuntimeEvent::ModuleActivationCompleted {
                sequence: 2,
                owner: owner.clone(),
                duration: std::time::Duration::from_millis(10),
                succeeded: true,
            },
            2.8,
        );
        state.record_runtime_event_at(
            &RuntimeEvent::ModuleActivationCompleted {
                sequence: 3,
                owner,
                duration: std::time::Duration::from_millis(10),
                succeeded: false,
            },
            3.0,
        );

        assert_eq!(state.buckets.len(), 2);
        assert_eq!(state.buckets[0].second, 2);
        assert_eq!(state.buckets[0].modules["sensory"].batch_ready, 1);
        assert_eq!(state.buckets[0].modules["sensory"].activations_completed, 1);
        assert_eq!(state.buckets[1].modules["sensory"].activations_failed, 1);
    }

    #[test]
    fn throughput_y_max_uses_selected_modules_in_window() {
        let mut state = ResourceMonitorState::default();
        let sensory = owner("sensory");
        let reward = owner("reward");
        for sequence in 1..=3 {
            state.record_runtime_event_at(
                &RuntimeEvent::ModuleBatchReady {
                    sequence,
                    owner: sensory.clone(),
                    batch_type: "Batch".to_string(),
                    batch_debug: "Batch".to_string(),
                },
                10.2,
            );
        }
        for sequence in 4..=8 {
            state.record_runtime_event_at(
                &RuntimeEvent::ModuleBatchReady {
                    sequence,
                    owner: reward.clone(),
                    batch_type: "Batch".to_string(),
                    batch_debug: "Batch".to_string(),
                },
                10.4,
            );
        }

        let y_max = throughput_y_max(&state, &["sensory".to_string()], 11.0);
        assert!((y_max - 3.45).abs() < 0.000_001);

        let hidden_y_max = throughput_y_max(&state, &["speak".to_string()], 11.0);
        assert!((hidden_y_max - 1.15).abs() < 0.000_001);
    }

    #[test]
    fn module_activity_load_points_are_normalized_event_density() {
        let mut state = ResourceMonitorState::default();
        let sensory = owner("sensory");
        let reward = owner("reward");
        state.record_runtime_event_at(
            &RuntimeEvent::ModuleBatchReady {
                sequence: 1,
                owner: sensory.clone(),
                batch_type: "Batch".to_string(),
                batch_debug: "Batch".to_string(),
            },
            10.1,
        );
        state.record_runtime_event_at(
            &RuntimeEvent::ModuleActivationCompleted {
                sequence: 2,
                owner: sensory,
                duration: std::time::Duration::from_millis(10),
                succeeded: true,
            },
            10.2,
        );
        for sequence in 3..=7 {
            state.record_runtime_event_at(
                &RuntimeEvent::ModuleBatchReady {
                    sequence,
                    owner: reward.clone(),
                    batch_type: "Batch".to_string(),
                    batch_debug: "Batch".to_string(),
                },
                10.3,
            );
        }

        let scale = module_activity_load_scale(&state, &["sensory".to_string()], 11.0);
        let points = module_activity_load_points(&state, "sensory", 11.0, scale);

        assert_eq!(scale, 2.0);
        assert!(points.points().iter().any(|point| {
            (point.x - -0.5).abs() < 0.000_001 && (point.y - 1.0).abs() < 0.000_001
        }));
        assert!(module_activity_load_points(&state, "speak", 11.0, scale).is_empty());
    }

    #[test]
    fn prunes_history_to_fifteen_minutes() {
        let mut state = ResourceMonitorState::default();
        state.record_snapshot_at(
            &snapshot_with_allocation("sensory", 0.2, 0, 2, 1.0, 60_000),
            0.0,
        );
        state.record_snapshot_at(
            &snapshot_with_allocation("sensory", 0.4, 1, 2, 2.0, 30_000),
            100.0,
        );
        state.record_snapshot_at(
            &snapshot_with_allocation("sensory", 0.8, 2, 2, 4.0, 15_000),
            1000.0,
        );

        assert_eq!(state.snapshots.len(), 2);
        assert_eq!(state.snapshots[0].at_secs, 100.0);
    }

    fn snapshot_with_allocation(
        module: &str,
        activation_ratio: f64,
        active_replicas: u8,
        replica_capacity: u8,
        bpm: f64,
        period_ms: u64,
    ) -> BlackboardSnapshot {
        BlackboardSnapshot {
            allocation: vec![AllocationView {
                module: module.to_string(),
                activation_ratio,
                active_replicas,
                bpm: Some(bpm),
                period_ms: Some(period_ms),
                tier: "default".to_string(),
                guidance: String::new(),
            }],
            module_policies: vec![ModulePolicyView {
                module: module.to_string(),
                replica_min: 0,
                replica_max: replica_capacity,
                replica_capacity,
                bpm_min: 1.0,
                bpm_max: 60.0,
                zero_replica_window: crate::ZeroReplicaWindowView::Disabled,
            }],
            interoception: InteroceptionView {
                mode: "wake".to_string(),
                wake_arousal: 0.3,
                nrem_pressure: 0.2,
                rem_pressure: 0.1,
                affect_arousal: 0.4,
                valence: -0.2,
                emotion: "alert".to_string(),
                last_updated: DateTime::<Utc>::from_timestamp(1, 0).unwrap(),
            },
            ..BlackboardSnapshot::default()
        }
    }

    fn owner(module: &str) -> ModuleInstanceId {
        ModuleInstanceId::new(
            ModuleId::new(module.to_string()).unwrap(),
            ReplicaIndex::ZERO,
        )
    }
}
