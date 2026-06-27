pub mod blackboard;
pub mod chat;
pub mod cognition;
pub mod errors;
mod i18n;
pub mod memories;
pub mod memos;
pub mod module_filter;
pub mod modules;
pub mod resource_monitor;
pub mod text;
pub mod window;

pub use eframe;
pub use egui;
pub use egui_hooks;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use egui_hooks::UseHookExt as _;
use font_kit::family_name::FamilyName;
use font_kit::handle::Handle;
use font_kit::properties::{Properties, Weight};
use font_kit::source::SystemSource;
use i18n::{EguiI18nExt as _, I18nArg, I18nCatalog, LOCALE_PERSISTENCE_KEY, Locale};
use nuillu_module::{ActionAffordance, RuntimeEvent};
pub use nuillu_visualizer_protocol::*;

const NOTO_SANS_JP_FONT_KEY: &str = "noto-sans-jp";
const NOTO_SANS_JP_FAMILY_NAME: &str = "Noto Sans JP";
const NOTO_SANS_JP_FONT_WEIGHT: Weight = Weight::MEDIUM;
const THEME_PERSISTENCE_KEY: &str = "visualizer-theme";
const ZOOM_FACTOR_PERSISTENCE_KEY: &str = "visualizer-zoom-factor";
const DEFAULT_ZOOM_FACTOR: f32 = 1.0;
const MIN_ZOOM_FACTOR: f32 = 0.5;
const MAX_ZOOM_FACTOR: f32 = 2.0;
const ZOOM_BUTTON_STEP_PERCENT: f32 = 1.0;
const ZOOM_BUTTON_DOUBLE_CLICK_TOTAL_PERCENT: f32 = 10.0;
const ZOOM_SYNC_EPSILON: f32 = 0.000_1;

pub struct VisualizerChannels {
    pub server_messages: Receiver<VisualizerServerMessage>,
    pub client_messages: Sender<VisualizerClientMessage>,
    pub remote: bool,
}

pub struct VisualizerApp {
    server_messages: Receiver<VisualizerServerMessage>,
    client_messages: Sender<VisualizerClientMessage>,
    remote: bool,
    i18n_catalog: I18nCatalog,
    current_locale: Locale,
    zoom_persistence_applied: bool,
    zoom_percent_input: String,
    zoom_percent_input_dirty: bool,
    zoom_percent_input_focused: bool,
    state: VisualizerState,
}

impl VisualizerApp {
    pub fn new(cc: &eframe::CreationContext<'_>, channels: VisualizerChannels) -> Self {
        install_visualizer_fonts(&cc.egui_ctx);
        install_visualizer_theme_styles(&cc.egui_ctx);
        let i18n_catalog =
            I18nCatalog::embedded().expect("embedded visualizer translations should be valid");
        let current_locale = Locale::default();
        cc.egui_ctx
            .install_i18n(i18n_catalog.for_locale(current_locale));

        Self {
            server_messages: channels.server_messages,
            client_messages: channels.client_messages,
            remote: channels.remote,
            i18n_catalog,
            current_locale,
            zoom_persistence_applied: false,
            zoom_percent_input: format_zoom_percent(DEFAULT_ZOOM_FACTOR),
            zoom_percent_input_dirty: false,
            zoom_percent_input_focused: false,
            state: VisualizerState::default(),
        }
    }

    fn install_locale(&mut self, ctx: &egui::Context, locale: Locale) {
        self.current_locale = locale;
        ctx.install_i18n(self.i18n_catalog.for_locale(locale));
    }

    fn drain_server_messages(&mut self) {
        loop {
            match self.server_messages.try_recv() {
                Ok(message) => {
                    dispatch_agent_action_event(&message, &self.client_messages);
                    self.state.apply_server_message(message);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.remote {
                        self.state.mark_disconnected();
                    }
                    break;
                }
            }
        }
    }
}

fn install_visualizer_fonts(ctx: &egui::Context) {
    let Some(font_data) = load_noto_sans_jp() else {
        eprintln!("warning: {NOTO_SANS_JP_FAMILY_NAME} was not loaded; using egui default fonts");
        return;
    };

    ctx.set_fonts(visualizer_font_definitions(font_data));
}

fn install_visualizer_theme_styles(ctx: &egui::Context) {
    for theme in [egui::Theme::Light, egui::Theme::Dark] {
        ctx.style_mut_of(theme, |style| {
            style.visuals.override_text_color = visualizer_override_text_color(theme);
            style.visuals.weak_text_color = visualizer_weak_text_color(theme);
            style.visuals.text_options.alpha_from_coverage =
                visualizer_text_alpha_from_coverage(theme);
            style.visuals.selection.bg_fill = visualizer_selection_fill(theme);
            style.visuals.selection.stroke =
                egui::Stroke::new(1.0, visualizer_selection_text_color(theme));
            style.visuals.hyperlink_color = visualizer_hyperlink_color(theme);
            style.visuals.warn_fg_color = visualizer_warning_text_color(theme);
            style.visuals.error_fg_color = visualizer_error_text_color(theme);
            style.visuals.widgets.noninteractive.fg_stroke.color =
                visualizer_normal_text_color(theme);
            style.visuals.widgets.inactive.fg_stroke.color =
                visualizer_interactive_text_color(theme);
            style.visuals.widgets.hovered.fg_stroke.color =
                visualizer_interactive_text_color(theme);
            style.visuals.widgets.active.fg_stroke.color = visualizer_interactive_text_color(theme);
            style.visuals.widgets.open.fg_stroke.color = visualizer_interactive_text_color(theme);
        });
    }
}

fn visualizer_normal_text_color(theme: egui::Theme) -> egui::Color32 {
    match theme {
        egui::Theme::Light => egui::Color32::from_gray(0),
        egui::Theme::Dark => egui::Color32::from_gray(222),
    }
}

fn visualizer_override_text_color(theme: egui::Theme) -> Option<egui::Color32> {
    match theme {
        egui::Theme::Light => Some(visualizer_normal_text_color(theme)),
        egui::Theme::Dark => None,
    }
}

fn visualizer_weak_text_color(theme: egui::Theme) -> Option<egui::Color32> {
    match theme {
        egui::Theme::Light => Some(egui::Color32::from_gray(64)),
        egui::Theme::Dark => None,
    }
}

fn visualizer_text_alpha_from_coverage(theme: egui::Theme) -> egui::epaint::AlphaFromCoverage {
    match theme {
        egui::Theme::Light => egui::epaint::AlphaFromCoverage::TwoCoverageMinusCoverageSq,
        egui::Theme::Dark => egui::epaint::AlphaFromCoverage::DARK_MODE_DEFAULT,
    }
}

fn visualizer_interactive_text_color(theme: egui::Theme) -> egui::Color32 {
    match theme {
        egui::Theme::Light => egui::Color32::from_gray(0),
        egui::Theme::Dark => egui::Color32::from_gray(236),
    }
}

fn visualizer_selection_fill(theme: egui::Theme) -> egui::Color32 {
    match theme {
        egui::Theme::Light => egui::Color32::from_rgb(206, 232, 255),
        egui::Theme::Dark => egui::Color32::from_rgb(0, 92, 128),
    }
}

fn visualizer_selection_text_color(theme: egui::Theme) -> egui::Color32 {
    match theme {
        egui::Theme::Light => egui::Color32::from_gray(0),
        egui::Theme::Dark => egui::Color32::from_rgb(192, 222, 255),
    }
}

fn visualizer_hyperlink_color(theme: egui::Theme) -> egui::Color32 {
    match theme {
        egui::Theme::Light => egui::Color32::from_rgb(0, 86, 160),
        egui::Theme::Dark => egui::Color32::from_rgb(90, 170, 255),
    }
}

fn visualizer_warning_text_color(theme: egui::Theme) -> egui::Color32 {
    match theme {
        egui::Theme::Light => egui::Color32::from_rgb(142, 72, 0),
        egui::Theme::Dark => egui::Color32::from_rgb(255, 143, 0),
    }
}

fn visualizer_error_text_color(theme: egui::Theme) -> egui::Color32 {
    match theme {
        egui::Theme::Light => egui::Color32::from_rgb(176, 0, 0),
        egui::Theme::Dark => egui::Color32::from_rgb(255, 0, 0),
    }
}

pub(crate) fn visualizer_selection_message_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        visuals.selection.bg_fill.linear_multiply(0.65)
    } else {
        egui::Color32::from_rgb(226, 242, 255)
    }
}

pub(crate) fn visualizer_selection_card_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        visuals.selection.bg_fill.linear_multiply(0.45)
    } else {
        egui::Color32::from_rgb(232, 245, 255)
    }
}

pub(crate) fn visualizer_selection_cell_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        visuals.selection.bg_fill.linear_multiply(0.55)
    } else {
        egui::Color32::from_rgb(219, 239, 255)
    }
}

pub(crate) fn visualizer_selection_row_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        visuals.selection.bg_fill.linear_multiply(0.22)
    } else {
        egui::Color32::from_rgb(240, 248, 255)
    }
}

pub(crate) fn visualizer_error_subtle_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        visuals.error_fg_color.linear_multiply(0.12)
    } else {
        egui::Color32::from_rgb(255, 242, 242)
    }
}

pub(crate) fn visualizer_error_banner_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        visuals.error_fg_color.linear_multiply(0.16)
    } else {
        egui::Color32::from_rgb(255, 237, 237)
    }
}

pub(crate) fn visualizer_error_row_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        visuals.error_fg_color.linear_multiply(0.18)
    } else {
        egui::Color32::from_rgb(255, 232, 232)
    }
}

fn visualizer_font_definitions(mut font_data: egui::FontData) -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    font_data.tweak = visualizer_font_tweak();
    fonts
        .font_data
        .insert(NOTO_SANS_JP_FONT_KEY.to_owned(), Arc::new(font_data));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, NOTO_SANS_JP_FONT_KEY.to_owned());
    }
    fonts
}

fn visualizer_font_properties() -> Properties {
    let mut properties = Properties::new();
    properties.weight(NOTO_SANS_JP_FONT_WEIGHT);
    properties
}

fn visualizer_font_tweak() -> egui::FontTweak {
    egui::FontTweak {
        coords: egui::epaint::text::VariationCoords::new([(b"wght", NOTO_SANS_JP_FONT_WEIGHT.0)]),
        ..Default::default()
    }
}

fn load_noto_sans_jp() -> Option<egui::FontData> {
    let handle = match SystemSource::new().select_best_match(
        &[FamilyName::Title(NOTO_SANS_JP_FAMILY_NAME.to_owned())],
        &visualizer_font_properties(),
    ) {
        Ok(handle) => handle,
        Err(error) => {
            eprintln!("warning: could not find {NOTO_SANS_JP_FAMILY_NAME}: {error}");
            return None;
        }
    };

    let (bytes, font_index) = match handle {
        Handle::Memory { bytes, font_index } => (bytes.to_vec(), font_index),
        Handle::Path { path, font_index } => match fs::read(&path) {
            Ok(bytes) => (bytes, font_index),
            Err(error) => {
                eprintln!(
                    "warning: could not read {NOTO_SANS_JP_FAMILY_NAME} from {}: {error}",
                    path.display()
                );
                return None;
            }
        },
    };

    let mut font_data = egui::FontData::from_owned(bytes);
    font_data.index = font_index;
    Some(font_data)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum VisualizerTheme {
    Light,
    Dark,
}

impl From<egui::Theme> for VisualizerTheme {
    fn from(theme: egui::Theme) -> Self {
        match theme {
            egui::Theme::Light => Self::Light,
            egui::Theme::Dark => Self::Dark,
        }
    }
}

impl From<VisualizerTheme> for egui::Theme {
    fn from(theme: VisualizerTheme) -> Self {
        match theme {
            VisualizerTheme::Light => Self::Light,
            VisualizerTheme::Dark => Self::Dark,
        }
    }
}

impl VisualizerTheme {
    fn label_key(self) -> &'static str {
        match self {
            Self::Light => "menu-theme-light",
            Self::Dark => "menu-theme-dark",
        }
    }
}

fn default_visualizer_theme(ctx: &egui::Context) -> VisualizerTheme {
    VisualizerTheme::from(ctx.theme())
}

fn render_theme_toggle(ui: &mut egui::Ui, theme: VisualizerTheme) -> Option<VisualizerTheme> {
    let mut next_theme = None;
    let hover = ui.ctx().tr("menu-theme-hover");
    egui::Frame::new()
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(14))
        .inner_margin(egui::Margin::symmetric(2, 2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.horizontal(|ui| {
                for candidate in [VisualizerTheme::Light, VisualizerTheme::Dark] {
                    let selected = candidate == theme;
                    let response =
                        ui.selectable_label(selected, ui.ctx().tr(candidate.label_key()));
                    if response.on_hover_text(hover.clone()).clicked() && !selected {
                        next_theme = Some(candidate);
                    }
                }
            });
        });
    next_theme
}

fn render_language_toggle(ui: &mut egui::Ui, locale: Locale) -> Option<Locale> {
    let mut next_locale = None;
    egui::Frame::new()
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .corner_radius(egui::CornerRadius::same(14))
        .inner_margin(egui::Margin::symmetric(2, 2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.horizontal(|ui| {
                for candidate in [Locale::JaJp, Locale::EnUs] {
                    let selected = candidate == locale;
                    if ui.selectable_label(selected, candidate.label()).clicked() && !selected {
                        next_locale = Some(candidate);
                    }
                }
            });
        });
    next_locale
}

struct ZoomControlOutcome {
    next_zoom_factor: Option<f32>,
    input_focused: bool,
}

fn render_zoom_control(
    ui: &mut egui::Ui,
    zoom_factor: f32,
    zoom_percent_input: &mut String,
    zoom_percent_input_dirty: &mut bool,
) -> ZoomControlOutcome {
    let mut next_zoom_factor = None;
    let zoom_hover = zoom_hover_text(ui.ctx());
    ui.label(ui.ctx().tr("menu-zoom"));

    let decrement = ui.small_button("-").on_hover_text(zoom_hover.clone());
    if let Some(percent_delta) = zoom_button_percent_delta(&decrement) {
        let zoom_factor = step_zoom_factor(zoom_factor, -percent_delta);
        *zoom_percent_input = format_zoom_percent(zoom_factor);
        *zoom_percent_input_dirty = false;
        next_zoom_factor = Some(zoom_factor);
    }

    let text_response = ui.add_sized(
        egui::vec2(48.0, ui.spacing().interact_size.y),
        egui::TextEdit::singleline(zoom_percent_input).desired_width(48.0),
    );
    if text_response.changed() {
        *zoom_percent_input_dirty = true;
    }
    let input_focused = text_response.has_focus();
    let commit_text_input = *zoom_percent_input_dirty
        && (text_response.lost_focus()
            || (input_focused && ui.input(|input| input.key_pressed(egui::Key::Enter))));
    text_response.on_hover_text(zoom_hover.clone());
    if commit_text_input {
        if let Some(zoom_factor) = parse_zoom_percent_input(zoom_percent_input) {
            *zoom_percent_input = format_zoom_percent(zoom_factor);
            next_zoom_factor = Some(zoom_factor);
        } else {
            *zoom_percent_input = format_zoom_percent(next_zoom_factor.unwrap_or(zoom_factor));
        }
        *zoom_percent_input_dirty = false;
    }

    ui.label("%").on_hover_text(zoom_hover.clone());

    let step_base = next_zoom_factor.unwrap_or(zoom_factor);
    let increment = ui.small_button("+").on_hover_text(zoom_hover);
    if let Some(percent_delta) = zoom_button_percent_delta(&increment) {
        let zoom_factor = step_zoom_factor(step_base, percent_delta);
        *zoom_percent_input = format_zoom_percent(zoom_factor);
        *zoom_percent_input_dirty = false;
        next_zoom_factor = Some(zoom_factor);
    }

    ZoomControlOutcome {
        next_zoom_factor,
        input_focused,
    }
}

fn zoom_button_percent_delta(response: &egui::Response) -> Option<f32> {
    if response.double_clicked() {
        Some(ZOOM_BUTTON_DOUBLE_CLICK_TOTAL_PERCENT - ZOOM_BUTTON_STEP_PERCENT)
    } else if response.clicked() {
        Some(ZOOM_BUTTON_STEP_PERCENT)
    } else {
        None
    }
}

fn zoom_hover_text(ctx: &egui::Context) -> String {
    ctx.tr_args(
        "menu-zoom-hover",
        &[
            ("min", zoom_percent_label(MIN_ZOOM_FACTOR).into()),
            ("max", zoom_percent_label(MAX_ZOOM_FACTOR).into()),
            (
                "step",
                zoom_button_percent_label(ZOOM_BUTTON_STEP_PERCENT).into(),
            ),
            (
                "jump",
                zoom_button_percent_label(ZOOM_BUTTON_DOUBLE_CLICK_TOTAL_PERCENT).into(),
            ),
        ],
    )
}

fn zoom_percent_label(zoom_factor: f32) -> i64 {
    zoom_factor_to_percent(zoom_factor).round() as i64
}

fn zoom_button_percent_label(percent: f32) -> i64 {
    percent.round() as i64
}

fn default_zoom_factor() -> f32 {
    DEFAULT_ZOOM_FACTOR
}

fn normalize_zoom_factor(zoom_factor: f32) -> f32 {
    if zoom_factor.is_finite() {
        zoom_factor.clamp(MIN_ZOOM_FACTOR, MAX_ZOOM_FACTOR)
    } else {
        DEFAULT_ZOOM_FACTOR
    }
}

fn zoom_factor_to_percent(zoom_factor: f32) -> f32 {
    normalize_zoom_factor(zoom_factor) * 100.0
}

fn format_zoom_percent(zoom_factor: f32) -> String {
    format!("{:.0}", zoom_factor_to_percent(zoom_factor).round())
}

fn parse_zoom_percent_input(input: &str) -> Option<f32> {
    let percent = input.trim().trim_end_matches('%').trim();
    if percent.is_empty() {
        return None;
    }
    percent.parse::<f32>().ok().map(zoom_percent_to_factor)
}

fn zoom_percent_to_factor(percent: f32) -> f32 {
    if percent.is_finite() {
        normalize_zoom_factor(percent / 100.0)
    } else {
        DEFAULT_ZOOM_FACTOR
    }
}

fn step_zoom_factor(zoom_factor: f32, percent_delta: f32) -> f32 {
    zoom_percent_to_factor(zoom_factor_to_percent(zoom_factor).round() + percent_delta)
}

fn zoom_factors_differ(left: f32, right: f32) -> bool {
    (left - right).abs() > ZOOM_SYNC_EPSILON
}

fn tr_tab_title(ctx: &egui::Context, key: &str, title: &str) -> String {
    ctx.tr_args(key, &[("title", I18nArg::from(title))])
}

impl eframe::App for VisualizerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let persisted_locale = ui.use_persisted_state(Locale::default, LOCALE_PERSISTENCE_KEY);
        let locale = *persisted_locale;
        self.install_locale(ui.ctx(), locale);
        let initial_theme = default_visualizer_theme(ui.ctx());
        let persisted_theme = ui.use_persisted_state(|| initial_theme, THEME_PERSISTENCE_KEY);
        let theme = *persisted_theme;
        ui.ctx().set_theme(egui::Theme::from(theme));
        let persisted_zoom =
            ui.use_persisted_state(default_zoom_factor, ZOOM_FACTOR_PERSISTENCE_KEY);
        let mut zoom_factor = normalize_zoom_factor(*persisted_zoom);
        if zoom_factors_differ(zoom_factor, *persisted_zoom) {
            persisted_zoom.set_next(zoom_factor);
        }
        if self.zoom_persistence_applied {
            let context_zoom = normalize_zoom_factor(ui.ctx().zoom_factor());
            if zoom_factors_differ(context_zoom, zoom_factor) {
                zoom_factor = context_zoom;
                persisted_zoom.set_next(zoom_factor);
            }
        } else {
            ui.ctx().set_zoom_factor(zoom_factor);
            self.zoom_persistence_applied = true;
        }
        if !self.zoom_percent_input_focused && !self.zoom_percent_input_dirty {
            self.zoom_percent_input = format_zoom_percent(zoom_factor);
        }
        self.drain_server_messages();
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));

        let mut next_locale = None;
        let mut next_theme = None;
        let mut next_zoom_factor = None;
        egui::Panel::top("nuillu-visualizer-tabs").show_inside(ui, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                next_theme = render_theme_toggle(ui, theme);
                next_locale = render_language_toggle(ui, locale);
                ui.allocate_ui_with_layout(
                    egui::vec2(176.0, ui.spacing().interact_size.y),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        let outcome = render_zoom_control(
                            ui,
                            zoom_factor,
                            &mut self.zoom_percent_input,
                            &mut self.zoom_percent_input_dirty,
                        );
                        next_zoom_factor = outcome.next_zoom_factor;
                        self.zoom_percent_input_focused = outcome.input_focused;
                    },
                );
                ui.horizontal_wrapped(|ui| {
                    for tab in self.state.tabs.values() {
                        let selected = self.state.selected.as_ref() == Some(&tab.id);
                        let label = format!("{} {}", tab_status_icon(tab.status), tab.title);
                        if ui.selectable_label(selected, label).clicked() {
                            self.state.selected = Some(tab.id.clone());
                        }
                    }
                    let view_tab_id = self
                        .state
                        .selected
                        .clone()
                        .or_else(|| self.state.tabs.keys().next().cloned());
                    if let Some(tab_id) = view_tab_id {
                        if let Some(tab) = self.state.tabs.get_mut(&tab_id) {
                            tab.view_menu(ui);
                        }
                    } else {
                        ui.menu_button(ui.ctx().tr("menu-view"), |ui| {
                            ui.label(ui.ctx().tr("menu-no-runtime-windows"));
                        });
                    }
                    for action in self.state.visible_actions() {
                        if ui.button(&action.label).clicked() {
                            let _ =
                                self.client_messages
                                    .send(VisualizerClientMessage::InvokeAction {
                                        action_id: action.id,
                                    });
                        }
                    }
                    if self.remote && self.state.disconnected {
                        ui.colored_label(
                            ui.visuals().error_fg_color,
                            ui.ctx().tr("status-eval-disconnected"),
                        );
                    }
                });
            });
        });
        if let Some(locale) = next_locale {
            persisted_locale.set_next(locale);
            self.install_locale(ui.ctx(), locale);
            ui.ctx().request_repaint();
        }
        if let Some(theme) = next_theme {
            persisted_theme.set_next(theme);
            ui.ctx().set_theme(egui::Theme::from(theme));
            ui.ctx().request_repaint();
        }
        if let Some(zoom_factor) = next_zoom_factor {
            persisted_zoom.set_next(zoom_factor);
            ui.ctx().set_zoom_factor(zoom_factor);
            ui.ctx().request_repaint();
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let selected = self
                .state
                .selected
                .clone()
                .or_else(|| self.state.tabs.keys().next().cloned());
            if let Some(tab_id) = selected {
                self.state.selected = Some(tab_id.clone());
                if let Some(tab) = self.state.tabs.get_mut(&tab_id) {
                    ui.push_id(tab_id.as_str(), |ui| {
                        tab.ui(ui, &self.client_messages);
                    });
                }
            } else {
                ui.centered_and_justified(|ui| {
                    if self.remote && self.state.disconnected {
                        ui.label(ui.ctx().tr("status-eval-process-disconnected"));
                    } else {
                        ui.label(ui.ctx().tr("status-no-runtime-tabs"));
                    }
                });
            }
        });
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_millis(1500)
    }

    fn on_exit(&mut self) {
        let _ = self.client_messages.send(VisualizerClientMessage::Command {
            command: VisualizerCommand::Shutdown,
        });
    }
}

fn dispatch_agent_action_event(
    message: &VisualizerServerMessage,
    commands: &Sender<VisualizerClientMessage>,
) {
    let VisualizerServerMessage::Event {
        event: VisualizerEvent::AgentActionInvocationRequested { tab_id, request },
    } = message
    else {
        return;
    };

    if request.action_id == "poet" {
        handle_poet_action(tab_id, request, commands);
    } else {
        handle_generic_external_action(tab_id, request, commands);
    }
}

fn handle_poet_action(
    tab_id: &VisualizerTabId,
    request: &AgentActionInvocationRequest,
    commands: &Sender<VisualizerClientMessage>,
) {
    let poem = request
        .arguments
        .get("poem")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim();
    if poem.is_empty() {
        send_agent_action_completion(
            tab_id,
            &request.invocation_id,
            false,
            "poet action requires a non-empty poem",
            commands,
        );
        return;
    }
    send_agent_action_completion(
        tab_id,
        &request.invocation_id,
        true,
        "poem recorded by visualizer",
        commands,
    );
    send_action_sensory_feedback(
        tab_id,
        format!("The poet action recorded a poem:\n{poem}"),
        commands,
    );
}

fn handle_generic_external_action(
    tab_id: &VisualizerTabId,
    request: &AgentActionInvocationRequest,
    commands: &Sender<VisualizerClientMessage>,
) {
    let arguments =
        serde_json::to_string(&request.arguments).unwrap_or_else(|_| "<invalid-json>".to_owned());
    send_agent_action_completion(
        tab_id,
        &request.invocation_id,
        true,
        format!(
            "external action {} accepted by visualizer",
            request.action_id
        ),
        commands,
    );
    send_action_sensory_feedback(
        tab_id,
        format!(
            "External action {} was invoked with arguments: {}",
            request.action_id, arguments
        ),
        commands,
    );
}

fn send_agent_action_completion(
    tab_id: &VisualizerTabId,
    invocation_id: &str,
    accepted: bool,
    message: impl Into<String>,
    commands: &Sender<VisualizerClientMessage>,
) {
    let _ = commands.send(VisualizerClientMessage::Command {
        command: VisualizerCommand::CompleteAgentActionInvocation {
            tab_id: tab_id.clone(),
            completion: AgentActionInvocationCompletion {
                invocation_id: invocation_id.to_owned(),
                accepted,
                message: message.into(),
            },
        },
    });
}

fn send_action_sensory_feedback(
    tab_id: &VisualizerTabId,
    content: String,
    commands: &Sender<VisualizerClientMessage>,
) {
    let _ = commands.send(VisualizerClientMessage::Command {
        command: VisualizerCommand::SendOneShotSensoryInput {
            tab_id: tab_id.clone(),
            input: OneShotSensoryInput {
                modality: "action".to_owned(),
                direction: None,
                content,
            },
        },
    });
}

#[derive(Default)]
pub struct VisualizerState {
    tabs: BTreeMap<VisualizerTabId, RuntimeTab>,
    selected: Option<VisualizerTabId>,
    actions: BTreeMap<String, VisualizerAction>,
    disconnected: bool,
}

impl VisualizerState {
    pub fn apply_server_message(&mut self, message: VisualizerServerMessage) {
        match message {
            VisualizerServerMessage::Hello { .. } => {
                self.disconnected = false;
            }
            VisualizerServerMessage::Event { event } => self.apply(event),
            VisualizerServerMessage::OfferAction { action } => {
                self.actions.insert(action.id.clone(), action);
            }
            VisualizerServerMessage::RevokeAction { action_id } => {
                self.actions.remove(&action_id);
            }
        }
    }

    pub fn mark_disconnected(&mut self) {
        if self.disconnected {
            return;
        }
        self.disconnected = true;
        self.actions.clear();
        for tab in self.tabs.values_mut() {
            tab.push_log("eval process disconnected".to_string());
        }
    }

    pub fn apply(&mut self, event: VisualizerEvent) {
        match event {
            VisualizerEvent::OpenTab { tab_id, title } => {
                let tab = self
                    .tabs
                    .entry(tab_id.clone())
                    .or_insert_with(|| RuntimeTab::new(tab_id.clone(), title.clone()));
                tab.title = title;
                tab.status = TabStatus::Running;
                self.selected.get_or_insert(tab_id);
            }
            VisualizerEvent::SetTabStatus { tab_id, status } => {
                self.tab_mut(tab_id).status = status;
            }
            VisualizerEvent::Log { tab_id, message } => {
                self.tab_mut(tab_id).push_log(message);
            }
            VisualizerEvent::SensoryInput { tab_id, input } => {
                self.tab_mut(tab_id).scene.push_sensory_input(input);
            }
            VisualizerEvent::UtteranceDelta { tab_id, utterance } => {
                self.tab_mut(tab_id).scene.push_utterance_delta(utterance);
            }
            VisualizerEvent::UtteranceCompleted { tab_id, utterance } => {
                self.tab_mut(tab_id)
                    .scene
                    .push_utterance_completed(utterance);
            }
            VisualizerEvent::OneShotSensoryInputRows { tab_id, rows } => {
                self.tab_mut(tab_id)
                    .scene
                    .apply_one_shot_sensory_input_rows(rows);
            }
            VisualizerEvent::OneShotSensoryInputAppended { tab_id, row } => {
                self.tab_mut(tab_id)
                    .scene
                    .append_one_shot_sensory_input_row(row);
            }
            VisualizerEvent::AmbientSensorySnapshotRows { tab_id, rows } => {
                self.tab_mut(tab_id)
                    .scene
                    .apply_ambient_sensory_snapshot_rows(rows);
            }
            VisualizerEvent::AmbientSensorySnapshotAppended { tab_id, row } => {
                self.tab_mut(tab_id)
                    .scene
                    .append_ambient_sensory_snapshot_row(row);
            }
            VisualizerEvent::UtteranceEventRows { tab_id, rows } => {
                self.tab_mut(tab_id).scene.apply_utterance_event_rows(rows);
            }
            VisualizerEvent::UtteranceEventAppended { tab_id, row } => {
                self.tab_mut(tab_id).scene.append_utterance_event_row(row);
            }
            VisualizerEvent::ExternalActionEventRows { tab_id, rows } => {
                self.tab_mut(tab_id)
                    .scene
                    .apply_external_action_event_rows(rows);
            }
            VisualizerEvent::ExternalActionEventAppended { tab_id, row } => {
                self.tab_mut(tab_id)
                    .scene
                    .append_external_action_event_row(row);
            }
            VisualizerEvent::ExternalActionEventUpdated { tab_id, row } => {
                self.tab_mut(tab_id)
                    .scene
                    .update_external_action_event_row(row);
            }
            VisualizerEvent::RuntimeEvent { tab_id, event } => {
                let tab = self.tab_mut(tab_id);
                let now_secs = tab.resource_monitor_elapsed_secs();
                tab.record_runtime_event_for_monitor(&event);
                modules::apply_runtime_event_at(&mut tab.modules, &event, now_secs);
                tab.runtime_events.push_back(event);
                if tab.runtime_events.len() > 256 {
                    tab.runtime_events.pop_front();
                }
            }
            VisualizerEvent::Error { tab_id, error } => {
                let selected = {
                    let tab = self.tab_mut(tab_id);
                    let errors_id = tab.errors_window_id();
                    tab.push_error(error);
                    tab.window_requests.insert(errors_id, true);
                    tab.id.clone()
                };
                self.selected = Some(selected);
            }
            VisualizerEvent::LlmObserved { tab_id, event } => {
                modules::apply_llm_observation(&mut self.tab_mut(tab_id).modules, event);
            }
            VisualizerEvent::LlmTranscriptSnapshot { tab_id, turns } => {
                modules::apply_llm_transcript_snapshot(&mut self.tab_mut(tab_id).modules, turns);
            }
            VisualizerEvent::BlackboardSnapshot { tab_id, snapshot } => {
                let tab = self.tab_mut(tab_id);
                tab.record_snapshot_for_monitor(&snapshot);
                modules::apply_blackboard_snapshot(&mut tab.modules, &snapshot);
                tab.blackboard = snapshot;
            }
            VisualizerEvent::MemoryRecordsLoaded {
                tab_id,
                scope,
                offset,
                records,
                has_more,
            } => {
                self.tab_mut(tab_id)
                    .memories
                    .apply_records_loaded(scope, offset, records, has_more);
            }
            VisualizerEvent::LinkedMemoryRecordsLoaded {
                tab_id,
                memory_index,
                offset,
                records,
                has_more,
            } => {
                self.tab_mut(tab_id).memories.apply_linked_records_loaded(
                    memory_index,
                    offset,
                    records,
                    has_more,
                );
            }
            VisualizerEvent::MemoryDeleted {
                tab_id,
                memory_index,
            } => {
                self.tab_mut(tab_id)
                    .memories
                    .apply_memory_deleted(&memory_index);
            }
            VisualizerEvent::AmbientSensoryRows { .. } => {}
            VisualizerEvent::SceneState { tab_id, state } => {
                self.tab_mut(tab_id).scene.set_scene_state(state);
            }
            VisualizerEvent::AgentActionAffordances {
                tab_id,
                affordances,
            } => {
                let tab = self.tab_mut(tab_id);
                tab.action_affordances = affordances;
                tab.push_log(format!(
                    "agent action affordances updated: {} available",
                    tab.action_affordances.len()
                ));
            }
            VisualizerEvent::AgentActionInvocationRequested { tab_id, request } => {
                self.tab_mut(tab_id).push_log(format!(
                    "agent invoked external action {}",
                    request.action_id
                ));
            }
        }
    }

    pub fn tabs(&self) -> &BTreeMap<VisualizerTabId, RuntimeTab> {
        &self.tabs
    }

    pub fn visible_actions(&self) -> Vec<VisualizerAction> {
        self.actions
            .values()
            .filter(|action| match &action.scope {
                VisualizerActionScope::Global => true,
                VisualizerActionScope::Tab { tab_id } => self.selected.as_ref() == Some(tab_id),
            })
            .cloned()
            .collect()
    }

    fn tab_mut(&mut self, tab_id: VisualizerTabId) -> &mut RuntimeTab {
        self.tabs
            .entry(tab_id.clone())
            .or_insert_with(|| RuntimeTab::new(tab_id, "runtime".to_string()))
    }
}

pub struct RuntimeTab {
    id: VisualizerTabId,
    title: String,
    status: TabStatus,
    view_mode: RuntimeTabViewMode,
    simplified_cognition_pane_tab: SimplifiedCognitionPaneTab,
    active_simplified_module_owner: Option<String>,
    scene: chat::SceneUiState,
    blackboard: BlackboardSnapshot,
    memories: memories::MemoriesState,
    modules: modules::ModulesState,
    resource_monitor: resource_monitor::ResourceMonitorState,
    runtime_events: VecDeque<RuntimeEvent>,
    errors: VecDeque<VisualizerErrorView>,
    session_error_count: u32,
    logs: VecDeque<String>,
    window_open: BTreeMap<String, bool>,
    window_requests: BTreeMap<String, bool>,
    memos_module_filter: module_filter::ModuleFilterState,
    llm_turns_module_filter: module_filter::ModuleFilterState,
    resource_monitor_module_filter: module_filter::ModuleFilterState,
    resource_monitor_started_at: Instant,
    action_affordances: Vec<ActionAffordance>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum RuntimeTabViewMode {
    #[default]
    Simplified,
    Windowed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum SimplifiedCognitionPaneTab {
    #[default]
    CognitionLog,
    Memo,
}

const SIMPLIFIED_PANE_GAP: f32 = 8.0;
const SIMPLIFIED_LOWER_PANES_MIN_HEIGHT: f32 = 180.0;

#[derive(Debug, Clone)]
struct ViewWindowSpec {
    id: String,
    title: String,
    default_open: bool,
    kind: ViewWindowKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewWindowKind {
    Normal,
    Module,
}

impl RuntimeTab {
    fn new(id: VisualizerTabId, title: String) -> Self {
        Self {
            id,
            title,
            status: TabStatus::Running,
            view_mode: RuntimeTabViewMode::default(),
            simplified_cognition_pane_tab: SimplifiedCognitionPaneTab::default(),
            active_simplified_module_owner: None,
            scene: chat::SceneUiState::default(),
            blackboard: BlackboardSnapshot::default(),
            memories: memories::MemoriesState::default(),
            modules: modules::ModulesState::default(),
            resource_monitor: resource_monitor::ResourceMonitorState::default(),
            runtime_events: VecDeque::new(),
            errors: VecDeque::new(),
            session_error_count: 0,
            logs: VecDeque::new(),
            window_open: BTreeMap::new(),
            window_requests: BTreeMap::new(),
            memos_module_filter: module_filter::ModuleFilterState::default(),
            llm_turns_module_filter: module_filter::ModuleFilterState::default(),
            resource_monitor_module_filter: module_filter::ModuleFilterState::default(),
            resource_monitor_started_at: Instant::now(),
            action_affordances: Vec::new(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, commands: &Sender<VisualizerClientMessage>) {
        match self.view_mode {
            RuntimeTabViewMode::Simplified => self.simplified_ui(ui, commands),
            RuntimeTabViewMode::Windowed => self.windows_ui(ui, commands),
        }
    }

    fn view_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button(ui.ctx().tr("menu-view"), |ui| {
            let simplified = self.view_mode == RuntimeTabViewMode::Simplified;
            let label = if simplified {
                ui.ctx().tr("menu-simplified-view-enabled")
            } else {
                ui.ctx().tr("menu-simplified-view-disabled")
            };
            if ui.button(label).clicked() {
                self.set_simplified_view(!simplified);
                ui.close();
            }
            if self.view_mode == RuntimeTabViewMode::Simplified {
                return;
            }

            ui.separator();
            let specs = self.window_specs(ui.ctx());
            if specs.iter().any(|spec| spec.kind == ViewWindowKind::Module) {
                if ui
                    .button(ui.ctx().tr("menu-close-all-module-windows"))
                    .clicked()
                {
                    self.close_all_module_windows();
                    ui.close();
                }
                ui.separator();
            }
            for spec in specs {
                let mut open = self
                    .window_open
                    .get(&spec.id)
                    .copied()
                    .unwrap_or(spec.default_open);
                ui.horizontal(|ui| {
                    if ui.add(egui::Checkbox::without_text(&mut open)).changed() {
                        self.window_requests.insert(spec.id.clone(), open);
                    }

                    if ui
                        .add(egui::Label::new(&spec.title).sense(egui::Sense::click()))
                        .clicked()
                    {
                        self.window_requests.insert(spec.id.clone(), true);
                        ui.close();
                    }
                });
            }
        });
    }

    fn set_simplified_view(&mut self, simplified: bool) {
        self.view_mode = if simplified {
            RuntimeTabViewMode::Simplified
        } else {
            RuntimeTabViewMode::Windowed
        };
        if self.view_mode == RuntimeTabViewMode::Windowed {
            self.active_simplified_module_owner = None;
        }
    }

    fn window_specs(&self, ctx: &egui::Context) -> Vec<ViewWindowSpec> {
        let base = self.id.as_str();
        let mut specs = vec![
            ViewWindowSpec {
                id: format!("{base}:chat"),
                title: tr_tab_title(ctx, "window-scene-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:blackboard"),
                title: tr_tab_title(ctx, "window-blackboard-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:memories"),
                title: tr_tab_title(ctx, "window-memory-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:memos"),
                title: tr_tab_title(ctx, "window-memo-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:llm-turns"),
                title: tr_tab_title(ctx, "window-llm-turns-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:cognition"),
                title: tr_tab_title(ctx, "window-cognition-log-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:errors"),
                title: tr_tab_title(ctx, "window-errors-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:logs"),
                title: tr_tab_title(ctx, "window-logs-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:resource-monitor"),
                title: tr_tab_title(ctx, "window-resource-monitor-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
            ViewWindowSpec {
                id: format!("{base}:modules"),
                title: tr_tab_title(ctx, "window-modules-title", &self.title),
                default_open: true,
                kind: ViewWindowKind::Normal,
            },
        ];
        for module in self.modules.iter() {
            specs.push(ViewWindowSpec {
                id: format!("{base}:module:{}", module.owner),
                title: modules::window_title(ctx, module),
                default_open: false,
                kind: ViewWindowKind::Module,
            });
        }
        specs
    }

    fn module_window_ids(&self) -> Vec<String> {
        let base = self.id.as_str();
        self.modules
            .iter()
            .map(|module| format!("{base}:module:{}", module.owner))
            .collect()
    }

    fn close_all_module_windows(&mut self) {
        for id in self.module_window_ids() {
            self.window_requests.insert(id, false);
        }
    }

    fn simplified_ui(&mut self, ui: &mut egui::Ui, commands: &Sender<VisualizerClientMessage>) {
        let available = ui.available_size();
        if available.x <= 1.0 || available.y <= 1.0 {
            return;
        }

        let resource_monitor_now_secs = self.resource_monitor_elapsed_secs();
        let right_width = (available.x * 0.32)
            .clamp(360.0, 560.0)
            .min((available.x - 320.0).max(260.0));
        let center_width = (available.x - right_width - SIMPLIFIED_PANE_GAP).max(260.0);
        let mut requested_module = None;

        ui.horizontal(|ui| {
            ui.set_min_height(available.y);
            ui.allocate_ui_with_layout(
                egui::vec2(center_width, available.y),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    let modules_max_height = simplified_modules_max_height(ui.available_height());
                    simplified_auto_section(
                        ui,
                        None,
                        egui::vec2(ui.available_width(), modules_max_height),
                        |ui| {
                            requested_module = self.render_modules_overview_contents(
                                ui,
                                commands,
                                resource_monitor_now_secs,
                            );
                        },
                    );
                    ui.add_space(SIMPLIFIED_PANE_GAP);

                    let lower_height = ui.available_height().max(1.0);
                    let lower_width = ui.available_width();
                    let column_width = ((lower_width - SIMPLIFIED_PANE_GAP) / 2.0).max(1.0);
                    let memory_title = ui.ctx().tr("section-memory");
                    ui.horizontal(|ui| {
                        simplified_section(
                            ui,
                            None,
                            egui::vec2(column_width, lower_height),
                            |ui| self.render_simplified_cognition_pane_contents(ui),
                        );
                        ui.add_space(SIMPLIFIED_PANE_GAP);
                        simplified_section(
                            ui,
                            Some(memory_title.as_str()),
                            egui::vec2(ui.available_width().max(1.0), lower_height),
                            |ui| self.render_memory_contents(ui, commands),
                        );
                    });
                },
            );

            ui.add_space(SIMPLIFIED_PANE_GAP);
            ui.allocate_ui_with_layout(
                egui::vec2(right_width.min(ui.available_width()).max(1.0), available.y),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    let scene_title = ui.ctx().tr("section-scene");
                    simplified_section(
                        ui,
                        Some(scene_title.as_str()),
                        egui::vec2(ui.available_width(), available.y),
                        |ui| self.render_scene_contents(ui, commands),
                    );
                },
            );
        });

        let module_opened_this_frame = requested_module.is_some();
        if let Some(owner) = requested_module {
            self.open_simplified_module(owner);
        }
        self.render_simplified_interoception_window(ui, resource_monitor_now_secs);
        self.render_simplified_module_popup(ui, commands, module_opened_this_frame);
    }

    fn render_scene_contents(
        &mut self,
        ui: &mut egui::Ui,
        commands: &Sender<VisualizerClientMessage>,
    ) {
        chat::ui(ui, &self.id, &mut self.scene, commands);
    }

    fn render_memory_contents(
        &mut self,
        ui: &mut egui::Ui,
        commands: &Sender<VisualizerClientMessage>,
    ) {
        memories::ui(ui, &self.id, &mut self.memories, commands);
    }

    fn render_simplified_cognition_pane_contents(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .selectable_label(
                    self.simplified_cognition_pane_tab == SimplifiedCognitionPaneTab::CognitionLog,
                    ui.ctx().tr("section-cognition-log"),
                )
                .clicked()
            {
                self.simplified_cognition_pane_tab = SimplifiedCognitionPaneTab::CognitionLog;
            }
            if ui
                .selectable_label(
                    self.simplified_cognition_pane_tab == SimplifiedCognitionPaneTab::Memo,
                    ui.ctx().tr("section-memo"),
                )
                .clicked()
            {
                self.simplified_cognition_pane_tab = SimplifiedCognitionPaneTab::Memo;
            }
        });
        ui.separator();

        match self.simplified_cognition_pane_tab {
            SimplifiedCognitionPaneTab::CognitionLog => self.render_cognition_contents(ui),
            SimplifiedCognitionPaneTab::Memo => {
                let memo_filter_modules = self.memo_filter_modules();
                memos::ui(
                    ui,
                    &self.blackboard.memos,
                    &mut self.memos_module_filter,
                    &memo_filter_modules,
                );
            }
        }
    }

    fn render_cognition_contents(&self, ui: &mut egui::Ui) {
        cognition::ui(ui, &self.blackboard.cognition_logs);
    }

    fn render_modules_overview_contents(
        &mut self,
        ui: &mut egui::Ui,
        commands: &Sender<VisualizerClientMessage>,
        now_secs: f64,
    ) -> Option<String> {
        let actions =
            modules::render_modules_overview(ui, &self.blackboard, &self.modules, now_secs);
        self.handle_module_overview_actions(actions, commands)
    }

    fn handle_module_overview_actions(
        &mut self,
        actions: Vec<modules::ModuleOverviewAction>,
        commands: &Sender<VisualizerClientMessage>,
    ) -> Option<String> {
        let mut requested_module = None;
        for action in actions {
            match action {
                modules::ModuleOverviewAction::OpenModule { owner } => {
                    requested_module = Some(owner);
                }
                modules::ModuleOverviewAction::SetDisabled { module, disabled } => {
                    let _ = commands.send(VisualizerClientMessage::Command {
                        command: VisualizerCommand::SetModuleDisabled {
                            tab_id: self.id.clone(),
                            module,
                            disabled,
                        },
                    });
                }
                modules::ModuleOverviewAction::SetModuleSettings { settings } => {
                    let _ = commands.send(VisualizerClientMessage::Command {
                        command: VisualizerCommand::SetModuleSettings {
                            tab_id: self.id.clone(),
                            settings,
                        },
                    });
                }
            }
        }
        requested_module
    }

    fn render_module_contents(
        &mut self,
        ui: &mut egui::Ui,
        owner: &str,
        commands: &Sender<VisualizerClientMessage>,
    ) {
        let Some(module) = self.modules.get(owner) else {
            return;
        };
        let module_window_actions = modules::render_module(ui, module, &self.blackboard.memos);
        for action in module_window_actions {
            match action {
                modules::ModuleWindowAction::ResetSessionHistory { owner } => {
                    let _ = commands.send(VisualizerClientMessage::Command {
                        command: VisualizerCommand::ResetModuleSessionHistory {
                            tab_id: self.id.clone(),
                            owner,
                        },
                    });
                }
            }
        }
    }

    fn open_simplified_module(&mut self, owner: String) {
        self.active_simplified_module_owner = Some(owner);
    }

    fn render_simplified_interoception_window(&self, ui: &mut egui::Ui, now_secs: f64) {
        egui::Window::new(tr_tab_title(
            ui.ctx(),
            "window-interoception-title",
            &self.title,
        ))
        .id(egui::Id::new((
            self.id.as_str(),
            "simplified-interoception",
        )))
        .order(egui::Order::Foreground)
        .collapsible(false)
        .default_pos(egui::pos2(24.0, 96.0))
        .default_size(egui::vec2(520.0, 260.0))
        .show(ui.ctx(), |ui| {
            resource_monitor::render_interoception_plot(ui, &self.resource_monitor, now_secs);
        });
    }

    fn render_simplified_module_popup(
        &mut self,
        ui: &mut egui::Ui,
        commands: &Sender<VisualizerClientMessage>,
        opened_this_frame: bool,
    ) {
        let Some(owner) = self.active_simplified_module_owner.clone() else {
            return;
        };
        let Some(module) = self.modules.get(&owner) else {
            self.active_simplified_module_owner = None;
            return;
        };
        let title = modules::window_title(ui.ctx(), module);
        let mut open = true;
        let response = egui::Window::new(title)
            .id(egui::Id::new((
                self.id.as_str(),
                "simplified-module-popup",
                owner.as_str(),
            )))
            .order(egui::Order::Foreground)
            .collapsible(false)
            .open(&mut open)
            .default_pos(egui::pos2(720.0, 128.0))
            .default_size(egui::vec2(520.0, 420.0))
            .show(ui.ctx(), |ui| {
                self.render_module_contents(ui, &owner, commands);
            });

        let Some(response) = response else {
            self.active_simplified_module_owner = None;
            return;
        };
        if !open {
            self.active_simplified_module_owner = None;
            return;
        }
        let (primary_clicked, interact_pos) = ui.ctx().input(|input| {
            (
                input.pointer.primary_clicked(),
                input.pointer.interact_pos(),
            )
        });
        self.close_simplified_module_popup_for_interaction(
            response.response.rect,
            opened_this_frame,
            primary_clicked,
            interact_pos,
        );
    }

    fn close_simplified_module_popup_for_interaction(
        &mut self,
        popup_rect: egui::Rect,
        opened_this_frame: bool,
        primary_clicked: bool,
        interact_pos: Option<egui::Pos2>,
    ) {
        if simplified_module_popup_interaction_closes(
            popup_rect,
            opened_this_frame,
            primary_clicked,
            interact_pos,
        ) {
            self.active_simplified_module_owner = None;
        }
    }

    fn windows_ui(&mut self, ui: &mut egui::Ui, commands: &Sender<VisualizerClientMessage>) {
        let base = self.id.as_str().to_string();
        let mut window_requests = std::mem::take(&mut self.window_requests);

        let chat_id = format!("{base}:chat");
        let chat_title = tr_tab_title(ui.ctx(), "window-scene-title", &self.title);
        let open = window::PersistedWindow::new(&chat_id, &chat_title)
            .open_override(window_requests.remove(&chat_id))
            .default_pos(24.0, 88.0)
            .default_size(760.0, 620.0)
            .show(ui, |ui| self.render_scene_contents(ui, commands));
        self.record_window_open(chat_id, open);

        let blackboard_id = format!("{base}:blackboard");
        let blackboard_title = tr_tab_title(ui.ctx(), "window-blackboard-title", &self.title);
        let open = window::PersistedWindow::new(&blackboard_id, &blackboard_title)
            .open_override(window_requests.remove(&blackboard_id))
            .default_pos(568.0, 88.0)
            .default_size(640.0, 520.0)
            .show(ui, |ui| blackboard::ui(ui, &self.blackboard));
        self.record_window_open(blackboard_id, open);

        let memories_id = format!("{base}:memories");
        let memories_title = tr_tab_title(ui.ctx(), "window-memory-title", &self.title);
        let open = window::PersistedWindow::new(&memories_id, &memories_title)
            .open_override(window_requests.remove(&memories_id))
            .default_pos(96.0, 636.0)
            .default_size(720.0, 360.0)
            .show(ui, |ui| self.render_memory_contents(ui, commands));
        self.record_window_open(memories_id, open);

        let memos_id = format!("{base}:memos");
        let memos_title = tr_tab_title(ui.ctx(), "window-memo-title", &self.title);
        let memo_filter_modules = self.memo_filter_modules();
        let open = window::PersistedWindow::new(&memos_id, &memos_title)
            .open_override(window_requests.remove(&memos_id))
            .default_pos(840.0, 636.0)
            .default_size(520.0, 360.0)
            .show(ui, |ui| {
                memos::ui(
                    ui,
                    &self.blackboard.memos,
                    &mut self.memos_module_filter,
                    &memo_filter_modules,
                )
            });
        self.record_window_open(memos_id, open);

        let llm_turns_id = format!("{base}:llm-turns");
        let llm_turns_title = tr_tab_title(ui.ctx(), "window-llm-turns-title", &self.title);
        let llm_turns_filter_modules = self.modules.module_names();
        let open = window::PersistedWindow::new(&llm_turns_id, &llm_turns_title)
            .open_override(window_requests.remove(&llm_turns_id))
            .default_pos(1384.0, 88.0)
            .default_size(640.0, 520.0)
            .show(ui, |ui| {
                modules::render_llm_turns(
                    ui,
                    &self.modules,
                    &mut self.llm_turns_module_filter,
                    &llm_turns_filter_modules,
                )
            });
        self.record_window_open(llm_turns_id, open);

        let cognition_id = format!("{base}:cognition");
        let cognition_title = tr_tab_title(ui.ctx(), "window-cognition-log-title", &self.title);
        let open = window::PersistedWindow::new(&cognition_id, &cognition_title)
            .open_override(window_requests.remove(&cognition_id))
            .default_pos(1384.0, 636.0)
            .default_size(560.0, 360.0)
            .show(ui, |ui| self.render_cognition_contents(ui));
        self.record_window_open(cognition_id, open);

        let errors_id = self.errors_window_id();
        let errors_title = tr_tab_title(ui.ctx(), "window-errors-title", &self.title);
        let open = window::PersistedWindow::new(&errors_id, &errors_title)
            .open_override(window_requests.remove(&errors_id))
            .default_pos(24.0, 1020.0)
            .default_size(640.0, 360.0)
            .show(ui, |ui| {
                errors::ui(
                    ui,
                    &mut self.errors,
                    self.session_error_count,
                    self.modules.session_live_llm_turn_count(),
                )
            });
        self.record_window_open(errors_id, open);

        let logs_id = format!("{base}:logs");
        let logs_title = tr_tab_title(ui.ctx(), "window-logs-title", &self.title);
        let open = window::PersistedWindow::new(&logs_id, &logs_title)
            .open_override(window_requests.remove(&logs_id))
            .default_pos(688.0, 1020.0)
            .default_size(520.0, 360.0)
            .show(ui, |ui| self.logs_ui(ui));
        self.record_window_open(logs_id, open);

        let resource_monitor_id = format!("{base}:resource-monitor");
        let resource_monitor_title =
            tr_tab_title(ui.ctx(), "window-resource-monitor-title", &self.title);
        let resource_monitor_modules = self.resource_monitor_modules();
        let resource_monitor_now_secs = self.resource_monitor_elapsed_secs();
        let open = window::PersistedWindow::new(&resource_monitor_id, &resource_monitor_title)
            .open_override(window_requests.remove(&resource_monitor_id))
            .default_pos(1232.0, 1020.0)
            .default_size(760.0, 560.0)
            .show(ui, |ui| {
                resource_monitor::ui(
                    ui,
                    &mut self.resource_monitor,
                    &mut self.resource_monitor_module_filter,
                    &resource_monitor_modules,
                    resource_monitor_now_secs,
                )
            });
        self.record_window_open(resource_monitor_id, open);

        let modules_id = format!("{base}:modules");
        let modules_title = tr_tab_title(ui.ctx(), "window-modules-title", &self.title);
        let mut requested_module = None;
        let open = window::PersistedWindow::new(&modules_id, &modules_title)
            .open_override(window_requests.remove(&modules_id))
            .default_pos(568.0, 1020.0)
            .default_size(640.0, 360.0)
            .show(ui, |ui| {
                requested_module =
                    self.render_modules_overview_contents(ui, commands, resource_monitor_now_secs);
            });
        self.record_window_open(modules_id, open);

        let ctx = ui.ctx().clone();
        let module_windows = self
            .modules
            .iter()
            .enumerate()
            .map(|(index, module)| {
                (
                    index,
                    module.owner.clone(),
                    modules::window_title(&ctx, module),
                )
            })
            .collect::<Vec<_>>();
        for (index, owner, module_title) in module_windows {
            let module_id = format!("{base}:module:{owner}");
            let x = 1232.0 + (index % 2) as f32 * 440.0;
            let y = 88.0 + (index / 2) as f32 * 380.0;
            let requested = if requested_module.as_deref() == Some(owner.as_str()) {
                Some(true)
            } else {
                window_requests.remove(&module_id)
            };
            let open = {
                let window_open = window::PersistedWindow::new(&module_id, &module_title)
                    .open_override(requested)
                    .default_open(false)
                    .default_pos(x, y)
                    .default_size(520.0, 360.0)
                    .show(ui, |ui| {
                        self.render_module_contents(ui, &owner, commands);
                    });
                window_open
            };
            self.record_window_open(module_id, open);
        }
    }

    fn record_window_open(&mut self, id: String, open: bool) {
        self.window_open.insert(id, open);
    }

    fn memo_filter_modules(&self) -> Vec<String> {
        self.modules
            .module_names()
            .into_iter()
            .chain(self.blackboard.memos.iter().map(|memo| memo.module.clone()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn resource_monitor_modules(&self) -> Vec<String> {
        self.modules
            .module_names()
            .into_iter()
            .chain(self.resource_monitor.module_names())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn record_runtime_event_for_monitor(&mut self, event: &RuntimeEvent) {
        self.resource_monitor
            .record_runtime_event_at(event, self.resource_monitor_elapsed_secs());
    }

    fn record_snapshot_for_monitor(&mut self, snapshot: &BlackboardSnapshot) {
        self.resource_monitor
            .record_snapshot_at(snapshot, self.resource_monitor_elapsed_secs());
    }

    fn resource_monitor_elapsed_secs(&self) -> f64 {
        self.resource_monitor_started_at.elapsed().as_secs_f64()
    }

    fn push_log(&mut self, message: String) {
        self.logs.push_back(message);
        if self.logs.len() > 512 {
            self.logs.pop_front();
        }
    }

    fn push_error(&mut self, error: VisualizerErrorView) {
        self.session_error_count = self.session_error_count.saturating_add(1);
        self.errors.push_back(error);
        if self.errors.len() > 256 {
            self.errors.pop_front();
        }
    }

    fn errors_window_id(&self) -> String {
        format!("{}:errors", self.id.as_str())
    }

    fn logs_ui(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            for log in &self.logs {
                text::wrapped_label(ui, log);
            }
        });
    }
}

fn simplified_section(
    ui: &mut egui::Ui,
    title: Option<&str>,
    size: egui::Vec2,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let size = egui::vec2(size.x.max(1.0), size.y.max(1.0));
    ui.allocate_ui_with_layout(size, egui::Layout::top_down(egui::Align::Min), |ui| {
        ui.set_min_size(size);
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.set_min_size(ui.available_size());
                if let Some(title) = title {
                    ui.strong(title);
                    ui.separator();
                }
                add_contents(ui);
            });
    });
}

fn simplified_auto_section(
    ui: &mut egui::Ui,
    title: Option<&str>,
    max_size: egui::Vec2,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let max_size = egui::vec2(max_size.x.max(1.0), max_size.y.max(1.0));
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.set_min_width(max_size.x);
            ui.set_max_width(max_size.x);
            ui.set_max_height(max_size.y);
            if let Some(title) = title {
                ui.strong(title);
                ui.separator();
            }
            add_contents(ui);
        });
}

fn simplified_modules_max_height(available_height: f32) -> f32 {
    (available_height - SIMPLIFIED_PANE_GAP - SIMPLIFIED_LOWER_PANES_MIN_HEIGHT).max(1.0)
}

fn simplified_module_popup_interaction_closes(
    popup_rect: egui::Rect,
    opened_this_frame: bool,
    primary_clicked: bool,
    interact_pos: Option<egui::Pos2>,
) -> bool {
    !opened_this_frame
        && primary_clicked
        && interact_pos.is_some_and(|pos| !popup_rect.contains(pos))
}

fn tab_status_icon(status: TabStatus) -> &'static str {
    match status {
        TabStatus::Running => "🟢",
        TabStatus::Passed => "✅",
        TabStatus::Failed => "❌",
        TabStatus::Invalid => "⚠️",
        TabStatus::Stopped => "⚪",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_batch_view() -> LlmBatchDebugView {
        LlmBatchDebugView {
            batch_type: "test::Batch".to_string(),
            debug: "batch".to_string(),
        }
    }

    fn test_i18n_context(locale: Locale) -> egui::Context {
        let ctx = egui::Context::default();
        let catalog = I18nCatalog::embedded().expect("embedded translations load");
        ctx.install_i18n(catalog.for_locale(locale));
        ctx
    }

    #[test]
    fn reducer_creates_and_updates_tabs() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply(VisualizerEvent::SetTabStatus {
            tab_id: tab_id.clone(),
            status: TabStatus::Passed,
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.title, "Case 1");
        assert_eq!(tab.status, TabStatus::Passed);
    }

    #[test]
    fn reducer_replaces_memory_records() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::MemoryRecordsLoaded {
            tab_id: tab_id.clone(),
            scope: MemoryRecordScope::Latest,
            offset: 0,
            records: vec![MemoryRecordView {
                index: "m1".to_string(),
                kind: "Statement".to_string(),
                rank: "episodic".to_string(),
                occurred_at: None,
                stored_at: chrono::Utc::now(),
                concepts: Vec::new(),
                tags: Vec::new(),
                affect_arousal: 0.0,
                valence: 0.0,
                emotion: String::new(),
                content: "learned rust".to_string(),
            }],
            has_more: false,
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.memories.scope, MemoryRecordScope::Latest);
        assert_eq!(tab.memories.records[0].content, "learned rust");
    }

    #[test]
    fn reducer_open_tab_marks_preopened_tab_running() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply(VisualizerEvent::SetTabStatus {
            tab_id: tab_id.clone(),
            status: TabStatus::Stopped,
        });
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.status, TabStatus::Running);
    }

    #[test]
    fn runtime_tab_defaults_to_simplified_view() {
        let tab = RuntimeTab::new(VisualizerTabId::new("case-1"), "Case 1".to_string());

        assert_eq!(tab.view_mode, RuntimeTabViewMode::Simplified);
        assert_eq!(
            tab.simplified_cognition_pane_tab,
            SimplifiedCognitionPaneTab::CognitionLog
        );
        assert_eq!(tab.active_simplified_module_owner, None);
    }

    #[test]
    fn runtime_tab_view_mode_toggles_between_simplified_and_windowed() {
        let mut tab = RuntimeTab::new(VisualizerTabId::new("case-1"), "Case 1".to_string());
        tab.open_simplified_module("sensory".to_string());
        tab.simplified_cognition_pane_tab = SimplifiedCognitionPaneTab::Memo;

        tab.set_simplified_view(false);

        assert_eq!(tab.view_mode, RuntimeTabViewMode::Windowed);
        assert_eq!(
            tab.simplified_cognition_pane_tab,
            SimplifiedCognitionPaneTab::Memo
        );
        assert_eq!(tab.active_simplified_module_owner, None);

        tab.set_simplified_view(true);

        assert_eq!(tab.view_mode, RuntimeTabViewMode::Simplified);
        assert_eq!(
            tab.simplified_cognition_pane_tab,
            SimplifiedCognitionPaneTab::Memo
        );
    }

    #[test]
    fn simplified_modules_max_height_keeps_lower_panes_visible() {
        let height = simplified_modules_max_height(600.0);

        assert_eq!(height, 412.0);
    }

    #[test]
    fn memo_filter_modules_combines_registered_modules_and_memo_modules() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply(VisualizerEvent::LlmObserved {
            tab_id: tab_id.clone(),
            event: LlmObservationEvent::ModelInput {
                turn_id: "turn-1".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                activation_id: 1,
                activation_attempt: 1,
                batch: test_batch_view(),
                items: Vec::new(),
            },
        });
        let tab = state.tabs.get_mut(&tab_id).expect("tab exists");
        tab.blackboard.memos = vec![MemoView {
            owner: "memory".to_string(),
            module: "memory".to_string(),
            replica: 0,
            index: 0,
            written_at: chrono::Utc::now(),
            cognitive: false,
            content: "memory memo".to_string(),
        }];

        assert_eq!(
            tab.memo_filter_modules(),
            vec!["memory".to_string(), "sensory".to_string()]
        );
    }

    #[test]
    fn zoom_percent_converts_to_factor() {
        assert_eq!(zoom_percent_to_factor(100.0), 1.0);
        assert_eq!(zoom_percent_to_factor(125.0), 1.25);
        assert_eq!(parse_zoom_percent_input("125"), Some(1.25));
        assert_eq!(parse_zoom_percent_input("125%"), Some(1.25));
        assert_eq!(parse_zoom_percent_input(""), None);
        assert_eq!(parse_zoom_percent_input("not a number"), None);
    }

    #[test]
    fn visualizer_theme_converts_to_and_from_egui_theme() {
        assert_eq!(
            VisualizerTheme::from(egui::Theme::Light),
            VisualizerTheme::Light
        );
        assert_eq!(
            VisualizerTheme::from(egui::Theme::Dark),
            VisualizerTheme::Dark
        );
        assert_eq!(
            egui::Theme::from(VisualizerTheme::Light),
            egui::Theme::Light
        );
        assert_eq!(egui::Theme::from(VisualizerTheme::Dark), egui::Theme::Dark);
    }

    #[test]
    fn visualizer_theme_label_keys_match_variants() {
        assert_eq!(VisualizerTheme::Light.label_key(), "menu-theme-light");
        assert_eq!(VisualizerTheme::Dark.label_key(), "menu-theme-dark");
    }

    #[test]
    fn default_visualizer_theme_uses_current_egui_theme() {
        let ctx = egui::Context::default();

        ctx.set_theme(egui::Theme::Light);
        assert_eq!(default_visualizer_theme(&ctx), VisualizerTheme::Light);

        ctx.set_theme(egui::Theme::Dark);
        assert_eq!(default_visualizer_theme(&ctx), VisualizerTheme::Dark);
    }

    #[test]
    fn visualizer_theme_styles_set_theme_specific_normal_text_color() {
        let ctx = egui::Context::default();

        install_visualizer_theme_styles(&ctx);

        for theme in [egui::Theme::Light, egui::Theme::Dark] {
            let style = ctx.style_of(theme);
            assert_eq!(
                style.visuals.override_text_color,
                visualizer_override_text_color(theme)
            );
            assert_eq!(
                style.visuals.weak_text_color,
                visualizer_weak_text_color(theme)
            );
            assert_eq!(
                style.visuals.text_options.alpha_from_coverage,
                visualizer_text_alpha_from_coverage(theme)
            );
            assert_eq!(
                style.visuals.selection.bg_fill,
                visualizer_selection_fill(theme)
            );
            assert_eq!(
                style.visuals.selection.stroke.color,
                visualizer_selection_text_color(theme)
            );
            assert_eq!(
                style.visuals.hyperlink_color,
                visualizer_hyperlink_color(theme)
            );
            assert_eq!(
                style.visuals.warn_fg_color,
                visualizer_warning_text_color(theme)
            );
            assert_eq!(
                style.visuals.error_fg_color,
                visualizer_error_text_color(theme)
            );
            assert_eq!(
                style.visuals.widgets.noninteractive.fg_stroke.color,
                visualizer_normal_text_color(theme)
            );
            assert_eq!(
                style.visuals.widgets.inactive.fg_stroke.color,
                visualizer_interactive_text_color(theme)
            );
            assert_eq!(
                style.visuals.widgets.hovered.fg_stroke.color,
                visualizer_interactive_text_color(theme)
            );
            assert_eq!(
                style.visuals.widgets.active.fg_stroke.color,
                visualizer_interactive_text_color(theme)
            );
            assert_eq!(
                style.visuals.widgets.open.fg_stroke.color,
                visualizer_interactive_text_color(theme)
            );
        }
    }

    #[test]
    fn visualizer_light_theme_text_colors_keep_readable_contrast() {
        let ctx = egui::Context::default();

        install_visualizer_theme_styles(&ctx);

        let style = ctx.style_of(egui::Theme::Light);
        let visuals = &style.visuals;
        let panel_fill = visuals.panel_fill;
        assert_contrast_at_least(visuals.text_color(), panel_fill, 4.5);
        assert_contrast_at_least(visuals.weak_text_color(), panel_fill, 4.5);
        assert_contrast_at_least(visuals.widgets.inactive.text_color(), panel_fill, 4.5);
        assert_contrast_at_least(visuals.widgets.hovered.text_color(), panel_fill, 4.5);
        assert_contrast_at_least(visuals.widgets.active.text_color(), panel_fill, 4.5);
        assert_contrast_at_least(visuals.widgets.open.text_color(), panel_fill, 4.5);
        assert_contrast_at_least(
            visuals.selection.stroke.color,
            visuals.selection.bg_fill,
            4.5,
        );
        assert_contrast_at_least(visuals.hyperlink_color, panel_fill, 4.5);
        assert_contrast_at_least(visuals.warn_fg_color, panel_fill, 4.5);
        assert_contrast_at_least(visuals.error_fg_color, panel_fill, 4.5);
    }

    #[test]
    fn visualizer_light_theme_tinted_fills_keep_readable_contrast() {
        let ctx = egui::Context::default();

        install_visualizer_theme_styles(&ctx);

        let style = ctx.style_of(egui::Theme::Light);
        let visuals = &style.visuals;
        let text = visuals.text_color();
        for fill in [
            visualizer_selection_message_fill(visuals),
            visualizer_selection_card_fill(visuals),
            visualizer_selection_cell_fill(visuals),
            visualizer_selection_row_fill(visuals),
            visualizer_error_subtle_fill(visuals),
            visualizer_error_banner_fill(visuals),
            visualizer_error_row_fill(visuals),
        ] {
            assert_contrast_at_least(text, fill, 4.5);
        }
        assert_contrast_at_least(
            visuals.error_fg_color,
            visualizer_error_banner_fill(visuals),
            4.5,
        );
    }

    fn assert_contrast_at_least(
        foreground: egui::Color32,
        background: egui::Color32,
        minimum: f32,
    ) {
        let actual = contrast_ratio(foreground, background);
        assert!(
            actual >= minimum,
            "contrast ratio {actual:.2} is below {minimum:.2} for {foreground:?} on {background:?}"
        );
    }

    fn contrast_ratio(left: egui::Color32, right: egui::Color32) -> f32 {
        let lighter = relative_luminance(left).max(relative_luminance(right));
        let darker = relative_luminance(left).min(relative_luminance(right));
        (lighter + 0.05) / (darker + 0.05)
    }

    fn relative_luminance(color: egui::Color32) -> f32 {
        let [r, g, b, _] = color.to_srgba_unmultiplied();
        0.2126 * srgb_channel_luminance(r)
            + 0.7152 * srgb_channel_luminance(g)
            + 0.0722 * srgb_channel_luminance(b)
    }

    fn srgb_channel_luminance(channel: u8) -> f32 {
        let channel = f32::from(channel) / 255.0;
        if channel <= 0.04045 {
            channel / 12.92
        } else {
            ((channel + 0.055) / 1.055).powf(2.4)
        }
    }

    #[test]
    fn zoom_factor_normalization_clamps_and_defaults() {
        assert_eq!(normalize_zoom_factor(0.01), MIN_ZOOM_FACTOR);
        assert_eq!(normalize_zoom_factor(100.0), MAX_ZOOM_FACTOR);
        assert_eq!(normalize_zoom_factor(f32::NAN), DEFAULT_ZOOM_FACTOR);
        assert_eq!(normalize_zoom_factor(f32::INFINITY), DEFAULT_ZOOM_FACTOR);
        assert_eq!(zoom_percent_to_factor(49.0), MIN_ZOOM_FACTOR);
        assert_eq!(zoom_percent_to_factor(201.0), MAX_ZOOM_FACTOR);
        assert_eq!(zoom_percent_to_factor(f32::NAN), DEFAULT_ZOOM_FACTOR);
    }

    #[test]
    fn zoom_step_changes_by_one_percent() {
        assert!((step_zoom_factor(1.0, 1.0) - 1.01).abs() < ZOOM_SYNC_EPSILON);
        assert!((step_zoom_factor(1.0, -1.0) - 0.99).abs() < ZOOM_SYNC_EPSILON);
        assert_eq!(step_zoom_factor(MIN_ZOOM_FACTOR, -1.0), MIN_ZOOM_FACTOR);
        assert_eq!(step_zoom_factor(MAX_ZOOM_FACTOR, 1.0), MAX_ZOOM_FACTOR);
    }

    #[test]
    fn zoom_second_double_click_step_completes_ten_percent_gesture() {
        let double_click_delta = ZOOM_BUTTON_DOUBLE_CLICK_TOTAL_PERCENT - ZOOM_BUTTON_STEP_PERCENT;
        assert!((step_zoom_factor(1.01, double_click_delta) - 1.10).abs() < ZOOM_SYNC_EPSILON);
        assert!((step_zoom_factor(0.99, -double_click_delta) - 0.90).abs() < ZOOM_SYNC_EPSILON);
    }

    #[test]
    fn reducer_tracks_offered_actions_by_scope() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply_server_message(VisualizerServerMessage::OfferAction {
            action: VisualizerAction::start_suite(),
        });
        state.apply_server_message(VisualizerServerMessage::OfferAction {
            action: VisualizerAction::start_activation(tab_id.clone()),
        });
        state.apply_server_message(VisualizerServerMessage::OfferAction {
            action: VisualizerAction::stop_runtime(tab_id.clone()),
        });

        let actions = state.visible_actions();
        assert_eq!(actions.len(), 3);
        assert!(
            actions
                .iter()
                .any(|action| action.id == START_SUITE_ACTION_ID)
        );
        assert!(
            actions
                .iter()
                .any(|action| action.id == start_activation_action_id(&tab_id))
        );
        assert!(
            actions
                .iter()
                .any(|action| action.id == stop_runtime_action_id(&tab_id))
        );

        state.apply_server_message(VisualizerServerMessage::RevokeAction {
            action_id: start_activation_action_id(&tab_id),
        });
        let actions = state.visible_actions();
        assert_eq!(actions.len(), 2);
        assert!(
            actions
                .iter()
                .any(|action| action.id == START_SUITE_ACTION_ID)
        );
        assert!(
            actions
                .iter()
                .any(|action| action.id == stop_runtime_action_id(&tab_id))
        );
    }

    #[test]
    fn view_window_specs_include_overview_and_module_windows() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply(VisualizerEvent::LlmObserved {
            tab_id: tab_id.clone(),
            event: LlmObservationEvent::ModelInput {
                turn_id: "turn-1".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                activation_id: 1,
                activation_attempt: 1,
                batch: test_batch_view(),
                items: Vec::new(),
            },
        });
        state
            .tabs
            .get_mut(&tab_id)
            .expect("tab exists")
            .set_simplified_view(false);

        let ctx = test_i18n_context(Locale::EnUs);
        let specs = state
            .tabs()
            .get(&tab_id)
            .expect("tab exists")
            .window_specs(&ctx);

        assert!(
            specs
                .iter()
                .any(|spec| { spec.id == "case-1:modules" && spec.title == "Modules - Case 1" })
        );
        assert!(
            specs
                .iter()
                .any(|spec| { spec.id == "case-1:chat" && spec.title == "Scene - Case 1" })
        );
        assert!(
            specs.iter().any(|spec| {
                spec.id == "case-1:llm-turns" && spec.title == "LLM Turns - Case 1"
            })
        );
        let module_spec = specs
            .iter()
            .find(|spec| spec.id == "case-1:module:sensory")
            .expect("module window spec exists");
        assert_eq!(module_spec.title, "Module - sensory");
        assert!(!module_spec.default_open);
        assert_eq!(module_spec.kind, ViewWindowKind::Module);
        assert!(
            !specs
                .iter()
                .any(|spec| spec.id.contains("simplified-interoception"))
        );
        assert!(
            !specs
                .iter()
                .any(|spec| spec.id.contains("simplified-module-popup"))
        );
    }

    #[test]
    fn simplified_module_popup_replaces_active_owner() {
        let mut tab = RuntimeTab::new(VisualizerTabId::new("case-1"), "Case 1".to_string());

        tab.open_simplified_module("sensory".to_string());
        tab.open_simplified_module("memory".to_string());

        assert_eq!(
            tab.active_simplified_module_owner.as_deref(),
            Some("memory")
        );
    }

    #[test]
    fn simplified_module_popup_outside_interaction_closes_active_owner() {
        let mut tab = RuntimeTab::new(VisualizerTabId::new("case-1"), "Case 1".to_string());
        let popup_rect = egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(100.0, 80.0));

        tab.open_simplified_module("sensory".to_string());
        tab.close_simplified_module_popup_for_interaction(
            popup_rect,
            false,
            true,
            Some(egui::pos2(50.0, 50.0)),
        );
        assert_eq!(
            tab.active_simplified_module_owner.as_deref(),
            Some("sensory")
        );

        tab.close_simplified_module_popup_for_interaction(
            popup_rect,
            true,
            true,
            Some(egui::pos2(200.0, 200.0)),
        );
        assert_eq!(
            tab.active_simplified_module_owner.as_deref(),
            Some("sensory")
        );

        tab.close_simplified_module_popup_for_interaction(
            popup_rect,
            false,
            true,
            Some(egui::pos2(200.0, 200.0)),
        );
        assert_eq!(tab.active_simplified_module_owner, None);
    }

    #[test]
    fn close_all_module_windows_requests_only_module_windows_closed() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");
        state.apply(VisualizerEvent::OpenTab {
            tab_id: tab_id.clone(),
            title: "Case 1".to_string(),
        });
        state.apply(VisualizerEvent::LlmObserved {
            tab_id: tab_id.clone(),
            event: LlmObservationEvent::ModelInput {
                turn_id: "turn-1".to_string(),
                owner: "sensory".to_string(),
                module: "sensory".to_string(),
                replica: 0,
                tier: "Default".to_string(),
                source: LlmObservationSource::ModuleTurn,
                session_key: None,
                operation: "text_turn".to_string(),
                activation_id: 1,
                activation_attempt: 1,
                batch: test_batch_view(),
                items: Vec::new(),
            },
        });
        let tab = state.tabs.get_mut(&tab_id).expect("tab exists");

        tab.close_all_module_windows();

        assert_eq!(
            tab.window_requests.get("case-1:module:sensory"),
            Some(&false)
        );
        assert!(!tab.window_requests.contains_key("case-1:memos"));
    }

    #[test]
    fn reducer_opens_errors_window_when_error_arrives() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");

        state.apply(VisualizerEvent::Error {
            tab_id: tab_id.clone(),
            error: VisualizerErrorView {
                at: chrono::Utc::now(),
                source: "runtime".to_string(),
                phase: "activate".to_string(),
                owner: Some("sensory".to_string()),
                message: "planned failure".to_string(),
            },
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.errors.len(), 1);
        assert_eq!(tab.session_error_count, 1);
        assert_eq!(state.selected.as_ref(), Some(&tab_id));
        assert_eq!(tab.window_requests.get("case-1:errors"), Some(&true));
    }

    #[test]
    fn reducer_applies_scene_state_to_selected_tab() {
        let mut state = VisualizerState::default();
        let tab_id = VisualizerTabId::new("case-1");

        state.apply(VisualizerEvent::SceneState {
            tab_id: tab_id.clone(),
            state: SceneStateView {
                people: vec![ScenePersonRowView {
                    id: "person-1".to_string(),
                    name: "Pibi".to_string(),
                    direction: "front".to_string(),
                    distance: "2m".to_string(),
                    state: "watching Nui".to_string(),
                }],
                derived_ambient: vec![DerivedAmbientSensoryRowView {
                    id: "scene:person:person-1".to_string(),
                    modality: "vision".to_string(),
                    content: "Pibi is present at front, 2m away; watching Nui.".to_string(),
                }],
                ..SceneStateView::default()
            },
        });

        let tab = state.tabs().get(&tab_id).expect("tab exists");
        assert_eq!(tab.scene.scene_view().people[0].name, "Pibi");
        assert_eq!(
            tab.scene.scene_view().derived_ambient[0].id,
            "scene:person:person-1"
        );
    }

    #[test]
    fn visualizer_fonts_prioritize_noto_for_proportional_and_monospace_text() {
        let fonts = visualizer_font_definitions(egui::FontData::from_owned(Vec::new()));

        let font_data = fonts
            .font_data
            .get(NOTO_SANS_JP_FONT_KEY)
            .expect("visualizer font is installed");
        assert_eq!(font_data.tweak, visualizer_font_tweak());
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            assert_eq!(
                fonts
                    .families
                    .get(&family)
                    .and_then(|families| families.first())
                    .map(String::as_str),
                Some(NOTO_SANS_JP_FONT_KEY)
            );
        }
    }

    #[test]
    fn visualizer_font_selection_requests_medium_weight() {
        assert_eq!(
            visualizer_font_properties().weight,
            NOTO_SANS_JP_FONT_WEIGHT
        );
    }
}
