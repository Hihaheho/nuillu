use std::sync::Arc;

use fluent_bundle::{FluentArgs, FluentResource, concurrent::FluentBundle};
use serde::{Deserialize, Serialize};
use unic_langid::LanguageIdentifier;

type Bundle = FluentBundle<Arc<FluentResource>>;

const I18N_CTX_KEY: &str = "nuillu-visualizer-i18n";
pub(crate) const LOCALE_PERSISTENCE_KEY: &str = "visualizer-locale";

const EN_US_FTL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/i18n/en-US/app.ftl"));
const JA_JP_FTL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/i18n/ja-JP/app.ftl"));

const MODULE_NAME_KEYS: &[(&str, &str)] = &[
    ("sensory", "module-name-sensory"),
    ("cognition-gate", "module-name-cognition-gate"),
    ("allocation", "module-name-allocation"),
    ("action", "module-name-action"),
    ("attention-schema", "module-name-attention-schema"),
    ("interpreter", "module-name-interpreter"),
    ("self-model", "module-name-self-model"),
    ("query-memory", "module-name-query-memory"),
    ("memory", "module-name-memory"),
    ("memory-compaction", "module-name-memory-compaction"),
    ("memory-association", "module-name-memory-association"),
    ("dreaming", "module-name-dreaming"),
    ("interoception", "module-name-interoception"),
    ("homeostasis", "module-name-homeostasis"),
    ("policy", "module-name-policy"),
    ("policy-compaction", "module-name-policy-compaction"),
    ("reward", "module-name-reward"),
    ("predict", "module-name-predict"),
    ("surprise", "module-name-surprise"),
    ("speak", "module-name-speak"),
    ("sleep", "module-name-sleep"),
    ("poet", "module-name-poet"),
    ("speak-gate", "module-name-speak-gate"),
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Locale {
    #[default]
    JaJp,
    EnUs,
}

impl Locale {
    pub fn label(self) -> &'static str {
        match self {
            Self::JaJp => "日本語",
            Self::EnUs => "English",
        }
    }

    fn language_tag(self) -> &'static str {
        match self {
            Self::JaJp => "ja-JP",
            Self::EnUs => "en-US",
        }
    }
}

/// User and embedded translations used by visualizer components.
///
/// [`crate::Visualizer`] installs its resources automatically. Lower-level
/// public modules fall back to the embedded default locale, while consumers can
/// install this value explicitly to select a locale before the first frame.
#[derive(Clone)]
pub struct VisualizerUiResources {
    catalog: I18nCatalog,
}

impl VisualizerUiResources {
    pub fn embedded() -> Result<Self, String> {
        Self::builder().build()
    }

    pub fn builder() -> VisualizerUiResourcesBuilder {
        VisualizerUiResourcesBuilder::default()
    }

    pub fn install(&self, ctx: &egui::Context, locale: Locale) {
        ctx.install_i18n(self.catalog.for_locale(locale));
    }

    /// Translates a key using the same user-plus-embedded fallback chain as the
    /// visualizer components.
    pub fn translate(&self, locale: Locale, id: &str) -> String {
        self.catalog.for_locale(locale).tr(id)
    }

    /// Translates a key with Fluent arguments.
    pub fn translate_args(&self, locale: Locale, id: &str, args: &[(&str, I18nArg<'_>)]) -> String {
        self.catalog.for_locale(locale).tr_args(id, args)
    }

    pub(crate) fn for_locale(&self, locale: Locale) -> Arc<I18n> {
        self.catalog.for_locale(locale)
    }
}

/// Builds visualizer translations from user FTL followed by embedded fallbacks.
#[derive(Debug, Default)]
pub struct VisualizerUiResourcesBuilder {
    ja: Vec<String>,
    en: Vec<String>,
}

impl VisualizerUiResourcesBuilder {
    /// Adds an FTL source for a locale.
    ///
    /// Sources added later have higher priority than earlier sources. User
    /// translations have priority over embedded translations.
    pub fn add_ftl(mut self, locale: Locale, source: impl Into<String>) -> Self {
        match locale {
            Locale::JaJp => self.ja.push(source.into()),
            Locale::EnUs => self.en.push(source.into()),
        }
        self
    }

    pub fn build(self) -> Result<VisualizerUiResources, String> {
        Ok(VisualizerUiResources {
            catalog: I18nCatalog::with_user_ftl(&self.ja, &self.en)?,
        })
    }
}

#[derive(Clone)]
pub(crate) struct I18nCatalog {
    ja: Arc<I18n>,
    en: Arc<I18n>,
}

impl I18nCatalog {
    pub(crate) fn embedded() -> Result<Self, String> {
        Self::with_user_ftl(&[], &[])
    }

    fn with_user_ftl(ja_user: &[String], en_user: &[String]) -> Result<Self, String> {
        let en_resources = locale_resources(Locale::EnUs, EN_US_FTL, en_user);
        let ja_resources = locale_resources(Locale::JaJp, JA_JP_FTL, ja_user);

        let en = Arc::new(I18n::load([&en_resources])?);
        let ja = Arc::new(I18n::load([&ja_resources, &en_resources])?);
        Ok(Self { ja, en })
    }

    pub(crate) fn for_locale(&self, locale: Locale) -> Arc<I18n> {
        match locale {
            Locale::JaJp => self.ja.clone(),
            Locale::EnUs => self.en.clone(),
        }
    }
}

struct LocaleResources<'a> {
    locale: Locale,
    sources: Vec<FtlSource<'a>>,
}

struct FtlSource<'a> {
    label: String,
    source: &'a str,
}

fn locale_resources<'a>(
    locale: Locale,
    embedded: &'a str,
    user_sources: &'a [String],
) -> LocaleResources<'a> {
    let mut sources = vec![FtlSource {
        label: format!("embedded `{}` FTL", locale.language_tag()),
        source: embedded,
    }];
    sources.extend(
        user_sources
            .iter()
            .enumerate()
            .map(|(index, source)| FtlSource {
                label: format!("user `{}` FTL #{}", locale.language_tag(), index + 1),
                source,
            }),
    );
    LocaleResources { locale, sources }
}

pub(crate) struct I18n {
    bundles: Vec<Bundle>,
}

impl I18n {
    fn load<'a>(
        locale_chain: impl IntoIterator<Item = &'a LocaleResources<'a>>,
    ) -> Result<Self, String> {
        let mut bundles = Vec::new();

        for resources in locale_chain {
            let locale = resources.locale.language_tag();
            let langid: LanguageIdentifier = locale
                .parse()
                .map_err(|error| format!("invalid locale `{locale}`: {error}"))?;
            let mut bundle: Bundle = FluentBundle::new_concurrent(vec![langid]);
            bundle.set_use_isolating(false);

            for input in &resources.sources {
                let resource = FluentResource::try_new(input.source.to_string()).map_err(
                    |(_resource, errors)| format!("failed to parse {}: {errors:?}", input.label),
                )?;
                bundle.add_resource_overriding(Arc::new(resource));
            }
            bundles.push(bundle);
        }

        Ok(Self { bundles })
    }

    fn tr(&self, id: &str) -> String {
        self.try_tr_fluent_args(id, None)
            .unwrap_or_else(|| format!("[[{id}]]"))
    }

    fn tr_args(&self, id: &str, args: &[(&str, I18nArg<'_>)]) -> String {
        let mut fluent_args = FluentArgs::new();
        for (name, value) in args {
            value.set(name, &mut fluent_args);
        }
        self.tr_fluent_args(id, Some(&fluent_args))
    }

    fn tr_fluent_args(&self, id: &str, args: Option<&FluentArgs<'_>>) -> String {
        self.try_tr_fluent_args(id, args)
            .unwrap_or_else(|| format!("[[{id}]]"))
    }

    fn try_tr_fluent_args(&self, id: &str, args: Option<&FluentArgs<'_>>) -> Option<String> {
        for bundle in &self.bundles {
            let Some(message) = bundle.get_message(id) else {
                continue;
            };
            let Some(pattern) = message.value() else {
                continue;
            };
            let mut errors = Vec::new();
            let value = bundle.format_pattern(pattern, args, &mut errors);
            if !errors.is_empty() {
                eprintln!("Fluent format errors for `{id}`: {errors:?}");
            }
            return Some(value.into_owned());
        }
        None
    }

    fn localized_module_name(&self, module_id: &str) -> String {
        let key = module_name_key(module_id)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("module-name-{module_id}"));
        self.try_tr_fluent_args(&key, None)
            .unwrap_or_else(|| module_id.to_string())
    }

    fn localized_module_name_with_id(&self, module_id: &str) -> String {
        let name = self.localized_module_name(module_id);
        if name == module_id {
            name
        } else {
            format!("{name} ({module_id})")
        }
    }
}

fn module_name_key(module_id: &str) -> Option<&'static str> {
    MODULE_NAME_KEYS
        .iter()
        .find_map(|(known_id, key)| (*known_id == module_id).then_some(*key))
}

#[derive(Debug, Clone)]
pub enum I18nArg<'a> {
    Str(&'a str),
    Owned(String),
    Usize(usize),
    U32(u32),
    I64(i64),
}

impl<'a> I18nArg<'a> {
    fn set<'args>(&'args self, name: &'args str, args: &mut FluentArgs<'args>) {
        match self {
            Self::Str(value) => args.set(name, *value),
            Self::Owned(value) => args.set(name, value.as_str()),
            Self::Usize(value) => args.set(name, *value as i64),
            Self::U32(value) => args.set(name, i64::from(*value)),
            Self::I64(value) => args.set(name, *value),
        };
    }
}

impl<'a> From<&'a str> for I18nArg<'a> {
    fn from(value: &'a str) -> Self {
        Self::Str(value)
    }
}

impl<'a> From<String> for I18nArg<'a> {
    fn from(value: String) -> Self {
        Self::Owned(value)
    }
}

impl<'a> From<usize> for I18nArg<'a> {
    fn from(value: usize) -> Self {
        Self::Usize(value)
    }
}

impl<'a> From<u32> for I18nArg<'a> {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}

impl<'a> From<i64> for I18nArg<'a> {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

pub(crate) trait EguiI18nExt {
    fn install_i18n(&self, i18n: Arc<I18n>);
    fn tr(&self, id: &str) -> String;
    fn tr_args(&self, id: &str, args: &[(&str, I18nArg<'_>)]) -> String;
}

fn installed_i18n(ctx: &egui::Context) -> Arc<I18n> {
    if let Some(i18n) = ctx.data(|data| data.get_temp::<Arc<I18n>>(egui::Id::new(I18N_CTX_KEY))) {
        return i18n;
    }

    let i18n = I18nCatalog::embedded()
        .expect("embedded visualizer translations should be valid")
        .for_locale(Locale::default());
    ctx.install_i18n(i18n.clone());
    i18n
}

pub(crate) fn localized_module_name(ctx: &egui::Context, module_id: &str) -> String {
    installed_i18n(ctx).localized_module_name(module_id)
}

pub(crate) fn localized_module_name_with_id(ctx: &egui::Context, module_id: &str) -> String {
    installed_i18n(ctx).localized_module_name_with_id(module_id)
}

impl EguiI18nExt for egui::Context {
    fn install_i18n(&self, i18n: Arc<I18n>) {
        self.data_mut(|data| {
            data.insert_temp(egui::Id::new(I18N_CTX_KEY), i18n);
        });
    }

    fn tr(&self, id: &str) -> String {
        installed_i18n(self).tr(id)
    }

    fn tr_args(&self, id: &str, args: &[(&str, I18nArg<'_>)]) -> String {
        installed_i18n(self).tr_args(id, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_loads_and_translates_both_locales() {
        let catalog = I18nCatalog::embedded().expect("embedded translations load");

        assert_eq!(
            catalog.for_locale(Locale::EnUs).tr("visualizer-locale-en"),
            "English"
        );
        assert_eq!(
            catalog.for_locale(Locale::JaJp).tr("visualizer-locale-ja"),
            "日本語"
        );
        assert_eq!(catalog.for_locale(Locale::EnUs).tr("menu-zoom"), "Zoom");
        assert_eq!(catalog.for_locale(Locale::JaJp).tr("menu-zoom"), "倍率");
        assert_eq!(
            catalog.for_locale(Locale::EnUs).tr("menu-theme-light"),
            "Light"
        );
        assert_eq!(
            catalog.for_locale(Locale::EnUs).tr("menu-theme-dark"),
            "Dark"
        );
        assert_eq!(
            catalog.for_locale(Locale::JaJp).tr("menu-theme-light"),
            "ライト"
        );
        assert_eq!(
            catalog.for_locale(Locale::JaJp).tr("menu-theme-dark"),
            "ダーク"
        );
        assert_eq!(
            catalog.for_locale(Locale::JaJp).tr("menu-theme-hover"),
            "visualizer のテーマを切り替えます。"
        );
        assert_eq!(
            catalog
                .for_locale(Locale::JaJp)
                .tr("module-name-attention-schema"),
            "注意スキーマ"
        );
        assert_eq!(
            catalog
                .for_locale(Locale::EnUs)
                .tr("module-name-attention-schema"),
            "attention-schema"
        );
    }

    #[test]
    fn ja_locale_falls_back_to_english() {
        let catalog = I18nCatalog::embedded().expect("embedded translations load");

        assert_eq!(
            catalog.for_locale(Locale::JaJp).tr("i18n-fallback-probe"),
            "English fallback"
        );
    }

    #[test]
    fn user_ftl_overrides_embedded_and_preserves_locale_fallbacks() {
        let resources = VisualizerUiResources::builder()
            .add_ftl(
                Locale::JaJp,
                "menu-zoom = ユーザー倍率\nuser-ja-only = ユーザー日本語\nuser-composed = { menu-theme-light } + ユーザー",
            )
            .add_ftl(
                Locale::EnUs,
                "i18n-fallback-probe = User English fallback\nuser-en-only = User English",
            )
            .build()
            .expect("user translations load");
        let ja = resources.for_locale(Locale::JaJp);
        let en = resources.for_locale(Locale::EnUs);

        assert_eq!(
            resources.translate(Locale::JaJp, "menu-zoom"),
            "ユーザー倍率"
        );
        assert_eq!(
            resources.translate(Locale::JaJp, "user-ja-only"),
            "ユーザー日本語"
        );
        assert_eq!(
            resources.translate(Locale::JaJp, "user-composed"),
            "ライト + ユーザー"
        );
        assert_eq!(ja.tr("menu-theme-light"), "ライト");
        assert_eq!(ja.tr("user-en-only"), "User English");
        assert_eq!(ja.tr("i18n-fallback-probe"), "User English fallback");
        assert_eq!(en.tr("i18n-fallback-probe"), "User English fallback");
        assert_eq!(en.tr("menu-theme-light"), "Light");
    }

    #[test]
    fn later_user_ftl_has_higher_priority() {
        let resources = VisualizerUiResources::builder()
            .add_ftl(Locale::EnUs, "menu-zoom = First user value")
            .add_ftl(Locale::EnUs, "menu-zoom = Second user value")
            .build()
            .expect("user translations load");

        assert_eq!(
            resources.for_locale(Locale::EnUs).tr("menu-zoom"),
            "Second user value"
        );
    }

    #[test]
    fn invalid_user_ftl_reports_its_source() {
        let error = VisualizerUiResources::builder()
            .add_ftl(Locale::JaJp, "broken = {")
            .build()
            .err()
            .expect("invalid user FTL is rejected");

        assert!(error.contains("user `ja-JP` FTL #1"), "{error}");
    }

    #[test]
    fn args_are_formatted() {
        let resources = VisualizerUiResources::embedded().expect("embedded translations load");

        assert_eq!(
            resources.translate_args(Locale::EnUs, "i18n-hello-name", &[("name", "egui".into())],),
            "Hello, egui."
        );
    }

    #[test]
    fn missing_keys_are_visible() {
        let catalog = I18nCatalog::embedded().expect("embedded translations load");

        assert_eq!(
            catalog.for_locale(Locale::EnUs).tr("missing-key"),
            "[[missing-key]]"
        );
    }

    #[test]
    fn localized_module_names_translate_known_ids_and_fallback_for_unknown_ids() {
        let catalog = I18nCatalog::embedded().expect("embedded translations load");
        let ja = catalog.for_locale(Locale::JaJp);
        let en = catalog.for_locale(Locale::EnUs);

        assert_eq!(ja.localized_module_name("sensory"), "感覚");
        assert_eq!(
            ja.localized_module_name_with_id("sensory"),
            "感覚 (sensory)"
        );
        assert_eq!(ja.localized_module_name("action"), "行動選択");
        assert_eq!(ja.localized_module_name("sleep"), "睡眠");
        assert_eq!(ja.localized_module_name("poet"), "詩作");
        assert_eq!(en.localized_module_name("sensory"), "sensory");
        assert_eq!(en.localized_module_name("action"), "action");
        assert_eq!(en.localized_module_name("sleep"), "sleep");
        assert_eq!(en.localized_module_name("poet"), "poet");
        assert_eq!(en.localized_module_name_with_id("sensory"), "sensory");
        assert_eq!(ja.localized_module_name("custom-module"), "custom-module");
        assert_eq!(
            ja.localized_module_name_with_id("custom-module"),
            "custom-module"
        );
    }

    #[test]
    fn user_ftl_can_name_a_dynamic_module() {
        let resources = VisualizerUiResources::builder()
            .add_ftl(Locale::JaJp, "module-name-ripgrep = ripgrep 検索")
            .add_ftl(Locale::EnUs, "module-name-ripgrep = ripgrep search")
            .build()
            .expect("dynamic module translations load");

        assert_eq!(
            resources
                .for_locale(Locale::JaJp)
                .localized_module_name("ripgrep"),
            "ripgrep 検索"
        );
        assert_eq!(
            resources
                .for_locale(Locale::EnUs)
                .localized_module_name_with_id("ripgrep"),
            "ripgrep search (ripgrep)"
        );
    }
}
