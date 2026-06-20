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
    ("attention-schema", "module-name-attention-schema"),
    ("interpreter", "module-name-interpreter"),
    ("self-model", "module-name-self-model"),
    ("query-memory", "module-name-query-memory"),
    ("memory", "module-name-memory"),
    ("memory-compaction", "module-name-memory-compaction"),
    ("memory-association", "module-name-memory-association"),
    ("memory-recombination", "module-name-memory-recombination"),
    ("interoception", "module-name-interoception"),
    ("homeostasis", "module-name-homeostasis"),
    ("policy", "module-name-policy"),
    ("policy-compaction", "module-name-policy-compaction"),
    ("reward", "module-name-reward"),
    ("predict", "module-name-predict"),
    ("surprise", "module-name-surprise"),
    ("speak", "module-name-speak"),
    ("speak-gate", "module-name-speak-gate"),
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Locale {
    #[default]
    JaJp,
    EnUs,
}

impl Locale {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::JaJp => "日本語",
            Self::EnUs => "English",
        }
    }
}

#[derive(Clone)]
pub(crate) struct I18nCatalog {
    ja: Arc<I18n>,
    en: Arc<I18n>,
}

impl I18nCatalog {
    pub(crate) fn embedded() -> Result<Self, String> {
        let en = Arc::new(I18n::load(&[("en-US", EN_US_FTL)]).map_err(|error| {
            format!("failed to load embedded en-US visualizer translations: {error}")
        })?);
        let ja = Arc::new(
            I18n::load(&[("ja-JP", JA_JP_FTL), ("en-US", EN_US_FTL)]).map_err(|error| {
                format!("failed to load embedded ja-JP visualizer translations: {error}")
            })?,
        );
        Ok(Self { ja, en })
    }

    pub(crate) fn for_locale(&self, locale: Locale) -> Arc<I18n> {
        match locale {
            Locale::JaJp => self.ja.clone(),
            Locale::EnUs => self.en.clone(),
        }
    }
}

pub(crate) struct I18n {
    bundles: Vec<Bundle>,
}

impl I18n {
    fn load(locale_chain: &[(&str, &str)]) -> Result<Self, String> {
        let mut bundles = Vec::new();

        for (locale, source) in locale_chain {
            let langid: LanguageIdentifier = locale
                .parse()
                .map_err(|error| format!("invalid locale `{locale}`: {error}"))?;
            let resource =
                FluentResource::try_new((*source).to_string()).map_err(|(_resource, errors)| {
                    format!("failed to parse `{locale}` FTL: {errors:?}")
                })?;
            let mut bundle: Bundle = FluentBundle::new_concurrent(vec![langid]);
            bundle.set_use_isolating(false);
            bundle
                .add_resource(Arc::new(resource))
                .map_err(|errors| format!("failed to add `{locale}` FTL: {errors:?}"))?;
            bundles.push(bundle);
        }

        Ok(Self { bundles })
    }

    fn tr(&self, id: &str) -> String {
        self.tr_fluent_args(id, None)
    }

    fn tr_args(&self, id: &str, args: &[(&str, I18nArg<'_>)]) -> String {
        let mut fluent_args = FluentArgs::new();
        for (name, value) in args {
            value.set(name, &mut fluent_args);
        }
        self.tr_fluent_args(id, Some(&fluent_args))
    }

    fn tr_fluent_args(&self, id: &str, args: Option<&FluentArgs<'_>>) -> String {
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
            return value.into_owned();
        }

        format!("[[{id}]]")
    }

    fn localized_module_name(&self, module_id: &str) -> String {
        module_name_key(module_id)
            .map(|key| self.tr(key))
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
pub(crate) enum I18nArg<'a> {
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
    ctx.data(|data| data.get_temp::<Arc<I18n>>(egui::Id::new(I18N_CTX_KEY)))
        .expect("I18n is not installed in egui::Context")
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
    fn args_are_formatted() {
        let catalog = I18nCatalog::embedded().expect("embedded translations load");

        assert_eq!(
            catalog
                .for_locale(Locale::EnUs)
                .tr_args("i18n-hello-name", &[("name", "egui".into())]),
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
        assert_eq!(en.localized_module_name("sensory"), "sensory");
        assert_eq!(en.localized_module_name_with_id("sensory"), "sensory");
        assert_eq!(ja.localized_module_name("custom-module"), "custom-module");
        assert_eq!(
            ja.localized_module_name_with_id("custom-module"),
            "custom-module"
        );
    }
}
