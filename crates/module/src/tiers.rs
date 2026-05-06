use lutum::Lutum;
use nuillu_types::ModelTier;

/// One [`Lutum`] per coarse model tier. Constructed once at boot and
/// shared by every [`LlmAccess`](crate::LlmAccess) capability handle.
///
/// `Lutum` is `Clone` (a thin wrapper around `Arc`-shared adapter and
/// budget manager); cloning is cheap and intentional.
#[derive(Clone)]
pub struct LutumTiers {
    pub cheap: Lutum,
    pub default: Lutum,
    pub premium: Lutum,
}

impl LutumTiers {
    pub fn pick(&self, tier: ModelTier) -> Lutum {
        match tier {
            ModelTier::Cheap => self.cheap.clone(),
            ModelTier::Default => self.default.clone(),
            ModelTier::Premium => self.premium.clone(),
        }
    }
}
