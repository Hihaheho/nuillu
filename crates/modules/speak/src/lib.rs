mod gate;
mod speak;

#[cfg(test)]
mod test_support;

pub use gate::{
    EvidenceGap, EvidenceGapSource, SpeakGateMemo, SpeakGateMemoKind, SpeakGateModule,
    SpeakGateSessionCompactionConfig,
};
pub use speak::SpeakModule;
