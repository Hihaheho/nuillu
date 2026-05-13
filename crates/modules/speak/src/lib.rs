mod gate;
mod speak;
mod utterance;

#[cfg(test)]
mod test_support;

pub use gate::{
    EvidenceGap, EvidenceGapSource, SpeakGateMemo, SpeakGateMemoKind, SpeakGateModule,
    SpeakGateSessionCompactionConfig,
};
pub use speak::SpeakModule;
pub use utterance::{NoopUtteranceSink, Utterance, UtteranceDelta, UtteranceSink, UtteranceWriter};
