mod speak;
mod utterance;

#[cfg(test)]
mod test_support;

pub use speak::{SpeakModule, SpeakModuleParts};
pub use speak::{generation_session_auto_compaction, planning_session_auto_compaction};
pub use utterance::{
    NoopUtteranceSink, Utterance, UtteranceAbort, UtteranceDelta, UtteranceSink, UtteranceWriter,
};
