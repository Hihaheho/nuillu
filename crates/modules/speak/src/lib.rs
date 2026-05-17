mod speak;
mod utterance;

#[cfg(test)]
mod test_support;

pub use speak::SpeakModule;
pub use utterance::{NoopUtteranceSink, Utterance, UtteranceDelta, UtteranceSink, UtteranceWriter};
