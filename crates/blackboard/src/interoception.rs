use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteroceptiveMode {
    Wake,
    NremPressure,
    RemPressure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteroceptiveState {
    pub mode: InteroceptiveMode,
    pub wake_arousal: f32,
    pub nrem_pressure: f32,
    pub rem_pressure: f32,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
    pub last_updated: DateTime<Utc>,
}

impl Default for InteroceptiveState {
    fn default() -> Self {
        Self {
            mode: InteroceptiveMode::Wake,
            wake_arousal: 0.0,
            nrem_pressure: 0.0,
            rem_pressure: 0.0,
            affect_arousal: 0.0,
            valence: 0.0,
            emotion: String::new(),
            last_updated: DateTime::<Utc>::from_timestamp(0, 0)
                .expect("unix epoch timestamp is valid"),
        }
    }
}

impl InteroceptiveState {
    pub fn apply_patch(&mut self, patch: InteroceptivePatch, now: DateTime<Utc>) {
        if let Some(mode) = patch.mode {
            self.mode = mode;
        }
        if let Some(wake_arousal) = patch.wake_arousal {
            self.wake_arousal = clamp_unit(wake_arousal);
        }
        if let Some(nrem_pressure) = patch.nrem_pressure {
            self.nrem_pressure = clamp_unit(nrem_pressure);
        }
        if let Some(rem_pressure) = patch.rem_pressure {
            self.rem_pressure = clamp_unit(rem_pressure);
        }
        if let Some(affect_arousal) = patch.affect_arousal {
            self.affect_arousal = clamp_unit(affect_arousal);
        }
        if let Some(valence) = patch.valence {
            self.valence = clamp_signed_unit(valence);
        }
        if let Some(emotion) = patch.emotion {
            self.emotion = emotion.trim().to_owned();
        }
        self.last_updated = now;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InteroceptivePatch {
    pub mode: Option<InteroceptiveMode>,
    pub wake_arousal: Option<f32>,
    pub nrem_pressure: Option<f32>,
    pub rem_pressure: Option<f32>,
    pub affect_arousal: Option<f32>,
    pub valence: Option<f32>,
    pub emotion: Option<String>,
}

fn clamp_unit(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn clamp_signed_unit(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_updates_only_specified_fields_and_clamps() {
        let mut state = InteroceptiveState {
            mode: InteroceptiveMode::Wake,
            wake_arousal: 0.4,
            nrem_pressure: 0.2,
            rem_pressure: 0.3,
            affect_arousal: 0.5,
            valence: -0.2,
            emotion: "uneasy".to_owned(),
            last_updated: DateTime::<Utc>::from_timestamp(1, 0).unwrap(),
        };
        let now = DateTime::<Utc>::from_timestamp(2, 0).unwrap();

        state.apply_patch(
            InteroceptivePatch {
                mode: Some(InteroceptiveMode::NremPressure),
                wake_arousal: Some(1.4),
                nrem_pressure: None,
                rem_pressure: Some(f32::NAN),
                affect_arousal: Some(-0.5),
                valence: Some(1.5),
                emotion: Some("  relieved and alert  ".to_owned()),
            },
            now,
        );

        assert_eq!(state.mode, InteroceptiveMode::NremPressure);
        assert_eq!(state.wake_arousal, 1.0);
        assert_eq!(state.nrem_pressure, 0.2);
        assert_eq!(state.rem_pressure, 0.0);
        assert_eq!(state.affect_arousal, 0.0);
        assert_eq!(state.valence, 1.0);
        assert_eq!(state.emotion, "relieved and alert");
        assert_eq!(state.last_updated, now);
    }
}
