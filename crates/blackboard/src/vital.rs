use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VitalMode {
    Wake,
    NremPressure,
    RemPressure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitalState {
    pub mode: VitalMode,
    pub wake_arousal: f32,
    pub nrem_pressure: f32,
    pub rem_pressure: f32,
    pub last_updated: DateTime<Utc>,
}

impl Default for VitalState {
    fn default() -> Self {
        Self {
            mode: VitalMode::Wake,
            wake_arousal: 0.0,
            nrem_pressure: 0.0,
            rem_pressure: 0.0,
            last_updated: DateTime::<Utc>::from_timestamp(0, 0)
                .expect("unix epoch timestamp is valid"),
        }
    }
}

impl VitalState {
    pub fn apply_patch(&mut self, patch: VitalPatch, now: DateTime<Utc>) {
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
        self.last_updated = now;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VitalPatch {
    pub mode: Option<VitalMode>,
    pub wake_arousal: Option<f32>,
    pub nrem_pressure: Option<f32>,
    pub rem_pressure: Option<f32>,
}

fn clamp_unit(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_updates_only_specified_fields_and_clamps() {
        let mut state = VitalState {
            mode: VitalMode::Wake,
            wake_arousal: 0.4,
            nrem_pressure: 0.2,
            rem_pressure: 0.3,
            last_updated: DateTime::<Utc>::from_timestamp(1, 0).unwrap(),
        };
        let now = DateTime::<Utc>::from_timestamp(2, 0).unwrap();

        state.apply_patch(
            VitalPatch {
                mode: Some(VitalMode::NremPressure),
                wake_arousal: Some(1.4),
                nrem_pressure: None,
                rem_pressure: Some(f32::NAN),
            },
            now,
        );

        assert_eq!(state.mode, VitalMode::NremPressure);
        assert_eq!(state.wake_arousal, 1.0);
        assert_eq!(state.nrem_pressure, 0.2);
        assert_eq!(state.rem_pressure, 0.0);
        assert_eq!(state.last_updated, now);
    }
}
