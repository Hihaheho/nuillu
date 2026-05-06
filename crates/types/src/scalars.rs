use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A real number clamped to the closed interval [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, JsonSchema)]
pub struct UnitF32(f32);

impl UnitF32 {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);

    pub fn new(v: f32) -> Result<Self, UnitF32Error> {
        if v.is_nan() {
            return Err(UnitF32Error::Nan);
        }
        if !(0.0..=1.0).contains(&v) {
            return Err(UnitF32Error::OutOfRange(v));
        }
        Ok(Self(v))
    }

    pub fn clamp(v: f32) -> Self {
        if v.is_nan() {
            Self::ZERO
        } else {
            Self(v.clamp(0.0, 1.0))
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum UnitF32Error {
    #[error("UnitF32 cannot be NaN")]
    Nan,
    #[error("UnitF32 must be in [0.0, 1.0], got {0}")]
    OutOfRange(f32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nan() {
        assert_eq!(UnitF32::new(f32::NAN), Err(UnitF32Error::Nan));
    }

    #[test]
    fn rejects_out_of_range() {
        assert!(matches!(
            UnitF32::new(1.5),
            Err(UnitF32Error::OutOfRange(_))
        ));
    }

    #[test]
    fn clamp_handles_nan() {
        assert_eq!(UnitF32::clamp(f32::NAN).get(), 0.0);
    }
}
