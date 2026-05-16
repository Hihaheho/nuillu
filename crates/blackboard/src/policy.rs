use chrono::{DateTime, Utc};
use nuillu_types::{PolicyIndex, PolicyRank, SignedUnitF32, UnitF32};
use serde::{Deserialize, Serialize};

/// Startup-loaded core policy content. This mirrors identity memory seeds but
/// keeps policy behavior separate from memory content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorePolicyRecord {
    pub index: PolicyIndex,
    pub trigger: String,
    pub behavior: String,
}

/// Mutable metadata about a policy entry. Durable trigger/behavior content and
/// adapter-local search state live in the external `PolicyStore`, not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMetadata {
    pub index: PolicyIndex,
    pub rank: PolicyRank,
    pub expected_reward: SignedUnitF32,
    pub confidence: UnitF32,
    pub value: SignedUnitF32,
    pub reward_tokens: u32,
    pub decay_remaining_secs: i64,
    pub last_reinforced_at: Option<DateTime<Utc>>,
    /// Count of reward-credited TD updates applied to this policy. This is
    /// separate from `reward_tokens`, which only counts settled predictions
    /// used for rank thresholds.
    #[serde(default)]
    pub reinforcement_count: u32,
}

impl PolicyMetadata {
    pub fn new_at(index: PolicyIndex, rank: PolicyRank, decay_remaining_secs: i64) -> Self {
        Self {
            index,
            rank,
            expected_reward: SignedUnitF32::ZERO,
            confidence: UnitF32::ZERO,
            value: SignedUnitF32::ZERO,
            reward_tokens: 0,
            decay_remaining_secs,
            last_reinforced_at: None,
            reinforcement_count: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PolicyMetaPatch {
    pub rank: Option<PolicyRank>,
    pub expected_reward: Option<SignedUnitF32>,
    pub confidence: Option<UnitF32>,
    pub value: Option<SignedUnitF32>,
    pub reward_tokens: Option<u32>,
    pub decay_remaining_secs: Option<i64>,
    pub reinforced_at: Option<DateTime<Utc>>,
}

impl PolicyMetaPatch {
    pub fn apply(self, meta: &mut PolicyMetadata) {
        if let Some(rank) = self.rank {
            meta.rank = rank;
        }
        if let Some(expected_reward) = self.expected_reward {
            meta.expected_reward = expected_reward;
        }
        if let Some(confidence) = self.confidence {
            meta.confidence = confidence;
        }
        if let Some(value) = self.value {
            meta.value = value;
        }
        if let Some(reward_tokens) = self.reward_tokens {
            meta.reward_tokens = reward_tokens;
        }
        if let Some(decay_remaining_secs) = self.decay_remaining_secs {
            meta.decay_remaining_secs = decay_remaining_secs;
        }
        if let Some(reinforced_at) = self.reinforced_at {
            meta.last_reinforced_at = Some(reinforced_at);
            meta.reinforcement_count = meta.reinforcement_count.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_deserializes_without_reinforcement_count() {
        let value = serde_json::json!({
            "index": "policy-1",
            "rank": "Tentative",
            "expected_reward": 0.0,
            "confidence": 0.0,
            "value": 0.0,
            "reward_tokens": 0,
            "decay_remaining_secs": 10,
            "last_reinforced_at": null
        });

        let decoded: PolicyMetadata = serde_json::from_value(value).unwrap();

        assert_eq!(decoded.reinforcement_count, 0);
    }
}
