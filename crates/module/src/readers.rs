//! Read-only views over agent state.
//!
//! Each reader exposes only the slice of the blackboard the design
//! permits the holding module to see. The compile-time signal is the
//! constructor signature: a module that takes only [`AttentionReader`]
//! cannot read the non-cognitive blackboard, regardless of what its
//! `run` body tries.

use nuillu_blackboard::{AttentionStream, Blackboard, BlackboardInner, ResourceAllocation};

/// Read-only access to the entire blackboard (memos + memory metadata).
///
/// Held by modules that legitimately need a wide view (summarize,
/// query, memory, memory-compaction). Pointedly *not* held by the
/// attention controller, which is restricted to the cognitive surface.
#[derive(Clone)]
pub struct BlackboardReader {
    blackboard: Blackboard,
}

impl BlackboardReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    /// Apply `f` to a borrowed snapshot. The read lock is held for the
    /// duration of `f`; do not await inside it.
    pub async fn read<R>(&self, f: impl FnOnce(&BlackboardInner) -> R) -> R {
        self.blackboard.read(f).await
    }
}

/// Read-only access to the cognitive attention stream. The holder
/// cannot see memos, memory metadata, or allocation through this
/// capability.
#[derive(Clone)]
pub struct AttentionReader {
    blackboard: Blackboard,
}

impl AttentionReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn read<R>(&self, f: impl FnOnce(&AttentionStream) -> R) -> R {
        self.blackboard
            .read(|bb| {
                let stream = bb.attention_stream();
                f(&stream)
            })
            .await
    }

    pub async fn snapshot(&self) -> nuillu_blackboard::AttentionStreamSet {
        self.blackboard.read(|bb| bb.attention_stream_set()).await
    }
}

/// Read-only access to the resource-allocation snapshot. Issued to
/// modules that need to consult allocation for their own decisions
/// (typically only the attention controller).
#[derive(Clone)]
pub struct AllocationReader {
    blackboard: Blackboard,
}

impl AllocationReader {
    pub(crate) fn new(blackboard: Blackboard) -> Self {
        Self { blackboard }
    }

    pub async fn read<R>(&self, f: impl FnOnce(&ResourceAllocation) -> R) -> R {
        self.blackboard.read(|bb| f(bb.allocation())).await
    }

    pub async fn snapshot(&self) -> ResourceAllocation {
        self.blackboard.read(|bb| bb.allocation().clone()).await
    }

    pub async fn replica_caps(
        &self,
    ) -> std::collections::HashMap<nuillu_types::ModuleId, nuillu_types::ReplicaCapRange> {
        self.blackboard.read(|bb| bb.replica_caps().clone()).await
    }

    pub async fn controller_schema_json(&self) -> serde_json::Value {
        let caps = self.replica_caps().await;
        let mut caps = caps.into_iter().collect::<Vec<_>>();
        caps.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));

        let patch_branches = caps
            .into_iter()
            .map(|(id, range)| {
                serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "module_id": {
                            "enum": [id.as_str()],
                        },
                        "replicas": {
                            "type": ["integer", "null"],
                            "minimum": range.min,
                            "maximum": range.max,
                        },
                        "tier": {
                            "enum": ["Cheap", "Default", "Premium", null],
                        },
                        "period_ms": {
                            "type": ["integer", "null"],
                            "minimum": 0,
                        },
                        "message_only": {
                            "type": "boolean",
                        },
                        "context_budget_tokens": {
                            "type": ["integer", "null"],
                            "minimum": 0,
                        },
                    },
                    "required": [
                        "module_id",
                        "replicas",
                        "tier",
                        "period_ms",
                        "message_only",
                        "context_budget_tokens",
                    ],
                })
            })
            .collect::<Vec<_>>();

        let patch_items = if patch_branches.is_empty() {
            serde_json::Value::Bool(false)
        } else {
            serde_json::json!({ "anyOf": patch_branches })
        };

        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "memo": {
                    "type": "string",
                },
                "patches": {
                    "type": "array",
                    "items": patch_items,
                },
            },
            "required": ["memo", "patches"],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use nuillu_blackboard::BlackboardCommand;
    use nuillu_types::{ReplicaCapRange, builtin};

    #[tokio::test]
    async fn controller_schema_enumerates_registered_modules_with_cap_ranges() {
        let blackboard = Blackboard::default();
        blackboard
            .apply(BlackboardCommand::SetReplicaCaps {
                caps: vec![
                    (builtin::query_vector(), ReplicaCapRange { min: 0, max: 3 }),
                    (builtin::speak(), ReplicaCapRange { min: 0, max: 1 }),
                ],
            })
            .await;
        let reader = AllocationReader::new(blackboard);

        let schema = reader.controller_schema_json().await;
        assert_eq!(
            schema,
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "memo": {
                        "type": "string",
                    },
                    "patches": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": false,
                                    "properties": {
                                        "module_id": {
                                            "enum": ["query-vector"],
                                        },
                                        "replicas": {
                                            "type": ["integer", "null"],
                                            "minimum": 0,
                                            "maximum": 3,
                                        },
                                        "tier": {
                                            "enum": ["Cheap", "Default", "Premium", null],
                                        },
                                        "period_ms": {
                                            "type": ["integer", "null"],
                                            "minimum": 0,
                                        },
                                        "message_only": {
                                            "type": "boolean",
                                        },
                                        "context_budget_tokens": {
                                            "type": ["integer", "null"],
                                            "minimum": 0,
                                        },
                                    },
                                    "required": [
                                        "module_id",
                                        "replicas",
                                        "tier",
                                        "period_ms",
                                        "message_only",
                                        "context_budget_tokens",
                                    ],
                                },
                                {
                                    "type": "object",
                                    "additionalProperties": false,
                                    "properties": {
                                        "module_id": {
                                            "enum": ["speak"],
                                        },
                                        "replicas": {
                                            "type": ["integer", "null"],
                                            "minimum": 0,
                                            "maximum": 1,
                                        },
                                        "tier": {
                                            "enum": ["Cheap", "Default", "Premium", null],
                                        },
                                        "period_ms": {
                                            "type": ["integer", "null"],
                                            "minimum": 0,
                                        },
                                        "message_only": {
                                            "type": "boolean",
                                        },
                                        "context_budget_tokens": {
                                            "type": ["integer", "null"],
                                            "minimum": 0,
                                        },
                                    },
                                    "required": [
                                        "module_id",
                                        "replicas",
                                        "tier",
                                        "period_ms",
                                        "message_only",
                                        "context_budget_tokens",
                                    ],
                                },
                            ],
                        },
                    },
                },
                "required": ["memo", "patches"],
            })
        );
    }
}
