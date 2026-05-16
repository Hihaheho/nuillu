use eure::{
    IntoEure,
    document::{constructor::DocumentConstructor, write::WriteError},
    edit::{EditError, EditableDocument},
    value::Text,
};
use eure_document::plan::{LayoutPlan, PlanError};
use eure_fmt::format_source_document;
use serde::{Serialize, Serializer};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DumpText {
    content: String,
}

impl DumpText {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.content
    }
}

impl Serialize for DumpText {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.content)
    }
}

impl IntoEure for DumpText {
    type Error = WriteError;

    fn write(value: Self, c: &mut DocumentConstructor) -> Result<(), Self::Error> {
        c.write(Text::block_implicit(value.content))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct FullAgentLastStateDump {
    pub case: FullAgentLastStateCaseDump,
    pub blackboard: BlackboardLastStateDump,
    pub memory: MemoryLastStateDump,
    pub utterances: Vec<UtteranceDump>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct FullAgentLastStateCaseDump {
    pub id: String,
    pub dumped_at: String,
    pub event_count: u64,
    pub output: Option<DumpText>,
    pub failure: Option<DumpText>,
}

#[derive(Debug, Clone, PartialEq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct BlackboardLastStateDump {
    pub memo_logs: Vec<MemoLogDump>,
    pub cognition_logs: Vec<CognitionLogDump>,
    pub agentic_deadlock: Option<AgenticDeadlockDump>,
    pub base_allocation: Vec<AllocationModuleDump>,
    pub allocation: Vec<AllocationModuleDump>,
    pub allocation_proposals: Vec<AllocationProposalDump>,
    pub replica_caps: Vec<ReplicaCapDump>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemoLogDump {
    pub module: String,
    pub replica: u8,
    pub index: u64,
    pub written_at: String,
    pub content: DumpText,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct CognitionLogDump {
    pub source: ModuleInstanceDump,
    pub entries: Vec<CognitionEntryDump>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct CognitionEntryDump {
    pub at: String,
    pub text: DumpText,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct AgenticDeadlockDump {
    pub at: String,
    pub idle_for_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct AllocationModuleDump {
    pub module: String,
    pub activation_ratio: f64,
    pub active_replicas: u8,
    pub tier: String,
    pub guidance: DumpText,
}

#[derive(Debug, Clone, PartialEq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct AllocationProposalDump {
    pub controller: ModuleInstanceDump,
    pub modules: Vec<AllocationModuleDump>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ReplicaCapDump {
    pub module: String,
    pub min: u8,
    pub max: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct ModuleInstanceDump {
    pub module: String,
    pub replica: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemoryLastStateDump {
    pub entries: Vec<MemoryEntryDump>,
}

#[derive(Debug, Clone, PartialEq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemoryEntryDump {
    pub index: String,
    pub content: Option<DumpText>,
    pub content_rank: Option<String>,
    pub occurred_at: Option<String>,
    pub affect_arousal: f32,
    pub valence: f32,
    pub emotion: String,
    pub metadata: MemoryMetadataDump,
    pub missing_content: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct MemoryMetadataDump {
    pub rank: String,
    pub occurred_at: Option<String>,
    pub decay_remaining_secs: i64,
    pub remember_tokens: u32,
    pub last_accessed: String,
    pub access_count: u32,
    pub use_count: u32,
    pub last_used: Option<String>,
    pub reinforcement_count: u32,
    pub last_reinforced_at: Option<String>,
    pub query_history: Vec<String>,
    pub use_history: Vec<String>,
    pub reinforcement_history: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, IntoEure)]
#[eure(crate = ::eure::document, rename_all = "kebab-case")]
pub struct UtteranceDump {
    pub sender: String,
    pub target: String,
    pub text: DumpText,
    pub emitted_at: String,
}

#[derive(Debug, Error)]
pub enum StateDumpRenderError {
    #[error("failed to write dump as Eure document: {0}")]
    Write(#[from] WriteError),
    #[error("failed to build dump source layout: {0}")]
    Plan(#[from] PlanError),
    #[error("failed to parse formatted dump with EditableDocument: {0}")]
    Edit(#[from] EditError),
}

pub fn render_full_agent_last_state_eure(
    dump: FullAgentLastStateDump,
) -> Result<String, StateDumpRenderError> {
    let mut constructor = DocumentConstructor::new();
    constructor.write(dump)?;
    let document = constructor.finish();
    let source = LayoutPlan::auto(document)?.emit();
    let formatted = format_source_document(&source);
    let editable = EditableDocument::parse(&formatted)?;
    Ok(editable.render())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_last_state_dump_as_parseable_eure() {
        let dump = FullAgentLastStateDump {
            case: FullAgentLastStateCaseDump {
                id: "case-1".to_string(),
                dumped_at: "2026-05-08T00:00:00Z".to_string(),
                event_count: 2,
                output: Some(DumpText::new("hello")),
                failure: None,
            },
            blackboard: BlackboardLastStateDump {
                memo_logs: vec![MemoLogDump {
                    module: "sensory".to_string(),
                    replica: 0,
                    index: 0,
                    written_at: "2026-05-08T00:00:00Z".to_string(),
                    content: DumpText::new("memo\nwith newline"),
                }],
                cognition_logs: vec![CognitionLogDump {
                    source: ModuleInstanceDump {
                        module: "cognition-gate".to_string(),
                        replica: 0,
                    },
                    entries: vec![CognitionEntryDump {
                        at: "2026-05-08T00:00:00Z".to_string(),
                        text: DumpText::new("cognition"),
                    }],
                }],
                agentic_deadlock: None,
                base_allocation: Vec::new(),
                allocation: Vec::new(),
                allocation_proposals: Vec::new(),
                replica_caps: Vec::new(),
            },
            memory: MemoryLastStateDump {
                entries: vec![MemoryEntryDump {
                    index: "memory-1".to_string(),
                    content: Some(DumpText::new("remember this")),
                    content_rank: Some("permanent".to_string()),
                    occurred_at: Some("2026-05-07T00:00:00Z".to_string()),
                    affect_arousal: 0.0,
                    valence: 0.0,
                    emotion: String::new(),
                    metadata: MemoryMetadataDump {
                        rank: "permanent".to_string(),
                        occurred_at: Some("2026-05-07T00:00:00Z".to_string()),
                        decay_remaining_secs: 0,
                        remember_tokens: 1,
                        last_accessed: "2026-05-08T00:00:00Z".to_string(),
                        access_count: 0,
                        use_count: 0,
                        last_used: None,
                        reinforcement_count: 0,
                        last_reinforced_at: None,
                        query_history: Vec::new(),
                        use_history: Vec::new(),
                        reinforcement_history: Vec::new(),
                    },
                    missing_content: false,
                }],
            },
            utterances: vec![UtteranceDump {
                sender: "speak".to_string(),
                target: "Koro".to_string(),
                text: DumpText::new("hello"),
                emitted_at: "2026-05-08T00:00:00Z".to_string(),
            }],
        };

        let rendered = render_full_agent_last_state_eure(dump).unwrap();
        assert!(rendered.contains("case"));
        assert!(rendered.contains("memory"));
        EditableDocument::parse(&rendered).unwrap();
    }
}
