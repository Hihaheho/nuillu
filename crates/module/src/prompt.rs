use chrono::{DateTime, Utc};
use nuillu_blackboard::IdentityMemoryRecord;
use nuillu_types::ModuleId;

use crate::render_memory_for_llm;

/// Build a system prompt that prepends a one-line description of every other
/// module registered in the agent. The owner module is excluded so each
/// module's prompt only lists its peers.
///
/// The catalog is post-boot static state (`Blackboard::module_catalog`) so
/// the produced string is stable across activations and friendly to LLM
/// prompt caching as long as `base` is also stable.
pub fn format_system_prompt(
    base: &str,
    catalog: &[(ModuleId, &'static str)],
    owner: &ModuleId,
    identity_memories: &[IdentityMemoryRecord],
    now: DateTime<Utc>,
) -> String {
    let mut peers = catalog
        .iter()
        .filter(|(id, _)| id != owner)
        .map(|(id, role)| format!("- {}: {}", id, role))
        .collect::<Vec<_>>();
    peers.sort();

    let mut prompt = base.to_owned();
    if !peers.is_empty() {
        prompt.push_str("\n\nYou are part of a cognitive system. Other modules in this brain:\n");
        prompt.push_str(&peers.join("\n"));
        prompt.push('\n');
    }
    if !identity_memories.is_empty() {
        if peers.is_empty() {
            prompt.push_str("\n\n");
        } else {
            prompt.push('\n');
        }
        prompt.push_str("Identity memory loaded at agent startup:\n");
        for memory in identity_memories {
            prompt.push_str("- [");
            prompt.push_str(memory.index.as_str());
            prompt.push_str("] ");
            prompt.push_str(&render_memory_for_llm(
                memory.content.as_str(),
                memory.occurred_at,
                now,
            ));
            prompt.push('\n');
        }
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;
    use nuillu_types::builtin;

    #[test]
    fn excludes_owner_and_sorts_peers() {
        let catalog = vec![
            (builtin::sensory(), "sensory role"),
            (builtin::speak(), "speak role"),
            (builtin::cognition_gate(), "gate role"),
        ];
        let prompt = format_system_prompt("BASE", &catalog, &builtin::sensory(), &[], test_now());
        assert!(prompt.starts_with("BASE\n\nYou are part of a cognitive system."));
        assert!(prompt.contains("- cognition-gate: gate role"));
        assert!(prompt.contains("- speak: speak role"));
        assert!(!prompt.contains("- sensory:"));
        // Peer list is sorted alphabetically.
        let gate_pos = prompt.find("cognition-gate").unwrap();
        let speak_pos = prompt.find("speak").unwrap();
        assert!(gate_pos < speak_pos);
    }

    #[test]
    fn empty_catalog_returns_base_unchanged() {
        let prompt = format_system_prompt("BASE", &[], &builtin::sensory(), &[], test_now());
        assert_eq!(prompt, "BASE");
    }

    #[test]
    fn solo_module_returns_base_unchanged() {
        let catalog = vec![(builtin::sensory(), "sensory role")];
        let prompt = format_system_prompt("BASE", &catalog, &builtin::sensory(), &[], test_now());
        assert_eq!(prompt, "BASE");
    }

    #[test]
    fn identity_memories_append_without_peer_catalog() {
        let memories = vec![IdentityMemoryRecord {
            index: nuillu_types::MemoryIndex::new("memory-1"),
            content: nuillu_types::MemoryContent::new("The agent is named Nuillu."),
            occurred_at: None,
        }];

        let prompt = format_system_prompt("BASE", &[], &builtin::sensory(), &memories, test_now());

        assert_eq!(
            prompt,
            "BASE\n\nIdentity memory loaded at agent startup:\n- [memory-1] The agent is named Nuillu.\n"
        );
    }

    #[test]
    fn identity_memory_can_include_temporal_context() {
        let memories = vec![IdentityMemoryRecord {
            index: nuillu_types::MemoryIndex::new("memory-1"),
            content: nuillu_types::MemoryContent::new("The agent met Koro."),
            occurred_at: Some(chrono::Utc.with_ymd_and_hms(2025, 5, 10, 0, 0, 0).unwrap()),
        }];

        let prompt = format_system_prompt("BASE", &[], &builtin::sensory(), &memories, test_now());

        assert!(prompt.contains("<one-year-ago occurred-at=\"2025-05-10T00:00:00Z\">"));
        assert!(prompt.contains("The agent met Koro."));
    }

    fn test_now() -> DateTime<Utc> {
        chrono::Utc.with_ymd_and_hms(2026, 5, 10, 0, 0, 0).unwrap()
    }
}
