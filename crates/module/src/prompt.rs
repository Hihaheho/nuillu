use nuillu_types::ModuleId;

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
) -> String {
    let mut peers = catalog
        .iter()
        .filter(|(id, _)| id != owner)
        .map(|(id, role)| format!("- {}: {}", id, role))
        .collect::<Vec<_>>();
    if peers.is_empty() {
        return base.to_owned();
    }
    peers.sort();
    format!(
        "{base}\n\nYou are part of a cognitive system. Other modules in this brain:\n{peers}\n",
        base = base,
        peers = peers.join("\n"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuillu_types::builtin;

    #[test]
    fn excludes_owner_and_sorts_peers() {
        let catalog = vec![
            (builtin::sensory(), "sensory role"),
            (builtin::speak(), "speak role"),
            (builtin::attention_gate(), "gate role"),
        ];
        let prompt = format_system_prompt("BASE", &catalog, &builtin::sensory());
        assert!(prompt.starts_with("BASE\n\nYou are part of a cognitive system."));
        assert!(prompt.contains("- attention-gate: gate role"));
        assert!(prompt.contains("- speak: speak role"));
        assert!(!prompt.contains("- sensory:"));
        // Peer list is sorted alphabetically.
        let gate_pos = prompt.find("attention-gate").unwrap();
        let speak_pos = prompt.find("speak").unwrap();
        assert!(gate_pos < speak_pos);
    }

    #[test]
    fn empty_catalog_returns_base_unchanged() {
        let prompt = format_system_prompt("BASE", &[], &builtin::sensory());
        assert_eq!(prompt, "BASE");
    }

    #[test]
    fn solo_module_returns_base_unchanged() {
        let catalog = vec![(builtin::sensory(), "sensory role")];
        let prompt = format_system_prompt("BASE", &catalog, &builtin::sensory());
        assert_eq!(prompt, "BASE");
    }
}
