use nuillu_blackboard::CorePolicyRecord;
use nuillu_types::ModuleId;

/// Build a system prompt that prepends peer-context entries for every other
/// module registered in the agent. The owner module is excluded so each
/// module's prompt only lists its peers.
///
/// The peer context catalog is post-boot static state
/// (`Blackboard::peer_contexts`) so the produced string is stable across
/// activations and friendly to LLM prompt caching as long as `base` is also
/// stable.
pub fn format_system_prompt(
    base: &str,
    catalog: &[(ModuleId, &'static str)],
    owner: &ModuleId,
    core_policies: &[CorePolicyRecord],
) -> String {
    let peers = sorted_peer_lines(catalog, owner);
    let mut prompt = base.to_owned();
    append_peer_section(&mut prompt, &peers);
    append_policy_section(&mut prompt, !peers.is_empty(), core_policies);
    prompt
}

/// Build a system prompt with stable policy context but without the peer
/// context catalog. Use this when a module already receives the needed sibling
/// output through activation context, or when exposing internal module
/// structure encourages process-talk.
pub fn format_policy_system_prompt(base: &str, core_policies: &[CorePolicyRecord]) -> String {
    let mut prompt = base.to_owned();
    append_policy_section(&mut prompt, false, core_policies);
    prompt
}

/// Build the stable, cache-friendly system prompt for one faculty. This
/// contains peer-structure context only; dynamic memory and
/// blackboard state should be supplied through session seed or ephemeral
/// activation context.
pub fn format_faculty_system_prompt(
    base: &str,
    catalog: &[(ModuleId, &'static str)],
    owner: &ModuleId,
) -> String {
    let peers = sorted_peer_lines(catalog, owner);
    let mut prompt = base.to_owned();
    append_peer_section(&mut prompt, &peers);
    prompt
}

fn sorted_peer_lines(catalog: &[(ModuleId, &'static str)], owner: &ModuleId) -> Vec<String> {
    let mut peers = catalog
        .iter()
        .filter(|(id, _)| id != owner)
        .map(|(id, role)| format!("- {}: {}", id, role))
        .collect::<Vec<_>>();
    peers.sort();
    peers
}

fn append_peer_section(prompt: &mut String, peers: &[String]) {
    if peers.is_empty() {
        return;
    }
    prompt.push_str("\n\nYou are part of a cognitive system. Other modules in this brain:\n");
    prompt.push_str(&peers.join("\n"));
    prompt.push('\n');
}

fn append_policy_section(
    prompt: &mut String,
    follows_peer_section: bool,
    core_policies: &[CorePolicyRecord],
) {
    if !core_policies.is_empty() {
        prompt.push_str(if follows_peer_section { "\n" } else { "\n\n" });
        prompt.push_str("Core policies loaded at agent startup:\n");
        for policy in core_policies {
            prompt.push_str("- [");
            prompt.push_str(policy.index.as_str());
            prompt.push_str("] When ");
            prompt.push_str(policy.trigger.trim());
            prompt.push_str(", do: ");
            prompt.push_str(policy.behavior.trim());
            prompt.push('\n');
        }
    }
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
            (builtin::cognition_gate(), "gate role"),
        ];
        let prompt = format_system_prompt("BASE", &catalog, &builtin::sensory(), &[]);
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
        let prompt = format_system_prompt("BASE", &[], &builtin::sensory(), &[]);
        assert_eq!(prompt, "BASE");
    }

    #[test]
    fn solo_module_returns_base_unchanged() {
        let catalog = vec![(builtin::sensory(), "sensory role")];
        let prompt = format_system_prompt("BASE", &catalog, &builtin::sensory(), &[]);
        assert_eq!(prompt, "BASE");
    }

    #[test]
    fn core_policies_append_without_peer_catalog() {
        let policies = vec![CorePolicyRecord {
            index: nuillu_types::PolicyIndex::new("policy-1"),
            trigger: "the agent is greeted".to_owned(),
            behavior: "answer gently".to_owned(),
        }];

        let prompt = format_system_prompt("BASE", &[], &builtin::sensory(), &policies);

        assert_eq!(
            prompt,
            "BASE\n\nCore policies loaded at agent startup:\n- [policy-1] When the agent is greeted, do: answer gently\n"
        );
    }

    #[test]
    fn policy_system_prompt_omits_peer_catalog() {
        let policies = vec![CorePolicyRecord {
            index: nuillu_types::PolicyIndex::new("policy-1"),
            trigger: "the agent is greeted".to_owned(),
            behavior: "answer gently".to_owned(),
        }];
        let prompt = format_policy_system_prompt("BASE", &policies);

        assert_eq!(
            prompt,
            "BASE\n\nCore policies loaded at agent startup:\n- [policy-1] When the agent is greeted, do: answer gently\n"
        );
        assert!(!prompt.contains("You are part of a cognitive system"));
        assert!(!prompt.contains("Identity memory loaded at agent startup"));
    }
}
