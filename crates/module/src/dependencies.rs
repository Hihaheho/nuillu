use std::collections::HashSet;

use nuillu_types::{ModuleId, builtin};

use crate::ModuleRegistry;

/// Shared boot-time dependency policy for the built-in cognitive modules.
///
/// These edges make a dependent module flush active evidence-producing
/// dependencies before it activates. Missing modules are ignored so the same
/// policy can be used by full-agent, module-eval, and server registries.
pub fn apply_standard_dependencies<I>(registry: ModuleRegistry, present: I) -> ModuleRegistry
where
    I: IntoIterator<Item = ModuleId>,
{
    standard_dependency_edges_for(present)
        .into_iter()
        .fold(registry, |registry, (dependent, dependency)| {
            registry.depends_on(dependent, dependency)
        })
}

pub fn standard_dependency_edges() -> Vec<(ModuleId, ModuleId)> {
    vec![
        (builtin::self_model(), builtin::query_memory()),
        (builtin::surprise(), builtin::predict()),
        (builtin::cognition_gate(), builtin::sensory()),
        (builtin::cognition_gate(), builtin::query_memory()),
        (builtin::cognition_gate(), builtin::policy()),
        (builtin::cognition_gate(), builtin::self_model()),
        (builtin::cognition_gate(), builtin::surprise()),
        (builtin::speak(), builtin::query_memory()),
        (builtin::speak(), builtin::self_model()),
        (builtin::speak(), builtin::surprise()),
        (builtin::speak(), builtin::cognition_gate()),
        (builtin::reward(), builtin::policy()),
        (builtin::policy_compaction(), builtin::reward()),
        (builtin::memory_compaction(), builtin::memory_association()),
        (
            builtin::memory_recombination(),
            builtin::memory_compaction(),
        ),
        (
            builtin::memory_compaction(),
            builtin::homeostatic_controller(),
        ),
        (
            builtin::memory_association(),
            builtin::homeostatic_controller(),
        ),
        (
            builtin::memory_recombination(),
            builtin::homeostatic_controller(),
        ),
        (
            builtin::policy_compaction(),
            builtin::homeostatic_controller(),
        ),
    ]
}

fn standard_dependency_edges_for<I>(present: I) -> Vec<(ModuleId, ModuleId)>
where
    I: IntoIterator<Item = ModuleId>,
{
    let present = present.into_iter().collect::<HashSet<_>>();
    standard_dependency_edges()
        .into_iter()
        .filter(|(dependent, dependency)| {
            present.contains(dependent) && present.contains(dependency)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_dependencies_include_speech_settling_edges() {
        let edges = standard_dependency_edges();

        assert!(edges.contains(&(builtin::speak(), builtin::query_memory())));
        assert!(edges.contains(&(builtin::speak(), builtin::self_model())));
        assert!(edges.contains(&(builtin::speak(), builtin::surprise())));
        assert!(edges.contains(&(builtin::speak(), builtin::cognition_gate())));
    }

    #[test]
    fn standard_dependencies_ignore_absent_modules() {
        let edges = standard_dependency_edges_for([builtin::speak(), builtin::cognition_gate()]);

        assert_eq!(edges, vec![(builtin::speak(), builtin::cognition_gate())]);
    }

    #[test]
    fn standard_dependencies_do_not_include_cycles() {
        let edges = standard_dependency_edges();
        let mut deps_of = std::collections::HashMap::<ModuleId, Vec<ModuleId>>::new();

        for (dependent, dependency) in &edges {
            assert!(
                !edges.contains(&(dependency.clone(), dependent.clone())),
                "dependency pair should not create a direct cycle: {dependent} <-> {dependency}"
            );
            deps_of
                .entry(dependent.clone())
                .or_default()
                .push(dependency.clone());
            deps_of.entry(dependency.clone()).or_default();
        }

        let mut visiting = HashSet::<ModuleId>::new();
        let mut visited = HashSet::<ModuleId>::new();
        for module in deps_of.keys() {
            assert_acyclic(module, &deps_of, &mut visiting, &mut visited);
        }
    }

    fn assert_acyclic(
        module: &ModuleId,
        deps_of: &std::collections::HashMap<ModuleId, Vec<ModuleId>>,
        visiting: &mut HashSet<ModuleId>,
        visited: &mut HashSet<ModuleId>,
    ) {
        if visited.contains(module) {
            return;
        }
        assert!(
            visiting.insert(module.clone()),
            "standard dependency graph contains a cycle at {module}"
        );
        if let Some(deps) = deps_of.get(module) {
            for dep in deps {
                assert_acyclic(dep, deps_of, visiting, visited);
            }
        }
        visiting.remove(module);
        visited.insert(module.clone());
    }
}
