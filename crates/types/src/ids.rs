use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Identifier for a module instance.
///
/// String-backed so that user-defined modules can be added without changing
/// the type system. Module ids are kebab-case: lowercase ASCII words separated
/// by single hyphens. The well-known cognitive modules from the design have
/// constants in [`builtin`] for ergonomic equality checks and routing.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct ModuleId(String);

impl ModuleId {
    pub fn new(name: impl Into<String>) -> Result<Self, ModuleIdParseError> {
        let name = name.into();
        if name.is_empty() {
            return Err(ModuleIdParseError::Empty);
        }
        if !is_kebab_case(&name) {
            return Err(ModuleIdParseError::InvalidChar);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable identifier for a reusable subsystem definition.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct SubsystemId(String);

impl SubsystemId {
    pub fn new(name: impl Into<String>) -> Result<Self, SubsystemIdParseError> {
        let name = name.into();
        if name.is_empty() {
            return Err(SubsystemIdParseError::Empty);
        }
        if !is_kebab_case(&name) {
            return Err(SubsystemIdParseError::InvalidChar);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SubsystemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identifier for a boot-time group of module roles.
///
/// Groups are host wiring metadata (for example, modules that the allocation
/// controller may target). They use the same stable kebab-case syntax as
/// [`ModuleId`].
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct ModuleGroupId(String);

impl ModuleGroupId {
    pub fn new(name: impl Into<String>) -> Result<Self, ModuleGroupIdParseError> {
        let name = name.into();
        if name.is_empty() {
            return Err(ModuleGroupIdParseError::Empty);
        }
        if !is_kebab_case(&name) {
            return Err(ModuleGroupIdParseError::InvalidChar);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModuleGroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn is_kebab_case(value: &str) -> bool {
    let bytes = value.as_bytes();
    if !bytes.first().is_some_and(|b| b.is_ascii_lowercase()) {
        return false;
    }

    let mut prev_hyphen = false;
    for &b in bytes {
        let valid = b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-';
        if !valid {
            return false;
        }
        if b == b'-' {
            if prev_hyphen {
                return false;
            }
            prev_hyphen = true;
        } else {
            prev_hyphen = false;
        }
    }
    !prev_hyphen
}

impl fmt::Display for ModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Zero-based index of one persistent replica for a module role.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct ReplicaIndex(u8);

impl ReplicaIndex {
    pub const ZERO: Self = Self(0);

    pub fn new(index: u8) -> Self {
        Self(index)
    }

    pub fn get(self) -> u8 {
        self.0
    }
}

impl fmt::Display for ReplicaIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One concrete replica of a subsystem definition in a scope path.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct SubsystemInstanceId {
    pub subsystem: SubsystemId,
    pub replica: ReplicaIndex,
}

impl SubsystemInstanceId {
    pub fn new(subsystem: SubsystemId, replica: ReplicaIndex) -> Self {
        Self { subsystem, replica }
    }
}

impl fmt::Display for SubsystemInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.subsystem, self.replica)
    }
}

/// Hierarchical runtime namespace containing module instances.
///
/// The empty path is the agent root. A path may contain any finite number of
/// subsystem instances; topology validation is responsible for rejecting
/// recursive subsystem definitions before expansion.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct ScopeId(Vec<SubsystemInstanceId>);

impl ScopeId {
    pub fn root() -> Self {
        Self::default()
    }

    pub fn from_path(path: Vec<SubsystemInstanceId>) -> Self {
        Self(path)
    }

    pub fn path(&self) -> &[SubsystemInstanceId] {
        &self.0
    }

    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    pub fn child(&self, child: SubsystemInstanceId) -> Self {
        let mut path = self.0.clone();
        path.push(child);
        Self(path)
    }

    pub fn parent(&self) -> Option<Self> {
        (!self.0.is_empty()).then(|| Self(self.0[..self.0.len() - 1].to_vec()))
    }
}

impl fmt::Display for ScopeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_root() {
            return f.write_str("/");
        }
        for instance in &self.0 {
            write!(f, "/{instance}")?;
        }
        Ok(())
    }
}

/// A module role resolved inside one runtime scope.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct ScopedModuleId {
    #[serde(default, skip_serializing_if = "ScopeId::is_root")]
    pub scope: ScopeId,
    pub module: ModuleId,
}

impl ScopedModuleId {
    pub fn new(scope: ScopeId, module: ModuleId) -> Self {
        Self { scope, module }
    }
}

impl fmt::Display for ScopedModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scope.is_root() {
            self.module.fmt(f)
        } else {
            write!(f, "{}/{}", self.scope, self.module)
        }
    }
}

/// One persistent module loop. Owner-stamped capabilities carry this value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ModuleInstanceId {
    #[serde(default, skip_serializing_if = "ScopeId::is_root")]
    pub scope: ScopeId,
    pub module: ModuleId,
    pub replica: ReplicaIndex,
}

impl ModuleInstanceId {
    pub fn new(module: ModuleId, replica: ReplicaIndex) -> Self {
        Self {
            scope: ScopeId::root(),
            module,
            replica,
        }
    }

    pub fn in_scope(scope: ScopeId, module: ModuleId, replica: ReplicaIndex) -> Self {
        Self {
            scope,
            module,
            replica,
        }
    }

    pub fn scoped_module(&self) -> ScopedModuleId {
        ScopedModuleId::new(self.scope.clone(), self.module.clone())
    }

    /// Stable module-key representation for storage schemas that keep module
    /// and replica in separate columns. Every owner, including root owners,
    /// carries an explicit self-describing JSON scope prefix.
    pub fn storage_module_key(&self) -> String {
        let scope = serde_json::to_string(&self.scope)
            .expect("ScopeId serialization is infallible for string-backed ids");
        format!("@{scope}:{}", self.module)
    }

    pub fn from_storage_parts(
        module_key: &str,
        replica: ReplicaIndex,
    ) -> Result<Self, ModuleInstanceStorageParseError> {
        let encoded = module_key
            .strip_prefix('@')
            .ok_or(ModuleInstanceStorageParseError::MissingScopePrefix)?;
        let (scope_json, module) = encoded
            .rsplit_once(':')
            .ok_or(ModuleInstanceStorageParseError::MissingModuleSeparator)?;
        let scope =
            serde_json::from_str(scope_json).map_err(ModuleInstanceStorageParseError::ScopeJson)?;
        Ok(Self::in_scope(scope, ModuleId::new(module)?, replica))
    }

    /// Stable owner key for a cognition entry stored in `log_scope`.
    ///
    /// Entries written to their owner's own scope use the explicitly scoped
    /// owner key. Cross-scope entries additionally encode the target log scope
    /// so persistence can restore both identities independently.
    pub fn cognition_storage_module_key(&self, log_scope: &ScopeId) -> String {
        if &self.scope == log_scope {
            return self.storage_module_key();
        }
        let scope = serde_json::to_string(log_scope).expect("ScopeId always serializes");
        format!("@cognition-scope:{scope}\n{}", self.storage_module_key())
    }

    pub fn from_cognition_storage_parts(
        module_key: &str,
        replica: ReplicaIndex,
    ) -> Result<(ScopeId, Self), ModuleInstanceStorageParseError> {
        let Some(encoded) = module_key.strip_prefix("@cognition-scope:") else {
            let owner = Self::from_storage_parts(module_key, replica)?;
            return Ok((owner.scope.clone(), owner));
        };
        let (scope_json, owner_key) = encoded
            .split_once('\n')
            .ok_or(ModuleInstanceStorageParseError::MissingModuleSeparator)?;
        let log_scope =
            serde_json::from_str(scope_json).map_err(ModuleInstanceStorageParseError::ScopeJson)?;
        let owner = Self::from_storage_parts(owner_key, replica)?;
        Ok((log_scope, owner))
    }

    /// Stable owner key for a memo stored in `memo_scope`.
    pub fn memo_storage_module_key(&self, memo_scope: &ScopeId) -> String {
        if &self.scope == memo_scope {
            return self.storage_module_key();
        }
        let scope = serde_json::to_string(memo_scope).expect("ScopeId always serializes");
        format!("@memo-scope:{scope}\n{}", self.storage_module_key())
    }

    pub fn from_memo_storage_parts(
        module_key: &str,
        replica: ReplicaIndex,
    ) -> Result<(ScopeId, Self), ModuleInstanceStorageParseError> {
        let Some(encoded) = module_key.strip_prefix("@memo-scope:") else {
            let owner = Self::from_storage_parts(module_key, replica)?;
            return Ok((owner.scope.clone(), owner));
        };
        let (scope_json, owner_key) = encoded
            .split_once('\n')
            .ok_or(ModuleInstanceStorageParseError::MissingModuleSeparator)?;
        let memo_scope =
            serde_json::from_str(scope_json).map_err(ModuleInstanceStorageParseError::ScopeJson)?;
        let owner = Self::from_storage_parts(owner_key, replica)?;
        Ok((memo_scope, owner))
    }
}

impl fmt::Display for ModuleInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scope.is_root() {
            if self.replica == ReplicaIndex::ZERO {
                self.module.fmt(f)
            } else {
                write!(f, "{}[{}]", self.module, self.replica)
            }
        } else if self.replica == ReplicaIndex::ZERO {
            write!(f, "{}/{}", self.scope, self.module)
        } else {
            write!(f, "{}/{}[{}]", self.scope, self.module, self.replica)
        }
    }
}

/// Monotonic runtime-local identifier for one module activation attempt.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct ModuleActivationId(u64);

impl ModuleActivationId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ModuleActivationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Boot-time policy limiting the replicas a module role may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReplicaCapRange {
    pub min: u8,
    pub max: u8,
}

impl ReplicaCapRange {
    pub const V1_MAX: u8 = 2;

    /// Construct a range of total active replicas. `new(0, 1)` means the
    /// module can be fully disabled or run one active replica.
    pub fn new(min: u8, max: u8) -> Result<Self, ReplicaCapRangeError> {
        if min > max {
            return Err(ReplicaCapRangeError::MinGreaterThanMax);
        }
        if max > Self::V1_MAX {
            return Err(ReplicaCapRangeError::AboveV1Max { max });
        }
        Ok(Self { min, max })
    }

    pub fn clamp(self, replicas: u8) -> u8 {
        replicas.clamp(self.min, self.max)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReplicaCapRangeError {
    #[error("replica cap range min must be <= max")]
    MinGreaterThanMax,
    #[error("replica cap range max {max} exceeds v1 limit")]
    AboveV1Max { max: u8 },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModuleIdParseError {
    #[error("module id must not be empty")]
    Empty,
    #[error("module id must be kebab-case: [a-z][a-z0-9]*(?:-[a-z0-9]+)*")]
    InvalidChar,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModuleGroupIdParseError {
    #[error("module group id must not be empty")]
    Empty,
    #[error("module group id must be kebab-case: [a-z][a-z0-9]*(?:-[a-z0-9]+)*")]
    InvalidChar,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SubsystemIdParseError {
    #[error("subsystem id must not be empty")]
    Empty,
    #[error("subsystem id must be kebab-case: [a-z][a-z0-9]*(?:-[a-z0-9]+)*")]
    InvalidChar,
}

#[derive(Debug, Error)]
pub enum ModuleInstanceStorageParseError {
    #[error(transparent)]
    ModuleId(#[from] ModuleIdParseError),
    #[error("module storage key is missing its explicit scope prefix")]
    MissingScopePrefix,
    #[error("scoped module storage key is missing its module separator")]
    MissingModuleSeparator,
    #[error("invalid scope JSON in module storage key: {0}")]
    ScopeJson(serde_json::Error),
}

/// Constructors for the cognitive modules defined in `attention-schema.md`.
///
/// These are *conventions*, not enum variants — the agent supports modules
/// outside this list. Use these so routing is consistent across the workspace.
pub mod builtin {
    use super::ModuleId;

    macro_rules! builtin {
        ($($fn_name:ident => $id:literal),* $(,)?) => {
            $(
                pub fn $fn_name() -> ModuleId {
                    ModuleId::new($id).expect("builtin id is valid")
                }
            )*
        };
    }

    builtin!(
        sensory               => "sensory",
        cognition_gate        => "cognition-gate",
        allocation => "allocation",
        action => "action",
        attention_schema      => "attention-schema",
        interpreter           => "interpreter",
        self_model            => "self-model",
        query_memory          => "query-memory",
        memory                => "memory",
        memory_compaction     => "memory-compaction",
        memory_association    => "memory-association",
        dreaming              => "dreaming",
        interoception        => "interoception",
        homeostasis => "homeostasis",
        policy                => "policy",
        policy_compaction     => "policy-compaction",
        reward                => "reward",
        predict               => "predict",
        surprise              => "surprise",
        subsystem_gate        => "subsystem-gate",
        speak_gate            => "speak-gate",
        speak                 => "speak",
        sleep                 => "sleep",
        poet                  => "poet",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty() {
        assert_eq!(ModuleId::new(""), Err(ModuleIdParseError::Empty));
    }

    #[test]
    fn rejects_uppercase() {
        assert_eq!(ModuleId::new("Foo"), Err(ModuleIdParseError::InvalidChar));
    }

    #[test]
    fn rejects_underscore() {
        assert_eq!(
            ModuleId::new("attention_schema"),
            Err(ModuleIdParseError::InvalidChar)
        );
    }

    #[test]
    fn rejects_bad_hyphens() {
        assert_eq!(
            ModuleId::new("-query"),
            Err(ModuleIdParseError::InvalidChar)
        );
        assert_eq!(
            ModuleId::new("query-"),
            Err(ModuleIdParseError::InvalidChar)
        );
        assert_eq!(
            ModuleId::new("query--memory"),
            Err(ModuleIdParseError::InvalidChar)
        );
    }

    #[test]
    fn accepts_kebab_case() {
        assert_eq!(ModuleId::new("query2").unwrap().as_str(), "query2");
        assert_eq!(
            ModuleId::new("attention-schema").unwrap().as_str(),
            "attention-schema"
        );
    }

    #[test]
    fn builtins_parse() {
        let _ = builtin::cognition_gate();
        let _ = builtin::allocation();
        let _ = builtin::attention_schema();
        let _ = builtin::self_model();
        let _ = builtin::query_memory();
    }

    #[test]
    fn replica_cap_range_validates_order_and_v1_limit() {
        assert_eq!(
            ReplicaCapRange::new(2, 1),
            Err(ReplicaCapRangeError::MinGreaterThanMax)
        );
        assert_eq!(
            ReplicaCapRange::new(0, 3),
            Err(ReplicaCapRangeError::AboveV1Max { max: 3 })
        );
        assert_eq!(
            ReplicaCapRange::new(0, 2).unwrap(),
            ReplicaCapRange { min: 0, max: 2 }
        );
        // Always-1-active is the typical default.
        assert_eq!(
            ReplicaCapRange::new(0, 0).unwrap(),
            ReplicaCapRange { min: 0, max: 0 }
        );
    }

    #[test]
    fn scoped_module_identity_round_trips_through_storage_key() {
        let scope = ScopeId::root()
            .child(SubsystemInstanceId::new(
                SubsystemId::new("arm").unwrap(),
                ReplicaIndex::new(3),
            ))
            .child(SubsystemInstanceId::new(
                SubsystemId::new("finger").unwrap(),
                ReplicaIndex::new(1),
            ));
        let owner =
            ModuleInstanceId::in_scope(scope.clone(), builtin::predict(), ReplicaIndex::new(1));

        assert_eq!(scope.parent().unwrap().to_string(), "/arm[3]");
        assert_eq!(owner.to_string(), "/arm[3]/finger[1]/predict[1]");
        assert_eq!(
            ModuleInstanceId::from_storage_parts(&owner.storage_module_key(), owner.replica,)
                .unwrap(),
            owner
        );
    }

    #[test]
    fn root_module_storage_key_has_an_explicit_scope() {
        let owner = ModuleInstanceId::new(builtin::memory(), ReplicaIndex::ZERO);

        assert_eq!(owner.storage_module_key(), "@[]:memory");
        assert_eq!(
            ModuleInstanceId::from_storage_parts(&owner.storage_module_key(), owner.replica)
                .unwrap(),
            owner
        );
        assert!(matches!(
            ModuleInstanceId::from_storage_parts("memory", ReplicaIndex::ZERO),
            Err(ModuleInstanceStorageParseError::MissingScopePrefix)
        ));
    }

    #[test]
    fn cross_scope_cognition_storage_round_trips_target_and_owner_scopes() {
        let inner_scope = ScopeId::root().child(SubsystemInstanceId::new(
            SubsystemId::new("arm").unwrap(),
            ReplicaIndex::new(2),
        ));
        let owner =
            ModuleInstanceId::in_scope(inner_scope, builtin::subsystem_gate(), ReplicaIndex::ZERO);
        let key = owner.cognition_storage_module_key(&ScopeId::root());

        assert_eq!(
            ModuleInstanceId::from_cognition_storage_parts(&key, ReplicaIndex::ZERO).unwrap(),
            (ScopeId::root(), owner)
        );
    }

    #[test]
    fn cross_scope_memo_storage_round_trips_target_and_owner_scopes() {
        let inner_scope = ScopeId::root().child(SubsystemInstanceId::new(
            SubsystemId::new("arm").unwrap(),
            ReplicaIndex::new(2),
        ));
        let owner =
            ModuleInstanceId::in_scope(inner_scope, builtin::subsystem_gate(), ReplicaIndex::ZERO);
        let key = owner.memo_storage_module_key(&ScopeId::root());

        assert_eq!(
            ModuleInstanceId::from_memo_storage_parts(&key, ReplicaIndex::ZERO).unwrap(),
            (ScopeId::root(), owner)
        );
    }
}
