use async_trait::async_trait;
use nuillu_blackboard::ResourceAllocation;
use nuillu_types::ModuleInstanceId;
use serde::{Deserialize, Serialize};

use crate::ports::PortError;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedAllocationSnapshot {
    pub version: u32,
    pub owner: ModuleInstanceId,
    pub targets: ResourceAllocation,
    pub suppressions: ResourceAllocation,
}

impl PersistedAllocationSnapshot {
    pub const VERSION: u32 = 1;

    pub fn new(
        owner: ModuleInstanceId,
        targets: ResourceAllocation,
        suppressions: ResourceAllocation,
    ) -> Self {
        Self {
            version: Self::VERSION,
            owner,
            targets,
            suppressions,
        }
    }

    pub fn validate_version(&self) -> Result<(), PortError> {
        if self.version == Self::VERSION {
            Ok(())
        } else {
            Err(PortError::InvalidData(format!(
                "unsupported allocation snapshot version: {}",
                self.version
            )))
        }
    }
}

#[async_trait(?Send)]
pub trait AllocationStore {
    async fn load_all(&self) -> Result<Vec<PersistedAllocationSnapshot>, PortError>;

    async fn save(&self, snapshot: &PersistedAllocationSnapshot) -> Result<(), PortError>;
}

#[derive(Debug, Default)]
pub struct NoopAllocationStore;

#[async_trait(?Send)]
impl AllocationStore for NoopAllocationStore {
    async fn load_all(&self) -> Result<Vec<PersistedAllocationSnapshot>, PortError> {
        Ok(Vec::new())
    }

    async fn save(&self, _snapshot: &PersistedAllocationSnapshot) -> Result<(), PortError> {
        Ok(())
    }
}
