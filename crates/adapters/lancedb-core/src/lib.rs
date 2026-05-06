//! Shared LanceDB adapter boundary traits.

use async_trait::async_trait;
use nuillu_module::ports::PortError;

#[async_trait(?Send)]
pub trait LanceDbEmbedder {
    fn dimensions(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError>;
}
