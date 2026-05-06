//! Model2Vec-backed LanceDB embedder.
//!
//! This crate intentionally loads only local Model2Vec assets. Constructors
//! validate the local model files before calling into `model2vec-rs`, so this
//! adapter does not download models from Hugging Face at runtime.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use lutum_lancedb_core::LanceDbEmbedder;
use model2vec_rs::model::StaticModel;
use nuillu_module::ports::PortError;
use sha2::{Digest, Sha256};

pub const POTION_BASE_8M_EXPECTED_DIMENSIONS: usize = 256;
pub const POTION_BASE_8M_MODEL_SHA256: &str =
    "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2";

const DEFAULT_MAX_LENGTH: usize = 128;
const DEFAULT_BATCH_SIZE: usize = 1024;
const DIMENSION_PROBE_TEXT: &str = "dimension probe";
const REQUIRED_MODEL_FILES: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];

#[derive(Clone, Debug)]
pub struct PotionBase8MEmbedder {
    model: Arc<StaticModel>,
    dimensions: usize,
    max_length: Option<usize>,
    batch_size: usize,
}

impl PotionBase8MEmbedder {
    /// Load a local `minishlab/potion-base-8M` directory.
    ///
    /// `model_dir` must contain `config.json`, `model.safetensors`, and
    /// `tokenizer.json`.
    pub fn from_local_dir(model_dir: impl AsRef<Path>) -> Result<Self, PortError> {
        Self::from_local_dir_with_options(model_dir, Some(DEFAULT_MAX_LENGTH), DEFAULT_BATCH_SIZE)
    }

    /// Load a local model directory after verifying `model.safetensors`.
    pub fn from_verified_local_dir(model_dir: impl AsRef<Path>) -> Result<Self, PortError> {
        let model_dir = model_dir.as_ref();
        ensure_local_model_files(model_dir)?;
        verify_sha256(
            &model_dir.join("model.safetensors"),
            POTION_BASE_8M_MODEL_SHA256,
        )?;
        Self::from_local_dir(model_dir)
    }

    pub fn from_local_dir_with_options(
        model_dir: impl AsRef<Path>,
        max_length: Option<usize>,
        batch_size: usize,
    ) -> Result<Self, PortError> {
        if batch_size == 0 {
            return Err(PortError::InvalidInput(
                "model2vec batch_size must be greater than zero".into(),
            ));
        }
        if max_length == Some(0) {
            return Err(PortError::InvalidInput(
                "model2vec max_length must be greater than zero when set".into(),
            ));
        }

        let model_dir = model_dir.as_ref();
        ensure_local_model_files(model_dir)?;

        let model = StaticModel::from_pretrained(
            model_dir, None, // hf token
            None, // normalize override; None uses config.json
            None, // subfolder
        )
        .map_err(|error| {
            PortError::Backend(format!(
                "failed to load potion-base-8M model from {}: {error}",
                model_dir.display()
            ))
        })?;

        let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.encode_single(DIMENSION_PROBE_TEXT)
        }))
        .map_err(|_| PortError::Backend("model2vec dimension probe panicked".into()))?;
        let dimensions = probe.len();
        if dimensions == 0 {
            return Err(PortError::InvalidData(
                "potion-base-8M produced a zero-length embedding".into(),
            ));
        }
        if probe.iter().any(|value| !value.is_finite()) {
            return Err(PortError::InvalidData(
                "potion-base-8M dimension probe contains NaN or infinity".into(),
            ));
        }

        Ok(Self {
            model: Arc::new(model),
            dimensions,
            max_length,
            batch_size,
        })
    }

    /// Embed multiple documents with Model2Vec's native batch path.
    pub fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, PortError> {
        if texts.iter().any(|text| text.trim().is_empty()) {
            return Err(PortError::InvalidInput(
                "empty text cannot be embedded; skip empty documents before indexing".into(),
            ));
        }

        let vectors = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.model
                .encode_with_args(texts, self.max_length, self.batch_size)
        }))
        .map_err(|_| PortError::Backend("model2vec batch embedding panicked".into()))?;

        if vectors.len() != texts.len() {
            return Err(PortError::InvalidData(format!(
                "embedding count mismatch: expected {}, got {}",
                texts.len(),
                vectors.len()
            )));
        }

        for vector in &vectors {
            self.validate_vector(vector)?;
        }

        Ok(vectors)
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<(), PortError> {
        if vector.len() != self.dimensions {
            return Err(PortError::InvalidData(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }

        if vector.iter().any(|value| !value.is_finite()) {
            return Err(PortError::InvalidData(
                "embedding contains NaN or infinity".into(),
            ));
        }

        Ok(())
    }
}

#[async_trait(?Send)]
impl LanceDbEmbedder for PotionBase8MEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError> {
        if text.trim().is_empty() {
            return Err(PortError::InvalidInput(
                "empty text cannot be embedded".into(),
            ));
        }

        let vector = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.model.encode_single(text)
        }))
        .map_err(|_| PortError::Backend("model2vec embedding panicked".into()))?;
        self.validate_vector(&vector)?;

        Ok(vector)
    }
}

fn ensure_local_model_files(model_dir: &Path) -> Result<(), PortError> {
    if !model_dir.is_dir() {
        return Err(PortError::InvalidInput(format!(
            "model directory does not exist: {}",
            model_dir.display()
        )));
    }

    for file in REQUIRED_MODEL_FILES {
        let path = model_dir.join(file);
        if !path.is_file() {
            return Err(PortError::InvalidInput(format!(
                "model directory is missing required file: {}",
                path.display()
            )));
        }
    }

    Ok(())
}

pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), PortError> {
    let mut file = File::open(path).map_err(|error| {
        PortError::Backend(format!(
            "failed to open model file {}: {error}",
            path.display()
        ))
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 64];

    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            PortError::Backend(format!(
                "failed to read model file {}: {error}",
                path.display()
            ))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    let actual = format!("{:x}", hasher.finalize());
    if actual != expected_hex {
        return Err(PortError::InvalidData(format!(
            "model checksum mismatch: expected {expected_hex}, got {actual}"
        )));
    }

    Ok(())
}

pub fn default_model_dir() -> PathBuf {
    PathBuf::from("models/potion-base-8M")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> PathBuf {
        default_model_dir()
    }

    fn skip_without_model() -> bool {
        let dir = model_dir();
        !(dir.join("config.json").is_file()
            && dir.join("model.safetensors").is_file()
            && dir.join("tokenizer.json").is_file())
    }

    #[test]
    fn rejects_zero_batch_size() {
        let error = PotionBase8MEmbedder::from_local_dir_with_options(model_dir(), Some(128), 0)
            .unwrap_err();

        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[test]
    fn rejects_zero_max_length() {
        let error =
            PotionBase8MEmbedder::from_local_dir_with_options(model_dir(), Some(0), 1).unwrap_err();

        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[test]
    fn potion_base_8m_loads_and_reports_dimensions() {
        if skip_without_model() {
            return;
        }

        let embedder =
            PotionBase8MEmbedder::from_verified_local_dir(model_dir()).expect("model should load");

        assert_eq!(embedder.dimensions(), POTION_BASE_8M_EXPECTED_DIMENSIONS);
    }

    #[tokio::test]
    async fn potion_base_8m_embeds_short_text() {
        if skip_without_model() {
            return;
        }

        let embedder =
            PotionBase8MEmbedder::from_verified_local_dir(model_dir()).expect("model should load");

        let vector = embedder
            .embed("Rust vector search")
            .await
            .expect("embedding should succeed");

        assert_eq!(vector.len(), embedder.dimensions());
        assert!(vector.iter().all(|value| value.is_finite()));
    }

    #[tokio::test]
    async fn potion_base_8m_rejects_empty_text() {
        if skip_without_model() {
            return;
        }

        let embedder =
            PotionBase8MEmbedder::from_verified_local_dir(model_dir()).expect("model should load");

        let result = embedder.embed("   ").await;
        assert!(matches!(result, Err(PortError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn potion_base_8m_outputs_normalized_vector() {
        if skip_without_model() {
            return;
        }

        let embedder =
            PotionBase8MEmbedder::from_verified_local_dir(model_dir()).expect("model should load");

        let vector = embedder
            .embed("Rust vector search")
            .await
            .expect("embedding should succeed");

        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();

        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected L2-normalized embedding, got norm={norm}"
        );
    }
}
