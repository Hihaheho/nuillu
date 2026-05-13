//! OpenAI-compatible text embedding adapter.
//!
//! Talks to any HTTP endpoint exposing the OpenAI `/v1/embeddings` API shape
//! (`{ model, input, dimensions, encoding_format }`). The adapter enforces the
//! caller-configured `target_dimensions` two ways:
//!
//! 1. It asks the server for that exact dimensionality via the `dimensions`
//!    request parameter. OpenAI's v3 embedding models are Matryoshka-trained
//!    and re-normalize the truncated vector server-side.
//! 2. After receiving the response it trims any extra coordinates and
//!    re-applies L2 normalization. This is the safety net for providers that
//!    silently ignore `dimensions` or for models that aren't Matryoshka.
//!
//! Either layer alone would suffice for OpenAI's own endpoint, but doing both
//! lets the rest of the system (libsql vector columns, `EmbeddingProfile`)
//! trust the dimensionality reported by `Embedder::dimensions()`.

use std::time::Duration;

use async_trait::async_trait;
use nuillu_module::ports::{Embedder, PortError};
use serde::{Deserialize, Serialize};

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const EMBEDDINGS_PATH: &str = "embeddings";
const MAX_RESPONSE_BODY_EXCERPT: usize = 512;

#[derive(Clone, Debug)]
pub struct OpenAiEmbedderConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub target_dimensions: usize,
    pub request_timeout: Option<Duration>,
}

#[derive(Clone, Debug)]
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    target_dimensions: usize,
}

impl OpenAiEmbedder {
    pub fn new(config: OpenAiEmbedderConfig) -> Result<Self, PortError> {
        let base_url = config.base_url.trim();
        if base_url.is_empty() {
            return Err(PortError::InvalidInput(
                "openai embedding base_url must not be empty".into(),
            ));
        }
        if config.api_key.trim().is_empty() {
            return Err(PortError::InvalidInput(
                "openai embedding api_key must not be empty".into(),
            ));
        }
        if config.model.trim().is_empty() {
            return Err(PortError::InvalidInput(
                "openai embedding model must not be empty".into(),
            ));
        }
        if config.target_dimensions == 0 {
            return Err(PortError::InvalidInput(
                "openai embedding target_dimensions must be greater than zero".into(),
            ));
        }

        let endpoint = build_endpoint(base_url);
        let timeout = config.request_timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT);
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|error| {
                PortError::Backend(format!("failed to build reqwest client: {error}"))
            })?;

        Ok(Self {
            client,
            endpoint,
            api_key: config.api_key,
            model: config.model,
            target_dimensions: config.target_dimensions,
        })
    }
}

#[async_trait(?Send)]
impl Embedder for OpenAiEmbedder {
    fn dimensions(&self) -> usize {
        self.target_dimensions
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError> {
        if text.trim().is_empty() {
            return Err(PortError::InvalidInput(
                "empty text cannot be embedded".into(),
            ));
        }

        let request = EmbeddingsRequest {
            model: &self.model,
            input: text,
            dimensions: self.target_dimensions,
            encoding_format: "float",
        };

        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|error| {
                PortError::Backend(format!(
                    "openai embedding request to {} failed: {error}",
                    self.endpoint
                ))
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(PortError::Backend(format!(
                "openai embedding request returned {status}: {}",
                excerpt(&body)
            )));
        }

        let body: EmbeddingsResponse = response.json().await.map_err(|error| {
            PortError::InvalidData(format!("failed to decode openai embedding response: {error}"))
        })?;

        let raw = body.data.into_iter().next().ok_or_else(|| {
            PortError::InvalidData("openai embedding response contained no data".into())
        })?;

        trim_and_renormalize(raw.embedding, self.target_dimensions)
    }
}

fn build_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    format!("{trimmed}/{EMBEDDINGS_PATH}")
}

fn excerpt(body: &str) -> String {
    if body.len() <= MAX_RESPONSE_BODY_EXCERPT {
        body.to_string()
    } else {
        let mut end = MAX_RESPONSE_BODY_EXCERPT;
        while end > 0 && !body.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &body[..end])
    }
}

pub fn trim_and_renormalize(
    mut vector: Vec<f32>,
    target_dimensions: usize,
) -> Result<Vec<f32>, PortError> {
    if target_dimensions == 0 {
        return Err(PortError::InvalidInput(
            "target_dimensions must be greater than zero".into(),
        ));
    }
    if vector.len() < target_dimensions {
        return Err(PortError::InvalidData(format!(
            "embedding shorter than configured dimensions: got {}, expected at least {}",
            vector.len(),
            target_dimensions
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(PortError::InvalidData(
            "embedding contains NaN or infinity".into(),
        ));
    }

    vector.truncate(target_dimensions);

    let norm_sq: f32 = vector.iter().map(|value| value * value).sum();
    if !norm_sq.is_finite() || norm_sq <= 0.0 {
        return Err(PortError::InvalidData(
            "trimmed embedding has zero or non-finite norm".into(),
        ));
    }
    let norm = norm_sq.sqrt();
    for value in vector.iter_mut() {
        *value /= norm;
    }

    Ok(vector)
}

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a str,
    dimensions: usize,
    encoding_format: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> OpenAiEmbedderConfig {
        OpenAiEmbedderConfig {
            base_url: "https://api.openai.com/v1".into(),
            api_key: "sk-test".into(),
            model: "text-embedding-3-small".into(),
            target_dimensions: 512,
            request_timeout: None,
        }
    }

    #[test]
    fn rejects_empty_base_url() {
        let mut config = sample_config();
        config.base_url = "   ".into();
        let error = OpenAiEmbedder::new(config).unwrap_err();
        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[test]
    fn rejects_empty_api_key() {
        let mut config = sample_config();
        config.api_key = "".into();
        let error = OpenAiEmbedder::new(config).unwrap_err();
        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[test]
    fn rejects_empty_model() {
        let mut config = sample_config();
        config.model = "".into();
        let error = OpenAiEmbedder::new(config).unwrap_err();
        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[test]
    fn rejects_zero_target_dimensions() {
        let mut config = sample_config();
        config.target_dimensions = 0;
        let error = OpenAiEmbedder::new(config).unwrap_err();
        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[test]
    fn dimensions_returns_configured_value() {
        let embedder = OpenAiEmbedder::new(sample_config()).expect("config is valid");
        assert_eq!(embedder.dimensions(), 512);
    }

    #[tokio::test]
    async fn embed_rejects_empty_input() {
        let embedder = OpenAiEmbedder::new(sample_config()).expect("config is valid");
        let error = embedder.embed("   ").await.unwrap_err();
        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[test]
    fn build_endpoint_handles_trailing_slash() {
        assert_eq!(
            build_endpoint("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/embeddings"
        );
        assert_eq!(
            build_endpoint("https://api.openai.com/v1"),
            "https://api.openai.com/v1/embeddings"
        );
    }

    #[test]
    fn trim_and_renormalize_truncates_and_normalizes() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let trimmed = trim_and_renormalize(input, 4).unwrap();

        assert_eq!(trimmed.len(), 4);

        let expected_norm: f32 = (1.0_f32 + 4.0 + 9.0 + 16.0).sqrt();
        let expected: Vec<f32> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| v / expected_norm)
            .collect();
        for (got, want) in trimmed.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
        let norm: f32 = trimmed.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trim_and_renormalize_rejects_zero_target() {
        let error = trim_and_renormalize(vec![1.0, 2.0], 0).unwrap_err();
        assert!(matches!(error, PortError::InvalidInput(_)));
    }

    #[test]
    fn trim_and_renormalize_rejects_short_vector() {
        let error = trim_and_renormalize(vec![1.0, 2.0], 4).unwrap_err();
        assert!(matches!(error, PortError::InvalidData(_)));
    }

    #[test]
    fn trim_and_renormalize_rejects_non_finite() {
        let error = trim_and_renormalize(vec![1.0, f32::NAN, 2.0, 3.0], 2).unwrap_err();
        assert!(matches!(error, PortError::InvalidData(_)));
    }

    #[test]
    fn trim_and_renormalize_rejects_zero_norm() {
        let error = trim_and_renormalize(vec![0.0, 0.0, 0.0, 0.0], 2).unwrap_err();
        assert!(matches!(error, PortError::InvalidData(_)));
    }
}
