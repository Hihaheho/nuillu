use nuillu_module::ports::PortError;

pub(crate) fn validate(
    label: &str,
    dimensions: usize,
    embedding: Vec<f32>,
) -> Result<Vec<f32>, PortError> {
    if embedding.len() != dimensions {
        return Err(PortError::InvalidInput(format!(
            "{label} embedding dimension mismatch: expected {dimensions}, got {}",
            embedding.len()
        )));
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err(PortError::InvalidData(format!(
            "{label} embedding contains NaN or infinity"
        )));
    }
    Ok(embedding)
}

pub(crate) fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        (dot / (left_norm * right_norm)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct TestEmbedder;

#[cfg(test)]
#[async_trait::async_trait(?Send)]
impl nuillu_module::ports::Embedder for TestEmbedder {
    fn dimensions(&self) -> usize {
        3
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError> {
        let text = text.to_ascii_lowercase();
        Ok(vec![
            f32::from(text.contains("alpha")),
            f32::from(text.contains("beta") || text.contains("unrelated")),
            f32::from(text.contains("launch") || text.contains("checklist")),
        ])
    }
}
