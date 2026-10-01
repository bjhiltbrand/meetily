//! Speaker-embedding extraction with a WeSpeaker ONNX model via ONNX Runtime.

use super::fbank::{apply_cmn, FbankComputer, NUM_MEL_BINS, SAMPLE_RATE};
use anyhow::{anyhow, Context, Result};
use ndarray::Array3;
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

/// Segments shorter than this are too short for a reliable voiceprint.
pub const MIN_EMBED_SECONDS: f64 = 0.5;
/// Long segments are embedded in windows of this length and averaged.
const WINDOW_SECONDS: usize = 10;

pub struct SpeakerEmbedder {
    session: Session,
    fbank: FbankComputer,
    input_name: String,
}

impl SpeakerEmbedder {
    pub fn new(model_path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load speaker model {}", model_path.display()))?;

        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow!("Speaker model has no inputs"))?;

        Ok(Self {
            session,
            fbank: FbankComputer::new(),
            input_name,
        })
    }

    fn embed_window(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        let mut feats = self.fbank.compute(samples);
        let num_frames = feats.len() / NUM_MEL_BINS;
        if num_frames == 0 {
            return Ok(None);
        }
        apply_cmn(&mut feats, num_frames);

        let input = Array3::from_shape_vec((1, num_frames, NUM_MEL_BINS), feats)?;
        let outputs = self
            .session
            .run(inputs![self.input_name.as_str() => TensorRef::from_array_view(input.view())?])?;
        let (_, output) = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow!("Speaker model returned no outputs"))?;
        let emb = output.try_extract_array::<f32>()?;
        Ok(Some(emb.iter().copied().collect()))
    }

    /// Compute an L2-normalized embedding for 16 kHz mono audio.
    /// Returns `None` if the audio is too short.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        if (samples.len() as f64) < MIN_EMBED_SECONDS * SAMPLE_RATE as f64 {
            return Ok(None);
        }

        let window = WINDOW_SECONDS * SAMPLE_RATE;
        let mut sum: Option<Vec<f32>> = None;
        let mut chunks: Vec<&[f32]> = samples.chunks(window).collect();
        // Fold a short tail into the previous window rather than embedding it alone
        if chunks.len() > 1
            && (chunks.last().unwrap().len() as f64) < MIN_EMBED_SECONDS * SAMPLE_RATE as f64 * 4.0
        {
            chunks.pop();
            let start = (chunks.len() - 1) * window;
            let last = chunks.len() - 1;
            chunks[last] = &samples[start..];
        }

        for chunk in chunks {
            if let Some(mut e) = self.embed_window(chunk)? {
                super::clustering::normalize(&mut e);
                match sum.as_mut() {
                    Some(s) => s.iter_mut().zip(&e).for_each(|(a, b)| *a += *b),
                    None => sum = Some(e),
                }
            }
        }

        Ok(sum.map(|mut s| {
            super::clustering::normalize(&mut s);
            s
        }))
    }
}
