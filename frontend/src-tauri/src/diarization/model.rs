//! Download and locate the speaker-embedding model.

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use log::info;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tauri::{AppHandle, Manager, Runtime};
use tokio::io::AsyncWriteExt;

/// WeSpeaker ResNet34 trained on VoxCeleb (Apache-2.0), ONNX export from the
/// sherpa-onnx project. Input: 80-dim fbank `[B, T, 80]`, output: `[B, 256]`.
pub const MODEL_FILE: &str = "wespeaker_en_voxceleb_resnet34.onnx";
const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx";
const MODEL_SHA256: &str = "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94";
const MODEL_SIZE: u64 = 26_534_365;

pub fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?
        .join("models")
        .join("diarization"))
}

pub fn model_path<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf> {
    Ok(models_dir(app)?.join(MODEL_FILE))
}

pub fn is_model_ready<R: Runtime>(app: &AppHandle<R>) -> bool {
    model_path(app)
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len() == MODEL_SIZE)
        .unwrap_or(false)
}

/// Return the model path, downloading and verifying it first if needed.
/// `on_progress(downloaded_bytes, total_bytes)` is called during download.
pub async fn ensure_model<R: Runtime>(
    app: &AppHandle<R>,
    on_progress: impl Fn(u64, u64),
) -> Result<PathBuf> {
    let path = model_path(app)?;
    if is_model_ready(app) {
        return Ok(path);
    }

    let dir = models_dir(app)?;
    tokio::fs::create_dir_all(&dir).await?;
    let tmp = dir.join(format!("{}.part", MODEL_FILE));

    info!("Downloading speaker model from {}", MODEL_URL);
    let response = reqwest::Client::new()
        .get(MODEL_URL)
        .send()
        .await
        .context("Failed to download speaker model")?
        .error_for_status()
        .context("Failed to download speaker model")?;

    let total = response.content_length().unwrap_or(MODEL_SIZE);
    let mut file = tokio::fs::File::create(&tmp).await?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut last_report: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Speaker model download interrupted")?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if downloaded - last_report > 512 * 1024 || downloaded == total {
            on_progress(downloaded, total);
            last_report = downloaded;
        }
    }
    file.flush().await?;
    drop(file);

    let digest = format!("{:x}", hasher.finalize());
    if digest != MODEL_SHA256 {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(anyhow!("Speaker model checksum mismatch (got {})", digest));
    }

    tokio::fs::rename(&tmp, &path).await?;
    info!("Speaker model saved to {}", path.display());
    Ok(path)
}
