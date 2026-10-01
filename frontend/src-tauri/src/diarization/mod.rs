//! Offline speaker diarization ("who spoke when").
//!
//! Runs after a meeting has been recorded and transcribed:
//! 1. Decode the meeting's audio file to 16 kHz mono.
//! 2. Compute a voiceprint (speaker embedding) for each transcript segment
//!    with a WeSpeaker ResNet34 ONNX model.
//! 3. Cluster the voiceprints (average-linkage, cosine distance).
//! 4. Store a speaker label ("speaker_1", "speaker_2", ...) on each transcript row.
//!
//! Display names live in the `meeting_speakers` table so users can rename
//! "Speaker 1" to a real name.

pub mod clustering;
pub mod commands;
pub mod embedding;
pub mod fbank;
pub mod model;

use crate::audio::decoder::decode_audio_file;
use crate::state::AppState;
use anyhow::{anyhow, Result};
use clustering::{ClusterTarget, DEFAULT_THRESHOLD, MIN_SPEAKER_SECONDS};
use embedding::SpeakerEmbedder;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter, Manager, Runtime};

static DIARIZATION_IN_PROGRESS: AtomicBool = AtomicBool::new(false);
static DIARIZATION_CANCELLED: AtomicBool = AtomicBool::new(false);

struct DiarizationGuard;

impl DiarizationGuard {
    fn acquire() -> Result<Self> {
        DIARIZATION_IN_PROGRESS
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| anyhow!("Speaker identification is already running"))?;
        DIARIZATION_CANCELLED.store(false, Ordering::SeqCst);
        Ok(Self)
    }
}

impl Drop for DiarizationGuard {
    fn drop(&mut self) {
        DIARIZATION_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}

pub fn is_in_progress() -> bool {
    DIARIZATION_IN_PROGRESS.load(Ordering::SeqCst)
}

pub fn cancel() {
    DIARIZATION_CANCELLED.store(true, Ordering::SeqCst);
}

fn check_cancelled() -> Result<()> {
    if DIARIZATION_CANCELLED.load(Ordering::SeqCst) {
        Err(anyhow!("Speaker identification cancelled"))
    } else {
        Ok(())
    }
}

/// Speaker id stored on transcript rows, e.g. "speaker_1".
pub fn speaker_id(label: usize) -> String {
    format!("speaker_{}", label + 1)
}

/// Default display name for a speaker id ("speaker_2" -> "Speaker 2").
pub fn default_speaker_name(speaker_id: &str) -> String {
    match speaker_id.strip_prefix("speaker_") {
        Some(n) => format!("Speaker {}", n),
        None => speaker_id.to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationProgress {
    pub meeting_id: String,
    /// "model", "decoding", "analyzing", "clustering", "saving", "complete"
    pub stage: String,
    pub progress_percentage: u32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationResult {
    pub meeting_id: String,
    pub num_speakers: usize,
    pub segments_labeled: usize,
}

fn emit_progress<R: Runtime>(app: &AppHandle<R>, meeting_id: &str, stage: &str, pct: u32, msg: &str) {
    let _ = app.emit(
        "diarization-progress",
        DiarizationProgress {
            meeting_id: meeting_id.to_string(),
            stage: stage.to_string(),
            progress_percentage: pct,
            message: msg.to_string(),
        },
    );
}

/// Run speaker identification for a meeting. Emits `diarization-progress`,
/// then `diarization-complete` or `diarization-error`.
pub async fn run<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
    num_speakers: Option<usize>,
) -> Result<DiarizationResult> {
    let _guard = DiarizationGuard::acquire()?;
    let result = run_inner(&app, &meeting_id, &meeting_folder_path, num_speakers).await;

    match &result {
        Ok(res) => {
            let _ = app.emit("diarization-complete", res.clone());
        }
        Err(e) => {
            let _ = app.emit(
                "diarization-error",
                serde_json::json!({ "meeting_id": meeting_id, "error": e.to_string() }),
            );
        }
    }
    result
}

async fn run_inner<R: Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
    meeting_folder_path: &str,
    num_speakers: Option<usize>,
) -> Result<DiarizationResult> {
    crate::ensure_onnx_runtime_available()?;

    // 1. Speaker model (downloaded on first use, ~26 MB)
    emit_progress(app, meeting_id, "model", 0, "Preparing speaker model...");
    let model_path = {
        let app_for_dl = app.clone();
        let mid = meeting_id.to_string();
        model::ensure_model(app, move |done, total| {
            let pct = if total > 0 { (done * 10 / total) as u32 } else { 0 };
            emit_progress(
                &app_for_dl,
                &mid,
                "model",
                pct,
                &format!("Downloading speaker model... {:.1} / {:.1} MB", done as f64 / 1e6, total as f64 / 1e6),
            );
        })
        .await?
    };
    check_cancelled()?;

    // 2. Transcript segments with timing
    let state = app
        .try_state::<AppState>()
        .ok_or_else(|| anyhow!("App state not available"))?;
    let pool = state.db_manager.pool().clone();

    let rows: Vec<(String, Option<f64>, Option<f64>)> = sqlx::query_as(
        "SELECT id, audio_start_time, audio_end_time FROM transcripts
         WHERE meeting_id = ? ORDER BY audio_start_time ASC",
    )
    .bind(meeting_id)
    .fetch_all(&pool)
    .await?;

    let segments: Vec<(String, f64, f64)> = rows
        .into_iter()
        .filter_map(|(id, s, e)| match (s, e) {
            (Some(s), Some(e)) if e > s => Some((id, s, e)),
            _ => None,
        })
        .collect();

    if segments.is_empty() {
        return Err(anyhow!(
            "This meeting's transcript has no audio timestamps, so speakers can't be matched to the recording"
        ));
    }

    // 3. Decode audio
    emit_progress(app, meeting_id, "decoding", 12, "Decoding audio...");
    let audio_path = find_audio_file(&PathBuf::from(meeting_folder_path))?;
    let audio = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
        Ok(decode_audio_file(&audio_path)?.to_whisper_format())
    })
    .await
    .map_err(|e| anyhow!("Decode task panicked: {}", e))??;
    check_cancelled()?;

    // 4. Embeddings + clustering (CPU-bound)
    emit_progress(app, meeting_id, "analyzing", 20, "Analyzing voices...");
    let app_for_embed = app.clone();
    let mid = meeting_id.to_string();
    let seg_times: Vec<(f64, f64)> = segments.iter().map(|(_, s, e)| (*s, *e)).collect();

    let labels = tokio::task::spawn_blocking(move || -> Result<Vec<usize>> {
        let mut embedder = SpeakerEmbedder::new(&model_path)?;
        let sr = fbank::SAMPLE_RATE as f64;
        let total = seg_times.len();

        let mut embedded_idx = Vec::new();
        let mut embeddings = Vec::new();
        let mut durations = Vec::new();

        for (i, (start, end)) in seg_times.iter().enumerate() {
            check_cancelled()?;
            let a = ((start * sr) as usize).min(audio.len());
            let b = ((end * sr) as usize).min(audio.len());
            if let Some(e) = embedder.embed(&audio[a..b])? {
                embedded_idx.push(i);
                embeddings.push(e);
                durations.push(end - start);
            }
            if i % 10 == 0 || i + 1 == total {
                let pct = 20 + (i as u32 + 1) * 70 / total as u32;
                emit_progress(
                    &app_for_embed,
                    &mid,
                    "analyzing",
                    pct,
                    &format!("Analyzing voices... {}/{} segments", i + 1, total),
                );
            }
        }

        if embeddings.is_empty() {
            return Err(anyhow!("Not enough speech in this recording to identify speakers"));
        }

        emit_progress(&app_for_embed, &mid, "clustering", 92, "Grouping voices...");
        let cluster_labels = match num_speakers {
            Some(k) if k > 0 => clustering::cluster(&embeddings, &ClusterTarget::NumSpeakers(k)),
            _ => {
                let raw = clustering::cluster(&embeddings, &ClusterTarget::Threshold(DEFAULT_THRESHOLD));
                clustering::merge_small_clusters(&embeddings, &raw, &durations, MIN_SPEAKER_SECONDS)
            }
        };

        // Segments too short to embed take the label of the nearest embedded segment in time
        let mut labels = vec![0usize; total];
        for (k, &i) in embedded_idx.iter().enumerate() {
            labels[i] = cluster_labels[k];
        }
        let embedded_set: std::collections::HashSet<usize> = embedded_idx.iter().copied().collect();
        for i in 0..total {
            if embedded_set.contains(&i) {
                continue;
            }
            let mid_t = (seg_times[i].0 + seg_times[i].1) / 2.0;
            if let Some(&nearest) = embedded_idx.iter().min_by(|&&x, &&y| {
                let dx = ((seg_times[x].0 + seg_times[x].1) / 2.0 - mid_t).abs();
                let dy = ((seg_times[y].0 + seg_times[y].1) / 2.0 - mid_t).abs();
                dx.partial_cmp(&dy).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                labels[i] = labels[nearest];
            }
        }
        Ok(labels)
    })
    .await
    .map_err(|e| anyhow!("Speaker analysis task panicked: {}", e))??;
    check_cancelled()?;

    // 5. Save
    emit_progress(app, meeting_id, "saving", 95, "Saving speaker labels...");
    let mut tx = pool.begin().await?;
    for ((id, _, _), label) in segments.iter().zip(&labels) {
        sqlx::query("UPDATE transcripts SET speaker = ? WHERE id = ?")
            .bind(speaker_id(*label))
            .bind(id)
            .execute(&mut *tx)
            .await?;
    }
    // Labels are renumbered on each run, so names from a previous run no longer apply
    sqlx::query("DELETE FROM meeting_speakers WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;

    let num = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    info!(
        "Speaker identification for meeting {}: {} speakers across {} segments",
        meeting_id,
        num,
        segments.len()
    );
    emit_progress(app, meeting_id, "complete", 100, "Done");

    Ok(DiarizationResult {
        meeting_id: meeting_id.to_string(),
        num_speakers: num,
        segments_labeled: segments.len(),
    })
}

fn find_audio_file(folder: &std::path::Path) -> Result<PathBuf> {
    crate::audio::retranscription::find_audio_file(folder)
}

/// Load display names for a meeting's speakers (speaker_id -> name).
pub async fn load_speaker_names(
    pool: &sqlx::SqlitePool,
    meeting_id: &str,
) -> HashMap<String, String> {
    match sqlx::query_as::<_, (String, String)>(
        "SELECT speaker_id, name FROM meeting_speakers WHERE meeting_id = ?",
    )
    .bind(meeting_id)
    .fetch_all(pool)
    .await
    {
        Ok(rows) => rows.into_iter().collect(),
        Err(e) => {
            warn!("Failed to load speaker names for {}: {}", meeting_id, e);
            HashMap::new()
        }
    }
}

/// Resolve the display name for a transcript's speaker id.
pub fn resolve_speaker_name(names: &HashMap<String, String>, speaker: &Option<String>) -> Option<String> {
    speaker.as_ref().map(|id| {
        names
            .get(id)
            .cloned()
            .unwrap_or_else(|| default_speaker_name(id))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_names() {
        assert_eq!(speaker_id(0), "speaker_1");
        assert_eq!(default_speaker_name("speaker_3"), "Speaker 3");
        let mut names = HashMap::new();
        names.insert("speaker_1".to_string(), "Alice".to_string());
        assert_eq!(resolve_speaker_name(&names, &Some("speaker_1".into())), Some("Alice".into()));
        assert_eq!(resolve_speaker_name(&names, &Some("speaker_2".into())), Some("Speaker 2".into()));
        assert_eq!(resolve_speaker_name(&names, &None), None);
    }

    /// End-to-end check against real audio. Skipped unless both env vars are set:
    /// DIARIZATION_TEST_MODEL=<path to wespeaker_en_voxceleb_resnet34.onnx>
    /// DIARIZATION_TEST_WAV=<path to sherpa-onnx's 0-four-speakers-zh.wav>
    #[test]
    fn four_speaker_recording() {
        let (Ok(model), Ok(wav)) = (
            std::env::var("DIARIZATION_TEST_MODEL"),
            std::env::var("DIARIZATION_TEST_WAV"),
        ) else {
            eprintln!("skipping: DIARIZATION_TEST_MODEL / DIARIZATION_TEST_WAV not set");
            return;
        };
        let audio = decode_audio_file(std::path::Path::new(&wav))
            .unwrap()
            .to_whisper_format();
        // Speech regions of the test file (seconds)
        let segs = [
            (0.63, 1.98), (2.34, 3.93), (4.44, 8.04), (8.55, 10.50), (11.49, 16.92),
            (22.17, 24.75), (27.69, 29.43), (30.03, 31.50), (33.72, 34.47), (34.80, 37.89),
            (48.09, 50.19), (52.62, 54.57),
        ];
        let mut embedder = SpeakerEmbedder::new(std::path::Path::new(&model)).unwrap();
        let embs: Vec<Vec<f32>> = segs
            .iter()
            .map(|(a, b)| {
                embedder
                    .embed(&audio[(a * 16000.0) as usize..(b * 16000.0) as usize])
                    .unwrap()
                    .unwrap()
            })
            .collect();
        let expected = vec![0, 0, 0, 1, 2, 0, 3, 3, 3, 3, 2, 0];
        assert_eq!(clustering::cluster(&embs, &ClusterTarget::NumSpeakers(4)), expected);
        assert_eq!(
            clustering::cluster(&embs, &ClusterTarget::Threshold(DEFAULT_THRESHOLD)),
            expected
        );
    }

    #[tokio::test]
    async fn speaker_names_table_roundtrip() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();

        sqlx::query("INSERT INTO meetings (id, title, created_at, updated_at) VALUES ('m1', 't', 'x', 'x')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(
            "INSERT INTO transcripts (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, speaker)
             VALUES ('t1', 'm1', 'hi', 'x', 0.0, 2.0, 'speaker_1')",
        )
        .execute(&pool)
        .await
        .unwrap();

        for name in ["Alice", "Alicia"] {
            sqlx::query(
                "INSERT INTO meeting_speakers (meeting_id, speaker_id, name) VALUES (?, ?, ?)
                 ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET name = excluded.name",
            )
            .bind("m1")
            .bind("speaker_1")
            .bind(name)
            .execute(&pool)
            .await
            .unwrap();
        }

        let names = load_speaker_names(&pool, "m1").await;
        assert_eq!(names.get("speaker_1").map(String::as_str), Some("Alicia"));

        let rows: Vec<(String, i64, Option<f64>)> = sqlx::query_as(
            "SELECT speaker, COUNT(*), SUM(COALESCE(audio_end_time - audio_start_time, 0))
             FROM transcripts
             WHERE meeting_id = ? AND speaker IS NOT NULL AND speaker LIKE 'speaker_%'
             GROUP BY speaker",
        )
        .bind("m1")
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(rows, vec![("speaker_1".to_string(), 1, Some(2.0))]);
    }
}
