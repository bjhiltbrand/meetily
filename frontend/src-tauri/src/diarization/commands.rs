//! Tauri commands for speaker identification.

use super::{default_speaker_name, load_speaker_names};
use crate::state::AppState;
use log::error;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Runtime};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSpeaker {
    pub speaker_id: String,
    /// Custom name, if the user set one
    pub name: Option<String>,
    /// Name to show (custom name or "Speaker N")
    pub display_name: String,
    pub segment_count: i64,
    pub total_seconds: f64,
}

/// Start speaker identification in the background.
/// `num_speakers`: `None`/0 = detect automatically.
#[tauri::command]
pub async fn start_diarization_command<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
    num_speakers: Option<usize>,
) -> Result<(), String> {
    if super::is_in_progress() {
        return Err("Speaker identification is already running".to_string());
    }
    tauri::async_runtime::spawn(async move {
        if let Err(e) = super::run(app, meeting_id, meeting_folder_path, num_speakers).await {
            error!("Speaker identification failed: {}", e);
        }
    });
    Ok(())
}

#[tauri::command]
pub async fn cancel_diarization_command() -> Result<(), String> {
    super::cancel();
    Ok(())
}

#[tauri::command]
pub async fn is_diarization_in_progress_command() -> bool {
    super::is_in_progress()
}

/// List the speakers identified in a meeting, with talk time.
#[tauri::command]
pub async fn get_meeting_speakers_command(
    meeting_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<MeetingSpeaker>, String> {
    let pool = state.db_manager.pool();
    let rows: Vec<(String, i64, Option<f64>)> = sqlx::query_as(
        "SELECT speaker, COUNT(*), SUM(COALESCE(audio_end_time - audio_start_time, 0))
         FROM transcripts
         WHERE meeting_id = ? AND speaker IS NOT NULL AND speaker LIKE 'speaker_%'
         GROUP BY speaker",
    )
    .bind(&meeting_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to load speakers: {}", e))?;

    let names = load_speaker_names(pool, &meeting_id).await;
    let mut speakers: Vec<MeetingSpeaker> = rows
        .into_iter()
        .map(|(speaker_id, count, secs)| {
            let name = names.get(&speaker_id).cloned();
            MeetingSpeaker {
                display_name: name.clone().unwrap_or_else(|| default_speaker_name(&speaker_id)),
                name,
                speaker_id,
                segment_count: count,
                total_seconds: secs.unwrap_or(0.0),
            }
        })
        .collect();

    // Natural order: speaker_2 before speaker_10
    speakers.sort_by_key(|s| {
        s.speaker_id
            .strip_prefix("speaker_")
            .and_then(|n| n.parse::<u32>().ok())
            .unwrap_or(u32::MAX)
    });
    Ok(speakers)
}

/// Rename a speaker for one meeting. An empty name restores the default.
#[tauri::command]
pub async fn rename_speaker_command(
    meeting_id: String,
    speaker_id: String,
    name: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let pool = state.db_manager.pool();
    let name = name.trim();
    let result = if name.is_empty() {
        sqlx::query("DELETE FROM meeting_speakers WHERE meeting_id = ? AND speaker_id = ?")
            .bind(&meeting_id)
            .bind(&speaker_id)
            .execute(pool)
            .await
    } else {
        sqlx::query(
            "INSERT INTO meeting_speakers (meeting_id, speaker_id, name) VALUES (?, ?, ?)
             ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET name = excluded.name",
        )
        .bind(&meeting_id)
        .bind(&speaker_id)
        .bind(name)
        .execute(pool)
        .await
    };
    result
        .map(|_| ())
        .map_err(|e| format!("Failed to rename speaker: {}", e))
}

/// Reassign a single transcript segment to a different speaker
/// (for fixing mistakes by hand).
#[tauri::command]
pub async fn set_transcript_speaker_command(
    transcript_id: String,
    speaker_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if !speaker_id.starts_with("speaker_") {
        return Err("Invalid speaker id".to_string());
    }
    sqlx::query("UPDATE transcripts SET speaker = ? WHERE id = ?")
        .bind(&speaker_id)
        .bind(&transcript_id)
        .execute(state.db_manager.pool())
        .await
        .map(|_| ())
        .map_err(|e| format!("Failed to update speaker: {}", e))
}

/// Whether the speaker model is already downloaded.
#[tauri::command]
pub async fn is_diarization_model_ready_command<R: Runtime>(app: AppHandle<R>) -> bool {
    super::model::is_model_ready(&app)
}
