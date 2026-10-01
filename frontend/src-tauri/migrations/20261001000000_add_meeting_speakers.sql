-- Migration: Speaker diarization display names
-- transcripts.speaker now holds a diarization label ("speaker_1", "speaker_2", ...).
-- This table stores user-chosen display names for those labels, per meeting.

CREATE TABLE IF NOT EXISTS meeting_speakers (
    meeting_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (meeting_id, speaker_id),
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcripts_meeting_speaker ON transcripts(meeting_id, speaker);
