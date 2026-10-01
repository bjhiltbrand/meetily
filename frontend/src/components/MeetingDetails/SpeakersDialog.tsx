"use client";

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Users, Loader2, X, Wand2, Check } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { toast } from 'sonner';
import { speakerColorClass } from '@/lib/speakers';

interface SpeakersDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  meetingId: string;
  meetingFolderPath: string | null;
  /** Called after speakers are identified or renamed so the transcript can refresh */
  onChanged?: () => void | Promise<void>;
}

interface MeetingSpeaker {
  speaker_id: string;
  name: string | null;
  display_name: string;
  segment_count: number;
  total_seconds: number;
}

interface DiarizationProgress {
  meeting_id: string;
  stage: string;
  progress_percentage: number;
  message: string;
}

const SPEAKER_COUNT_OPTIONS = ['auto', '2', '3', '4', '5', '6', '7', '8', '9', '10'];

function formatDuration(seconds: number): string {
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
}

export function SpeakersDialog({
  open,
  onOpenChange,
  meetingId,
  meetingFolderPath,
  onChanged,
}: SpeakersDialogProps) {
  const [speakers, setSpeakers] = useState<MeetingSpeaker[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const [speakerCount, setSpeakerCount] = useState('auto');
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<DiarizationProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [showRerun, setShowRerun] = useState(false);

  const onChangedRef = useRef(onChanged);
  useEffect(() => { onChangedRef.current = onChanged; }, [onChanged]);

  const loadSpeakers = useCallback(async () => {
    try {
      const result = await invoke<MeetingSpeaker[]>('get_meeting_speakers_command', { meetingId });
      setSpeakers(result);
      setNames(Object.fromEntries(result.map(s => [s.speaker_id, s.name ?? ''])));
    } catch (e) {
      console.error('Failed to load speakers:', e);
    }
  }, [meetingId]);

  // Reset and load when opened
  useEffect(() => {
    if (!open) return;
    setError(null);
    setProgress(null);
    setShowRerun(false);
    loadSpeakers();
    invoke<boolean>('is_diarization_in_progress_command').then(setIsRunning).catch(() => {});
  }, [open, loadSpeakers]);

  // Listen for diarization events while open
  useEffect(() => {
    if (!open) return;
    const unlisteners: UnlistenFn[] = [];
    let cleanedUp = false;

    (async () => {
      const handlers = await Promise.all([
        listen<DiarizationProgress>('diarization-progress', (event) => {
          if (event.payload.meeting_id === meetingId) setProgress(event.payload);
        }),
        listen<{ meeting_id: string; num_speakers: number }>('diarization-complete', async (event) => {
          if (event.payload.meeting_id !== meetingId) return;
          setIsRunning(false);
          setProgress(null);
          setShowRerun(false);
          toast.success(`Found ${event.payload.num_speakers} speaker${event.payload.num_speakers === 1 ? '' : 's'}`);
          await loadSpeakers();
          await onChangedRef.current?.();
        }),
        listen<{ meeting_id: string; error: string }>('diarization-error', (event) => {
          if (event.payload.meeting_id !== meetingId) return;
          setIsRunning(false);
          setProgress(null);
          if (!event.payload.error.includes('cancelled')) setError(event.payload.error);
        }),
      ]);
      if (cleanedUp) handlers.forEach(u => u());
      else unlisteners.push(...handlers);
    })();

    return () => {
      cleanedUp = true;
      unlisteners.forEach(u => u());
    };
  }, [open, meetingId, loadSpeakers]);

  const handleIdentify = async () => {
    if (!meetingFolderPath) return;
    setError(null);
    setIsRunning(true);
    setProgress({ meeting_id: meetingId, stage: 'starting', progress_percentage: 0, message: 'Starting...' });
    try {
      await invoke('start_diarization_command', {
        meetingId,
        meetingFolderPath,
        numSpeakers: speakerCount === 'auto' ? null : parseInt(speakerCount, 10),
      });
    } catch (e) {
      setIsRunning(false);
      setProgress(null);
      setError(String(e));
    }
  };

  const handleCancel = async () => {
    try {
      await invoke('cancel_diarization_command');
    } catch (e) {
      console.error('Failed to cancel:', e);
    }
  };

  const hasNameChanges = speakers.some(s => (names[s.speaker_id] ?? '').trim() !== (s.name ?? ''));

  const handleSaveNames = async () => {
    setIsSaving(true);
    try {
      for (const s of speakers) {
        const newName = (names[s.speaker_id] ?? '').trim();
        if (newName !== (s.name ?? '')) {
          await invoke('rename_speaker_command', { meetingId, speakerId: s.speaker_id, name: newName });
        }
      }
      toast.success('Speaker names saved');
      await loadSpeakers();
      await onChangedRef.current?.();
    } catch (e) {
      toast.error(`Failed to save names: ${e}`);
    } finally {
      setIsSaving(false);
    }
  };

  const hasSpeakers = speakers.length > 0;
  const showSetup = !isRunning && (!hasSpeakers || showRerun);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Speakers
          </DialogTitle>
          <DialogDescription>
            {hasSpeakers && !showRerun
              ? 'Rename speakers to show real names in the transcript, copies and summaries.'
              : 'Identify who is speaking in this recording. This runs locally on your computer.'}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {showSetup && (
            <div className="space-y-2">
              <span className="text-sm font-medium">Number of speakers</span>
              <Select value={speakerCount} onValueChange={setSpeakerCount}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SPEAKER_COUNT_OPTIONS.map(v => (
                    <SelectItem key={v} value={v}>
                      {v === 'auto' ? 'Detect automatically' : `${v} speakers`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Choosing the exact number gives the best results. The first run downloads a 26 MB voice model.
              </p>
            </div>
          )}

          {isRunning && (
            <div className="space-y-2">
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${Math.min(progress?.progress_percentage ?? 0, 100)}%` }}
                />
              </div>
              <p className="text-sm text-muted-foreground text-center flex items-center justify-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                {progress?.message ?? 'Working...'}
              </p>
            </div>
          )}

          {!isRunning && hasSpeakers && !showRerun && (
            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {speakers.map(s => (
                <div key={s.speaker_id} className="flex items-center gap-3">
                  <span
                    className={`text-xs font-semibold px-2 py-1 rounded-md whitespace-nowrap ${speakerColorClass(s.speaker_id)}`}
                  >
                    {s.speaker_id.replace('speaker_', 'Speaker ')}
                  </span>
                  <Input
                    value={names[s.speaker_id] ?? ''}
                    placeholder={s.speaker_id.replace('speaker_', 'Speaker ')}
                    onChange={(e) => setNames(prev => ({ ...prev, [s.speaker_id]: e.target.value }))}
                    onKeyDown={(e) => { if (e.key === 'Enter' && hasNameChanges) handleSaveNames(); }}
                    className="h-8"
                  />
                  <span className="text-xs text-muted-foreground whitespace-nowrap w-20 text-right">
                    {formatDuration(s.total_seconds)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {!meetingFolderPath && (
            <p className="text-xs text-muted-foreground">
              This meeting has no saved recording, so speakers can't be identified.
            </p>
          )}
        </div>

        <DialogFooter className="gap-2">
          {isRunning && (
            <Button variant="outline" onClick={handleCancel}>
              <X className="h-4 w-4 mr-2" />
              Cancel
            </Button>
          )}

          {showSetup && (
            <>
              <Button
                variant="outline"
                onClick={() => (hasSpeakers ? setShowRerun(false) : onOpenChange(false))}
              >
                {hasSpeakers ? 'Back' : 'Cancel'}
              </Button>
              <Button
                onClick={handleIdentify}
                disabled={!meetingFolderPath}
                className="bg-blue-600 hover:bg-blue-700"
              >
                <Wand2 className="h-4 w-4 mr-2" />
                Identify speakers
              </Button>
            </>
          )}

          {!isRunning && hasSpeakers && !showRerun && (
            <>
              <Button variant="outline" onClick={() => setShowRerun(true)} disabled={!meetingFolderPath}>
                Re-run
              </Button>
              <Button onClick={handleSaveNames} disabled={!hasNameChanges || isSaving}>
                {isSaving ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Check className="h-4 w-4 mr-2" />}
                Save names
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
