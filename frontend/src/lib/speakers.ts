// Helpers for speaker diarization labels ("speaker_1", "speaker_2", ...)

// Full class strings so Tailwind keeps them (src/lib is in tailwind content paths)
const SPEAKER_COLORS = [
  'bg-blue-100 text-blue-800',
  'bg-emerald-100 text-emerald-800',
  'bg-amber-100 text-amber-800',
  'bg-purple-100 text-purple-800',
  'bg-rose-100 text-rose-800',
  'bg-cyan-100 text-cyan-800',
  'bg-lime-100 text-lime-800',
  'bg-orange-100 text-orange-800',
  'bg-indigo-100 text-indigo-800',
  'bg-teal-100 text-teal-800',
];

/** Stable color for a speaker id like "speaker_3". */
export function speakerColorClass(speakerId?: string | null): string {
  const n = speakerId ? parseInt(speakerId.replace(/^speaker_/, ''), 10) : NaN;
  if (Number.isNaN(n) || n < 1) return 'bg-gray-100 text-gray-700';
  return SPEAKER_COLORS[(n - 1) % SPEAKER_COLORS.length];
}

/** Prefix a transcript line with its speaker name, if diarization has run. */
export function withSpeaker(text: string, speakerName?: string | null): string {
  return speakerName ? `${speakerName}: ${text}` : text;
}
