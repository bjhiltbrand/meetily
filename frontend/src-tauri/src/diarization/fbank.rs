//! Kaldi-compatible log-mel filterbank features.
//!
//! Matches `torchaudio.compliance.kaldi.fbank` / `kaldi-native-fbank` with the
//! settings WeSpeaker uses to train its speaker-embedding models:
//! 25 ms Hamming window, 10 ms shift, 80 mel bins, pre-emphasis 0.97,
//! DC removal, no dither, `snip_edges = true`, power spectrum, log energies.

use realfft::RealFftPlanner;

pub const SAMPLE_RATE: usize = 16_000;
pub const NUM_MEL_BINS: usize = 80;
const FRAME_LENGTH: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const FFT_SIZE: usize = 512; // next power of two >= FRAME_LENGTH
const PREEMPH: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

pub struct FbankComputer {
    window: Vec<f32>,
    /// For each mel bin: (first fft bin index, weights)
    mel_banks: Vec<(usize, Vec<f32>)>,
    planner: RealFftPlanner<f32>,
}

impl Default for FbankComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl FbankComputer {
    pub fn new() -> Self {
        let window = (0..FRAME_LENGTH)
            .map(|i| {
                0.54 - 0.46
                    * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_LENGTH as f32 - 1.0)).cos()
            })
            .collect();

        let num_fft_bins = FFT_SIZE / 2;
        let fft_bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;
        let high_freq = SAMPLE_RATE as f32 / 2.0;
        let mel_low = mel_scale(LOW_FREQ);
        let mel_high = mel_scale(high_freq);
        let mel_delta = (mel_high - mel_low) / (NUM_MEL_BINS as f32 + 1.0);

        let mel_banks = (0..NUM_MEL_BINS)
            .map(|bin| {
                let left = mel_low + bin as f32 * mel_delta;
                let center = left + mel_delta;
                let right = center + mel_delta;
                let mut first: Option<usize> = None;
                let mut weights = Vec::new();
                for i in 0..num_fft_bins {
                    let mel = mel_scale(fft_bin_width * i as f32);
                    if mel > left && mel < right {
                        let w = if mel <= center {
                            (mel - left) / (center - left)
                        } else {
                            (right - mel) / (right - center)
                        };
                        if first.is_none() {
                            first = Some(i);
                        }
                        weights.push(w);
                    }
                }
                (first.unwrap_or(0), weights)
            })
            .collect();

        Self {
            window,
            mel_banks,
            planner: RealFftPlanner::new(),
        }
    }

    /// Number of frames produced for `num_samples` input samples.
    pub fn num_frames(num_samples: usize) -> usize {
        if num_samples < FRAME_LENGTH {
            0
        } else {
            1 + (num_samples - FRAME_LENGTH) / FRAME_SHIFT
        }
    }

    /// Compute log-mel fbank features for 16 kHz mono audio in [-1, 1].
    /// Samples are scaled to the int16 range first (WeSpeaker convention).
    /// Returns a row-major `[num_frames * NUM_MEL_BINS]` buffer.
    pub fn compute(&mut self, samples: &[f32]) -> Vec<f32> {
        let num_frames = Self::num_frames(samples.len());
        let mut out = Vec::with_capacity(num_frames * NUM_MEL_BINS);
        if num_frames == 0 {
            return out;
        }

        let fft = self.planner.plan_fft_forward(FFT_SIZE);
        let mut input = fft.make_input_vec();
        let mut spectrum = fft.make_output_vec();
        let mut frame = vec![0.0f32; FRAME_LENGTH];
        let mut power = vec![0.0f32; FFT_SIZE / 2 + 1];

        for f in 0..num_frames {
            let start = f * FRAME_SHIFT;
            for (dst, src) in frame.iter_mut().zip(&samples[start..start + FRAME_LENGTH]) {
                *dst = *src * 32768.0;
            }

            // Remove DC offset
            let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
            frame.iter_mut().for_each(|x| *x -= mean);

            // Pre-emphasis (Kaldi style, in place from the end)
            for i in (1..FRAME_LENGTH).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];

            // Window + zero pad
            for i in 0..FRAME_LENGTH {
                input[i] = frame[i] * self.window[i];
            }
            input[FRAME_LENGTH..].iter_mut().for_each(|x| *x = 0.0);

            fft.process(&mut input, &mut spectrum)
                .expect("FFT buffer sizes are fixed and valid");
            for (p, c) in power.iter_mut().zip(spectrum.iter()) {
                *p = c.norm_sqr();
            }

            for (first, weights) in &self.mel_banks {
                let energy: f32 = weights
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * power[first + k])
                    .sum();
                out.push(energy.max(f32::EPSILON).ln());
            }
        }

        out
    }
}

/// Subtract the per-dimension mean over time (utterance-level CMN).
pub fn apply_cmn(feats: &mut [f32], num_frames: usize) {
    if num_frames == 0 {
        return;
    }
    let mut mean = [0.0f32; NUM_MEL_BINS];
    for frame in feats.chunks_exact(NUM_MEL_BINS) {
        for (m, v) in mean.iter_mut().zip(frame) {
            *m += *v;
        }
    }
    mean.iter_mut().for_each(|m| *m /= num_frames as f32);
    for frame in feats.chunks_exact_mut(NUM_MEL_BINS) {
        for (v, m) in frame.iter_mut().zip(mean.iter()) {
            *v -= *m;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_kaldi_snip_edges() {
        assert_eq!(FbankComputer::num_frames(399), 0);
        assert_eq!(FbankComputer::num_frames(400), 1);
        assert_eq!(FbankComputer::num_frames(16_000), 98);
    }

    #[test]
    fn produces_finite_features() {
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16_000.0).sin() * 0.3)
            .collect();
        let mut fb = FbankComputer::new();
        let feats = fb.compute(&samples);
        assert_eq!(feats.len(), 98 * NUM_MEL_BINS);
        assert!(feats.iter().all(|v| v.is_finite()));
    }
}
