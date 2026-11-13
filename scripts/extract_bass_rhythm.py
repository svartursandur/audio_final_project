"""Measure how the bass and the upper notes move against each other, as 24 numbers.

The idea: these forms have typical accompaniments that overall rhythm can miss.
  - A waltz puts a bass note on beat 1 and chords on beats 2 and 3 ("oom-pah-pah").
  - A march alternates bass and chord in twos.
  - A nocturne has a flowing left hand under a melody.
  - A fugue has independent voices, so there is no steady accompaniment pattern.

Steps for one 8-second window:
  1. Find note starts below 300 Hz (bass) and above it (upper notes) separately.
  2. Find the beat from all the notes together, as extract_rhythm.py does.
  3. Ask how strongly the bass repeats after 1 to 4 beats, and the same for the upper notes.
  4. Ask how strongly the upper notes follow the bass after 0, 1 and 2 beats.
  5. Add the share of note starts that are in the bass.
Each of those 12 values is averaged over the piece, and its spread is kept too.

Usage:
    /usr/bin/python3 scripts/extract_bass_rhythm.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA
from extract_mfcc import audio_paths, load_audio
from extract_rhythm import HOP_LENGTH, LONGEST_BEAT, SAMPLE_RATE, SHORTEST_BEAT, WINDOW_FRAMES, WINDOW_HOP, autocorrelation

FFT_SIZE = 1024
BASS_CUTOFF_HZ = 300
BASS_BINS = int(BASS_CUTOFF_HZ * FFT_SIZE / SAMPLE_RATE)  # frequency bins below the cutoff
REPEAT_BEATS = [1, 2, 3, 4]
FOLLOW_BEATS = [0, 1, 2]
WINDOW_VALUES = 2 * len(REPEAT_BEATS) + len(FOLLOW_BEATS) + 1
BASS_RHYTHM_SIZE = 2 * WINDOW_VALUES
OUTPUT_PATH = DATA / "features" / "bass_rhythm.npz"
SAVE_EVERY = 25  # files, so an interruption loses little work

SPECTROGRAM = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE, hop_length=HOP_LENGTH, power=2)


def onsets_by_register(waveform):
    """Return the energy increases in the bass, the upper notes and everything together."""
    with torch.no_grad():
        log_energy = torch.log1p(100 * SPECTROGRAM(waveform))
    increases = (log_energy[:, 1:] - log_energy[:, :-1]).clamp(min=0)
    return (increases[:BASS_BINS].mean(dim=0).numpy(),
            increases[BASS_BINS:].mean(dim=0).numpy(),
            increases.sum(dim=0).numpy())


def follow_strength(first, second):
    """How much `second` looks like `first` at each time shift, with `second` happening later."""
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    energy = np.sqrt((first_centered ** 2).sum() * (second_centered ** 2).sum())
    if energy == 0:
        return np.zeros(len(first))
    return np.correlate(second_centered, first_centered, mode="full")[len(first) - 1:] / energy


def peak_near(values, lag):
    """The largest value within one frame of a time shift, allowing for small timing differences."""
    lag_values = values[max(lag - 1, 0):lag + 2]
    return float(lag_values.max()) if len(lag_values) else 0.0


def window_features(bass, upper, everything):
    """The 12 values for one window, or None for a silent window."""
    overall = autocorrelation(everything)
    if overall is None:
        return None
    beat = SHORTEST_BEAT + int(np.argmax(overall[SHORTEST_BEAT:LONGEST_BEAT + 1]))
    empty = np.zeros(len(bass))
    bass_repeat = autocorrelation(bass)
    upper_repeat = autocorrelation(upper)
    bass_repeat = empty if bass_repeat is None else bass_repeat
    upper_repeat = empty if upper_repeat is None else upper_repeat
    upper_after_bass = follow_strength(bass, upper)

    values = [peak_near(bass_repeat, beat * count) for count in REPEAT_BEATS]
    values += [peak_near(upper_repeat, beat * count) for count in REPEAT_BEATS]
    values += [peak_near(upper_after_bass, beat * count) for count in FOLLOW_BEATS]
    values.append(bass.mean() / (bass.mean() + upper.mean() + 1e-6))
    return values


def bass_rhythm_summary(path):
    """Return 24 numbers for one audio file: the average and spread of the 12 window values."""
    bass, upper, everything = onsets_by_register(load_audio(path, sample_rate=SAMPLE_RATE))
    windows = []
    for start in range(0, max(len(everything) - WINDOW_FRAMES, 1), WINDOW_HOP):
        end = start + WINDOW_FRAMES
        values = window_features(bass[start:end], upper[start:end], everything[start:end])
        if values is not None:
            windows.append(values)
    if not windows:
        return np.zeros(BASS_RHYTHM_SIZE, dtype=np.float32)
    table = np.array(windows, dtype=np.float32)
    return np.concatenate([table.mean(axis=0), table.std(axis=0)])


def load_bass_rhythm():
    """Read saved features as {audio path: 24 numbers}."""
    if not OUTPUT_PATH.exists():
        return {}
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def save(features):
    np.savez(OUTPUT_PATH, paths=np.array(list(features)), features=np.stack(list(features.values())))


def main():
    torch.set_num_threads(2)  # leave the CPU free for other work
    features = load_bass_rhythm()
    todo = [path for path in audio_paths() if path not in features]
    print(f"{len(features)} cached, {len(todo)} to compute", flush=True)
    for index, path in enumerate(todo, 1):
        features[path] = bass_rhythm_summary(DATA / path)
        if index % SAVE_EVERY == 0:
            save(features)
            print(f"{index}/{len(todo)}", flush=True)
    save(features)
    print(f"saved {len(features)} files to {OUTPUT_PATH.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
