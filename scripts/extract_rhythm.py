"""Measure the rhythm of each piece as 18 numbers.

The idea: find where notes start, then ask how strongly that pattern repeats after
1, 2, 3, or more beats. A waltz groups its beats in threes and a march in twos,
and timbre features like MFCCs and AST averages barely capture that.

Steps for one piece:
  1. Onset strength: how much the sound suddenly gets louder, moment by moment.
  2. Cut it into 8-second windows, because players speed up and slow down.
  3. Autocorrelation: how similar each window is to itself shifted in time.
  4. The beat is the strongest repeat between 0.3 and 1.0 seconds.
  5. Read off the autocorrelation at multiples of that beat.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA
from extract_mfcc import audio_paths, load_audio

SAMPLE_RATE = 16000
HOP_LENGTH = 256
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH
WINDOW_FRAMES = int(8 * FRAMES_PER_SECOND)
WINDOW_HOP = WINDOW_FRAMES // 2
SHORTEST_BEAT = int(0.3 * FRAMES_PER_SECOND)
LONGEST_BEAT = int(1.0 * FRAMES_PER_SECOND)
BEAT_MULTIPLES = [0.5, 1, 1.5, 2, 3, 4, 6]  # how many beats later to compare
OUTPUT_PATH = DATA / "features" / "rhythm.npz"
SAVE_EVERY = 25  # files, so an interruption loses little work

MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=40)


def onset_strength(waveform):
    """Add up the increases in energy across frequency bands, frame by frame."""
    with torch.no_grad():
        log_energy = torch.log1p(100 * MEL_SPECTROGRAM(waveform))
    increases = (log_energy[:, 1:] - log_energy[:, :-1]).clamp(min=0)
    return increases.sum(dim=0).numpy()


def autocorrelation(window):
    """Similarity of a window to itself at each time shift. Shift 0 is scaled to 1."""
    centered = window - window.mean()
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    return correlation / correlation[0] if correlation[0] > 0 else None


def window_features(window):
    """Tempo plus repeat strength at each beat multiple, or None for a silent window."""
    correlation = autocorrelation(window)
    if correlation is None:
        return None
    beat = SHORTEST_BEAT + int(np.argmax(correlation[SHORTEST_BEAT:LONGEST_BEAT + 1]))
    features = [60 * FRAMES_PER_SECOND / beat]  # tempo in beats per minute
    for multiple in BEAT_MULTIPLES:
        lag = int(round(beat * multiple))
        features.append(float(correlation[lag - 1:lag + 2].max()) if lag + 2 < len(correlation) else 0.0)
    return features


def rhythm_summary(path):
    """Return 18 numbers: the average and spread of 8 window values, plus 2 overall values."""
    onsets = onset_strength(load_audio(path, sample_rate=SAMPLE_RATE))
    windows = []
    for start in range(0, max(len(onsets) - WINDOW_FRAMES, 1), WINDOW_HOP):
        features = window_features(onsets[start:start + WINDOW_FRAMES])
        if features is not None:
            windows.append(features)
    if not windows:
        return np.zeros(18, dtype=np.float32)
    table = np.array(windows, dtype=np.float32)
    overall = [onsets.mean(), onsets.std() / (onsets.mean() + 1e-6)]  # note density and how spiky it is
    return np.concatenate([table.mean(axis=0), table.std(axis=0), np.array(overall, dtype=np.float32)])


def load_existing():
    """Read already computed features so that a restart can skip them."""
    if not OUTPUT_PATH.exists():
        return {}
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def save(features):
    np.savez(OUTPUT_PATH, paths=np.array(list(features)), features=np.stack(list(features.values())))


def main():
    torch.set_num_threads(4)  # leave most of the CPU free for other work
    features = load_existing()
    todo = [path for path in audio_paths() if path not in features]
    print(f"{len(features)} cached, {len(todo)} to compute", flush=True)
    for index, path in enumerate(todo, 1):
        features[path] = rhythm_summary(DATA / path)
        if index % SAVE_EVERY == 0:
            save(features)
            print(f"{index}/{len(todo)}", flush=True)
    save(features)
    print(f"saved {len(features)} files to {OUTPUT_PATH.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
