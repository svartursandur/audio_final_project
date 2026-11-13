"""Make altered copies of each clip and describe them with the frozen AST.

AST never changes here. Altered copies give the small classifier more examples
to learn from. Two kinds of change, two copies of each:
  - Audio: shift the clip by up to 2 seconds, change the volume, and add quiet noise.
  - SpecAugment: blank out one band of frequencies and one stretch of time in the
    spectrogram, so the classifier can't depend on one small detail.

Only the 318 non-test pieces are processed, and only the embeddings are saved
(a few MB, no audio). Randomness is seeded from each file's name, so a rerun gives
the same copies.

Usage:
    /usr/bin/python3 scripts/extract_augmented.py
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, check_disk
from experiment_more_clips import load_existing as load_clips
from extract_ast import AST_SAMPLE_RATE, CLIP_POSITIONS, DEVICE, load_model, take_clips
from extract_mfcc import load_audio
from train_baseline import load_rows

COPIES = 2
AUDIO_VERSIONS = [0, 1]  # positions of the audio copies in each saved table
SPECTROGRAM_VERSIONS = [2, 3]  # positions of the SpecAugment copies

MAX_SHIFT_SECONDS = 2
MAX_GAIN_DB = 6
NOISE_SNR_RANGE_DB = (20, 40)  # higher means quieter noise

MAX_FREQUENCY_MASK = 48  # out of 128 frequency bands, the width used in the AST paper
MAX_TIME_MASK = 192  # out of 1024 time frames, also from the AST paper

OUTPUT_PATH = DATA / "features" / "ast_augmented.npz"
SAVE_EVERY = 25  # files, so an interruption loses little work
NUMBER_OF_THREADS = 8


def altered_audio_clips(waveform, rng):
    """Cut the three clips slightly off-centre, then change their volume and add noise."""
    shift = rng.integers(-MAX_SHIFT_SECONDS * AST_SAMPLE_RATE, MAX_SHIFT_SECONDS * AST_SAMPLE_RATE + 1)
    positions = [position + shift / len(waveform) for position in CLIP_POSITIONS]
    altered = []
    for clip in take_clips(waveform, positions):
        signal_level = np.sqrt(np.mean(clip ** 2))
        noise_level = signal_level / 10 ** (rng.uniform(*NOISE_SNR_RANGE_DB) / 20)
        gain = 10 ** (rng.uniform(-MAX_GAIN_DB, MAX_GAIN_DB) / 20)
        noisy = clip + rng.normal(0, noise_level, len(clip))
        altered.append(np.clip(noisy * gain, -1, 1).astype(np.float32))
    return altered


def mask_spectrograms(spectrograms, rng):
    """Blank one frequency band and one time stretch per clip. Zero is AST's average level."""
    masked = spectrograms.clone()
    frames, bands = masked.shape[1:]
    for one_clip in masked:
        width = rng.integers(1, MAX_FREQUENCY_MASK + 1)
        start = rng.integers(0, bands - width + 1)
        one_clip[:, start:start + width] = 0
        length = rng.integers(1, MAX_TIME_MASK + 1)
        start = rng.integers(0, frames - length + 1)
        one_clip[start:start + length, :] = 0
    return masked


def to_spectrograms(clips, feature_extractor):
    return feature_extractor(clips, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt").input_values


def embed(spectrograms, model):
    with torch.no_grad():
        return model(input_values=spectrograms.to(DEVICE)).pooler_output.cpu().numpy()


def augmented_table(path, feature_extractor, model, rng):
    """Return one file's embeddings as a table of 4 versions x 3 clips x 768 numbers."""
    waveform = load_audio(path, sample_rate=AST_SAMPLE_RATE)
    original = to_spectrograms(take_clips(waveform), feature_extractor)
    versions = [embed(to_spectrograms(altered_audio_clips(waveform, rng), feature_extractor), model)
                for _ in range(COPIES)]
    versions += [embed(mask_spectrograms(original, rng), model) for _ in range(COPIES)]
    return np.stack(versions)


def rng_for(path):
    return np.random.default_rng(zlib.crc32(path.encode()))


def load_augmented():
    """Read saved tables as {audio path: table}, or nothing if none exist yet."""
    if not OUTPUT_PATH.exists():
        return {}
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def save(tables):
    np.savez(OUTPUT_PATH, paths=np.array(list(tables)), features=np.stack(list(tables.values())))


def non_test_paths():
    """The files with cached clips, after confirming that no test piece is among them."""
    paths = list(load_clips())
    test_paths = {row["audio_path"] for row in load_rows("random.csv") if row["split"] == "test"}
    if test_paths & set(paths):
        raise RuntimeError("a test file would be augmented, stopping")
    return paths


def main():
    torch.set_num_threads(NUMBER_OF_THREADS)
    check_disk()
    feature_extractor, model = load_model()
    tables = load_augmented()
    todo = [path for path in non_test_paths() if path not in tables]
    print(f"{len(tables)} cached, {len(todo)} to compute", flush=True)
    for index, path in enumerate(todo, 1):
        tables[path] = augmented_table(DATA / path, feature_extractor, model, rng_for(path))
        if index % SAVE_EVERY == 0:
            check_disk()
            save(tables)
            print(f"{index}/{len(todo)}", flush=True)
    save(tables)
    print(f"saved {len(tables)} files to {OUTPUT_PATH.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
