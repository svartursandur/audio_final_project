"""Compute one small MFCC summary per audio file and cache it to disk.

MFCCs describe the timbre of a short moment of sound. Each file gets 20 MFCCs
per frame, and we keep their average and their spread over the whole file, so
one piece becomes a list of just 40 numbers. Averaging throws away the order
of events, which is why this is a baseline and not the final model.
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA

SAMPLE_RATE = 32000
N_MFCC = 20
OUTPUT_PATH = DATA / "features" / "mfcc_summary.npz"
SAVE_EVERY = 25  # files, so an interruption loses little work

MFCC_TRANSFORM = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64},
)


def load_audio(path, sample_rate=SAMPLE_RATE):
    """Decode an audio file to a tensor of samples between -1 and 1, using ffmpeg."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", str(path),
               "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True, timeout=600)
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples.copy())


def summarize(path):
    """Return 40 numbers for one file: 20 MFCC means followed by 20 standard deviations."""
    waveform = load_audio(path)
    with torch.no_grad():
        mfcc = MFCC_TRANSFORM(waveform)  # shape: (20 coefficients, number of frames)
    return torch.cat([mfcc.mean(dim=1), mfcc.std(dim=1)]).numpy()


def audio_paths():
    """List every audio file used by the splits."""
    paths = []
    for name in ["random.csv", "real_test.csv"]:
        with (DATA / "splits" / name).open(encoding="utf-8", newline="") as handle:
            paths += [row["audio_path"] for row in csv.DictReader(handle)]
    return paths


def load_existing():
    """Read already computed features so that a restart can skip them."""
    if not OUTPUT_PATH.exists():
        return {}
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def save(features):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH, paths=np.array(list(features)), features=np.stack(list(features.values())))


def main():
    torch.set_num_threads(4)  # leave most of the CPU free for other work
    features = load_existing()
    todo = [path for path in audio_paths() if path not in features]
    print(f"{len(features)} cached, {len(todo)} to compute", flush=True)
    for index, path in enumerate(todo, 1):
        features[path] = summarize(DATA / path)
        if index % SAVE_EVERY == 0:
            save(features)
            print(f"{index}/{len(todo)}", flush=True)
    save(features)
    print(f"saved {len(features)} files to {OUTPUT_PATH.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
