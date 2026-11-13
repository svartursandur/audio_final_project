"""Turn each audio file into one embedding from the pretrained AST model.

AST (Audio Spectrogram Transformer) was trained on general sounds from AudioSet.
Here it is frozen: we only use it to describe the audio as 768 numbers, and a
simple classifier is trained on top in train_probe.py. That is a "linear probe".

AST takes 10 seconds of 16 kHz audio. A piece is longer than that, so we take
three clips from different parts of it and average their embeddings.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA
from extract_mfcc import audio_paths, load_audio

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
# Use the graphics card when there is one (the GPU server), otherwise the CPU (this laptop).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AST_SAMPLE_RATE = 16000
CLIP_SAMPLES = 10 * AST_SAMPLE_RATE
CLIP_POSITIONS = [0.25, 0.50, 0.75]  # where in the piece each clip is centred
OUTPUT_PATH = DATA / "features" / "ast_embeddings.npz"
SAVE_EVERY = 25  # files, so an interruption loses little work


def take_clips(waveform, positions=CLIP_POSITIONS):
    """Cut one 10-second clip around each position in the piece."""
    clips = []
    for position in positions:
        start = max(0, int(position * len(waveform)) - CLIP_SAMPLES // 2)
        clips.append(waveform[start:start + CLIP_SAMPLES].numpy())
    return clips


def embed_clips(path, feature_extractor, model, positions=CLIP_POSITIONS):
    """Return one AST embedding per clip: a table with 768 numbers per clip."""
    clips = take_clips(load_audio(path, sample_rate=AST_SAMPLE_RATE), positions)
    inputs = feature_extractor(clips, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        return model(input_values=inputs["input_values"].to(DEVICE)).pooler_output.cpu().numpy()


def embed_file(path, feature_extractor, model):
    """Return the average AST embedding of a file's three clips (768 numbers)."""
    return embed_clips(path, feature_extractor, model).mean(axis=0)


def extra_audio_paths():
    """Audio rendered later for step 2, listed in any data/extra_*render_log.csv."""
    paths = []
    for log in sorted(DATA.glob("extra_*render_log.csv")):
        with log.open(encoding="utf-8", newline="") as handle:
            paths += [row["flac_path"] for row in csv.DictReader(handle) if row["status"] == "ok"]
    return paths


def load_existing():
    """Read already computed embeddings so that a restart can skip them."""
    if not OUTPUT_PATH.exists():
        return {}
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def save(embeddings):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH, paths=np.array(list(embeddings)), features=np.stack(list(embeddings.values())))


def load_model():
    """Load AST in evaluation mode, using the safetensors weights only."""
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.eval()
    return feature_extractor, model.to(DEVICE)


def main():
    torch.set_num_threads(6)  # leave part of the CPU free for other work
    feature_extractor, model = load_model()
    embeddings = load_existing()
    todo = [path for path in audio_paths() + extra_audio_paths() if path not in embeddings]
    print(f"{len(embeddings)} cached, {len(todo)} to compute", flush=True)
    for index, path in enumerate(todo, 1):
        embeddings[path] = embed_file(DATA / path, feature_extractor, model)
        if index % SAVE_EVERY == 0:
            save(embeddings)
            print(f"{index}/{len(todo)}", flush=True)
    save(embeddings)
    print(f"saved {len(embeddings)} files to {OUTPUT_PATH.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
