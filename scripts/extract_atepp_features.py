"""Compute every feature the final model needs for the rendered ATEPP pieces, in one run.

Reads data/atepp_render_log.csv (written by render_midi.py) and adds each rendered piece
to the same files the rest of the data uses:
  ast_five_clips.npz   AST numbers for 5 clips (the model uses the middle three)
  rhythm.npz           18 rhythm numbers
  bass_rhythm.npz      24 bass-rhythm numbers
  ast_augmented.npz    AST numbers for the altered audio copies used in training

Pieces already in a file are skipped, so the run can be stopped and restarted.

Usage:
    /usr/bin/python3 scripts/extract_atepp_features.py
"""
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_more_clips as five_clips
import extract_augmented as augmented
import extract_bass_rhythm as bass
import extract_rhythm as rhythm
from common import DATA, check_disk
from extract_ast import embed_clips, load_model

RENDER_LOG = DATA / "atepp_render_log.csv"
SAVE_EVERY = 10  # files, so an interruption loses little work


def rendered_paths():
    with RENDER_LOG.open(encoding="utf-8", newline="") as handle:
        return [row["flac_path"] for row in csv.DictReader(handle) if row["status"] == "ok"]


def save_all(clips, rhythms, bass_rhythms, tables):
    five_clips.save(clips)
    rhythm.save(rhythms)
    bass.save(bass_rhythms)
    augmented.save(tables)


def main():
    torch.set_num_threads(augmented.NUMBER_OF_THREADS)
    check_disk()
    clips, rhythms = five_clips.load_existing(), rhythm.load_existing()
    bass_rhythms, tables = bass.load_bass_rhythm(), augmented.load_augmented()
    paths = rendered_paths()
    todo = [path for path in paths
            if not all(path in features for features in (clips, rhythms, bass_rhythms, tables))]
    print(f"{len(paths)} rendered ATEPP pieces, {len(todo)} to compute", flush=True)

    feature_extractor, model = load_model()
    for index, path in enumerate(todo, 1):
        audio = DATA / path
        clips[path] = embed_clips(audio, feature_extractor, model, five_clips.ALL_POSITIONS)
        rhythms[path] = rhythm.rhythm_summary(audio)
        bass_rhythms[path] = bass.bass_rhythm_summary(audio)
        tables[path] = augmented.augmented_table(audio, feature_extractor, model, augmented.rng_for(path))
        if index % SAVE_EVERY == 0:
            save_all(clips, rhythms, bass_rhythms, tables)
            print(f"{index}/{len(todo)}", flush=True)
    save_all(clips, rhythms, bass_rhythms, tables)
    print(f"saved features for {len(todo)} pieces")


if __name__ == "__main__":
    main()
