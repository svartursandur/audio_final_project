"""Step 2: does hearing five clips of a piece beat hearing three?

The five positions include the original three. Each clip's embedding is saved
separately, so both versions come from the same run and can be compared fairly.
Only training and validation works are processed; the test sets stay untouched.

Usage:
    /usr/bin/python3 scripts/experiment_more_clips.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA
from experiment_step2 import C_VALUE, composer_scores, describe, validation_score
from extract_ast import embed_clips, load_model
from train_baseline import FOLDS, load_rows, log_result, print_scores

ALL_POSITIONS = [0.10, 0.25, 0.50, 0.75, 0.90]
ORIGINAL_THREE = [1, 2, 3]  # the indexes of 0.25, 0.50 and 0.75 in the list above
OUTPUT_PATH = DATA / "features" / "ast_five_clips.npz"
SAVE_EVERY = 25  # files, so an interruption loses little work


def load_existing():
    """Read already computed clips so that a restart can skip them."""
    if not OUTPUT_PATH.exists():
        return {}
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def save(clips):
    np.savez(OUTPUT_PATH, paths=np.array(list(clips)), features=np.stack(list(clips.values())))


def compute_clips(rows):
    """Embed the five clips of every file that isn't cached yet."""
    clips = load_existing()
    todo = [row["audio_path"] for row in rows if row["audio_path"] not in clips]
    print(f"{len(clips)} cached, {len(todo)} to compute", flush=True)
    if todo:
        torch.set_num_threads(6)  # leave part of the CPU free for other work
        feature_extractor, model = load_model()
        for index, path in enumerate(todo, 1):
            clips[path] = embed_clips(DATA / path, feature_extractor, model, ALL_POSITIONS)
            if index % SAVE_EVERY == 0:
                save(clips)
                print(f"{index}/{len(todo)}", flush=True)
        save(clips)
    return clips


def average_clips(clips, clip_indexes):
    """Average the chosen clips, giving one row of 768 numbers per file."""
    return {path: table[clip_indexes].mean(axis=0) for path, table in clips.items()}


def main():
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_works = {row["work_id"] for row in split_rows if row["split"] == "test"}
    fold_rows = [row for row in load_rows("composer_holdout.csv") if row["work_id"] not in test_works]
    known_folds = {row["composer"]: int(row["fold"]) for row in fold_rows}

    clips = compute_clips(train_rows + val_rows)

    reference = None
    for name, clip_indexes in [("3 clips per piece", ORIGINAL_THREE), ("5 clips per piece", list(range(5)))]:
        embeddings = average_clips(clips, clip_indexes)
        print(f"\n{name}")
        validation = validation_score(train_rows, val_rows, embeddings)
        print_scores("  validation", validation)
        log_result(f"AST probe, {name}", "step2_validation", validation, f"C={C_VALUE}")

        results = composer_scores(fold_rows, [], known_folds, embeddings)
        values, text = describe(results)
        print(f"  unseen composers: {text}, per fold {values.round(3)}")
        averages = {metric: float(np.mean([result[metric] for result in results])) for metric in results[0]}
        log_result(f"AST probe, {name}", "step2_composer_holdout_mean", averages, f"C={C_VALUE}")

        if reference is None:
            reference = values
        else:
            better = int((values > reference).sum())
            print(f"  change from 3 clips: {(values - reference).round(3)}, "
                  f"mean {(values - reference).mean():+.3f}, better in {better} of {FOLDS} folds")


if __name__ == "__main__":
    main()
