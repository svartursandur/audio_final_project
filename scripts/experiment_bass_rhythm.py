"""Step 2: does the bass and upper-note rhythm feature improve the per-clip classifier?

Everything else stays the same as the best setting from experiment_tuning.py, so any
difference comes from the added numbers. The test sets are not used.

Usage:
    /usr/bin/python3 scripts/experiment_bass_rhythm.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_more_clips import load_existing as load_clips
from experiment_rhythm import load_rhythm
from experiment_tuning import per_clip, try_setting
from extract_bass_rhythm import load_bass_rhythm
from train_baseline import load_rows

C_VALUE = 0.1  # the best per-clip value from experiment_tuning.py


def main():
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_works = {row["work_id"] for row in split_rows if row["split"] == "test"}
    fold_rows = [row for row in load_rows("composer_holdout.csv") if row["work_id"] not in test_works]

    clips = load_clips()
    rhythm = load_rhythm()
    bass = load_bass_rhythm()
    rhythm_and_bass = {path: np.concatenate([rhythm[path], bass[path]]) for path in rhythm}

    experiments = [("per clip, AST+rhythm", rhythm),
                   ("per clip, AST+bass rhythm", bass),
                   ("per clip, AST+rhythm+bass rhythm", rhythm_and_bass)]
    for name, extra_features in experiments:
        try_setting(f"{name} C={C_VALUE}", per_clip(clips, extra_features, C_VALUE), train_rows, val_rows, fold_rows)


if __name__ == "__main__":
    main()
