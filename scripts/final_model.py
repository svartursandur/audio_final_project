"""The current best model, in one place, so the demo, the analyses and the final test agree.

How it works:
  - Each piece has three 10-second clips. AST turns each clip into 768 numbers.
  - The whole piece also gets 18 rhythm numbers and 24 bass-rhythm numbers.
  - Each clip is one example: its AST numbers followed by the piece's rhythm numbers.
  - In training, every clip also gets two altered audio copies (extract_augmented.py).
  - Logistic regression with C=0.1. A piece's answer is the average over its real clips.

Running this file saves both versions of the trained model to models/:
    /usr/bin/python3 scripts/final_model.py

The options let an analysis switch one part off at a time:
  parts     which features to use: "ast", "rhythm" and "bass"
  versions  which altered copies to add in training (an empty tuple means none)
  per_clip  False averages the three clips into one example instead
"""
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA
from experiment_more_clips import ORIGINAL_THREE, load_existing as load_clips
from experiment_step2 import fold_of_composer
from experiment_rhythm import load_rhythm
from extract_augmented import AUDIO_VERSIONS, load_augmented
from extract_bass_rhythm import load_bass_rhythm
from train_baseline import load_rows, logistic_model

C_VALUE = 0.1
ALL_PARTS = ("ast", "rhythm", "bass")

# Version 2 removes these training pieces, found in the error analysis of the validation
# mistakes (results/v1_before_cleaning/validation_mistakes_notes.md). Validation and test
# pieces are never removed. Version 1 is the same model trained with them.
WRONG_LABELS = {
    "ropartz-guy|fugue|eminor":
        "the MIDI comes from a video titled 'Piece in E flat minor', a different piece, not a fugue",
    "busoni-ferruccio|fugue|variationen-und-fuge-freier-form-bv-213":
        "mostly variations with a fugue at the end, so the clips are probably not fugue "
        "(the same reason 'Prelude and Fugue' titles are excluded)",
}
MODEL_VERSION = "v3"
MODELS = Path(__file__).resolve().parent.parent / "models"

# Version 3 also trains on the ATEPP fugues (select_atepp.py, experiment_atepp.py).
# Only fugues: adding all ATEPP forms made the unseen-composer score worse.
EXTRA_FORMS = ("fugue",)
ATEPP_RENDER_LOG = DATA / "atepp_render_log.csv"


def load_features():
    """Everything the model reads, saved earlier by the extract_*.py scripts."""
    return {"clips": load_clips(), "augmented": load_augmented(),
            "rhythm": load_rhythm(), "bass": load_bass_rhythm()}


def load_splits(remove_wrong_labels=True):
    """Training and validation pieces, plus the pieces used for the unseen-composer folds.

    The random test pieces are left out of the folds, so the test sets stay untouched.
    remove_wrong_labels=False gives version 1, which still has the WRONG_LABELS pieces.
    """
    removed = set(WRONG_LABELS) if remove_wrong_labels else set()
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train" and row["work_id"] not in removed]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_works = {row["work_id"] for row in split_rows if row["split"] == "test"}
    fold_rows = [row for row in load_rows("composer_holdout.csv")
                 if row["work_id"] not in test_works and row["work_id"] not in removed]
    return train_rows, val_rows, fold_rows


def extra_pieces():
    """The rendered ATEPP pieces of EXTRA_FORMS, each with the unseen-composer fold of its composer."""
    known_folds = {row["composer"]: int(row["fold"]) for row in load_rows("composer_holdout.csv")}
    with ATEPP_RENDER_LOG.open(encoding="utf-8", newline="") as handle:
        return [{"composer": row["composer"], "form": row["form"], "audio_path": row["flac_path"],
                 "fold": str(fold_of_composer(row["composer"], known_folds))}
                for row in csv.DictReader(handle) if row["status"] == "ok" and row["form"] in EXTRA_FORMS]


def with_extra_pieces(rows):
    """Add the extra pieces to a list of training pieces.

    For validation every extra piece is added. For an unseen-composer fold, only extra
    pieces whose composer's fold is among the training folds are added, so a composer
    is never both trained on and tested on.
    """
    folds_in_training = {row.get("fold") for row in rows}
    extra = extra_pieces()
    if None in folds_in_training:  # rows from random.csv have no fold: the validation setup
        return rows + extra
    return rows + [row for row in extra if row["fold"] in folds_in_training]


def piece_examples(features, path, parts=ALL_PARTS, versions=(), per_clip=True):
    """Return one piece's examples: its real clips first, then any altered copies."""
    whole_piece = [features[part][path] for part in ("rhythm", "bass") if part in parts]
    clip_sets = [features["clips"][path][ORIGINAL_THREE]]
    clip_sets += [features["augmented"][path][version] for version in versions]
    examples = []
    for clips in clip_sets:
        if not per_clip:
            clips = clips.mean(axis=0, keepdims=True)
        for clip in clips:
            ast = [clip] if "ast" in parts else []
            examples.append(np.concatenate(ast + whole_piece))
    return examples


def train(features, rows, parts=ALL_PARTS, versions=AUDIO_VERSIONS, per_clip=True, add_extra=True):
    """Fit the classifier on the given pieces, plus the extra ATEPP pieces unless add_extra=False."""
    if add_extra:
        rows = with_extra_pieces(rows)
    examples, labels = [], []
    for row in rows:
        piece = piece_examples(features, row["audio_path"], parts, versions, per_clip)
        examples += piece
        labels += [row["form"]] * len(piece)
    return logistic_model(C_VALUE).fit(np.stack(examples), np.array(labels))


def piece_probabilities(model, features, rows, parts=ALL_PARTS, per_clip=True):
    """One row of class probabilities per piece: the average over its real clips."""
    return np.stack([model.predict_proba(np.stack(piece_examples(features, row["audio_path"], parts,
                                                                 per_clip=per_clip))).mean(axis=0)
                     for row in rows])


def save_versions():
    """Train both versions on the training pieces and save them to models/.

    Retraining from the saved features gives the same model, so these files are a
    record and a shortcut, not the only copy.
    """
    MODELS.mkdir(exist_ok=True)
    features = load_features()
    versions = [("v1_before_cleaning", False, False), ("v2_wrong_labels_removed", True, False),
                ("v3_atepp_fugues", True, True)]
    for name, remove, add_extra in versions:
        train_rows, _, _ = load_splits(remove_wrong_labels=remove)
        path = MODELS / f"{name}.joblib"
        joblib.dump(train(features, train_rows, add_extra=add_extra), path)
        extra = len(extra_pieces()) if add_extra else 0
        print(f"saved {path.relative_to(MODELS.parent)} ({len(train_rows)} training pieces + {extra} ATEPP)")


def fit_and_predict(features, parts=ALL_PARTS, versions=AUDIO_VERSIONS, per_clip=True):
    """Make a function for experiment_tuning.try_setting: train on some pieces, predict others."""
    def run(train_rows, test_rows):
        model = train(features, train_rows, parts, versions, per_clip)
        probabilities = piece_probabilities(model, features, test_rows, parts, per_clip)
        return model.classes_[probabilities.argmax(axis=1)]
    return run


if __name__ == "__main__":
    save_versions()
