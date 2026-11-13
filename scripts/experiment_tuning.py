"""Step 2: can the AST plus rhythm classifier improve using only features already saved?

AST itself does not change. Three ideas change only the small classifier on top:
  1. C: the setting C=10 was chosen for AST alone, so try other values.
  2. PCA: squeeze AST's 768 numbers into fewer, so that 18 rhythm numbers aren't drowned out.
  3. Clips: train on each 10-second clip as its own example instead of their average.
     Three clips of a piece can be three examples, which is extra training data for free.
     To predict a piece, average the probabilities of its clips.

Every setting is scored on validation and on the unseen-composer folds. The test sets
are not used.

Usage:
    /usr/bin/python3 scripts/experiment_tuning.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_more_clips import ORIGINAL_THREE, load_existing as load_clips
from experiment_rhythm import RHYTHM_SIZE, load_rhythm
from train_baseline import CLASSES, FOLDS, load_rows, log_result, logistic_model, make_xy
from train_probe import EMBEDDING_SIZE, load_embeddings

COLUMNS = EMBEDDING_SIZE + RHYTHM_SIZE
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
PCA_SIZES = [30, 60, 120]
PCA_C_GRID = [0.1, 1.0, 10.0]
CLIP_C_GRID = [0.1, 1.0, 10.0]


def plain(features, c_value):
    """The classifier used so far, with a chosen C."""
    def fit_and_predict(train_rows, test_rows):
        X_train, y_train = make_xy(train_rows, features, COLUMNS)
        X_test, _ = make_xy(test_rows, features, COLUMNS)
        return logistic_model(c_value).fit(X_train, y_train).predict(X_test)
    return fit_and_predict


def with_pca(features, pca_size, c_value):
    """Shrink only the AST columns. Whitening gives every remaining column the same scale."""
    ast_columns = list(range(EMBEDDING_SIZE))
    rhythm_columns = list(range(EMBEDDING_SIZE, COLUMNS))

    def fit_and_predict(train_rows, test_rows):
        X_train, y_train = make_xy(train_rows, features, COLUMNS)
        X_test, _ = make_xy(test_rows, features, COLUMNS)
        shrink_ast = make_pipeline(StandardScaler(), PCA(pca_size, whiten=True, random_state=0))
        columns = ColumnTransformer([("ast", shrink_ast, ast_columns), ("rhythm", StandardScaler(), rhythm_columns)])
        model = make_pipeline(columns, LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000))
        return model.fit(X_train, y_train).predict(X_test)
    return fit_and_predict


def per_clip(clips, rhythm, c_value):
    """Train on every clip separately, then average each test piece's clip probabilities."""
    def clip_matrix(rows):
        return np.stack([np.concatenate([clips[row["audio_path"]][index], rhythm[row["audio_path"]]])
                         for row in rows for index in ORIGINAL_THREE])

    def fit_and_predict(train_rows, test_rows):
        y_train = np.array([row["form"] for row in train_rows for _ in ORIGINAL_THREE])
        model = logistic_model(c_value).fit(clip_matrix(train_rows), y_train)
        probabilities = model.predict_proba(clip_matrix(test_rows))
        by_piece = probabilities.reshape(len(test_rows), len(ORIGINAL_THREE), -1).mean(axis=1)
        return model.classes_[by_piece.argmax(axis=1)]
    return fit_and_predict


def measure(rows, predictions):
    truth = [row["form"] for row in rows]
    return {
        "accuracy": accuracy_score(truth, predictions),
        "balanced_accuracy": balanced_accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, labels=CLASSES, average="macro", zero_division=0),
    }


def try_setting(name, fit_and_predict, train_rows, val_rows, fold_rows):
    """Score one setting on validation and the five folds, and log both."""
    validation = measure(val_rows, fit_and_predict(train_rows, val_rows))
    log_result(name, "step2_validation", validation, "experiment_tuning.py")

    results = []
    for fold in range(FOLDS):
        train = [row for row in fold_rows if int(row["fold"]) != fold]
        held_out = [row for row in fold_rows if int(row["fold"]) == fold]
        results.append(measure(held_out, fit_and_predict(train, held_out)))
    averages = {metric: float(np.mean([result[metric] for result in results])) for metric in results[0]}
    log_result(name, "step2_composer_holdout_mean", averages, "experiment_tuning.py")

    folds = np.array([result["macro_f1"] for result in results])
    print(f"{name:<34} validation {validation['macro_f1']:.3f}   unseen {folds.mean():.3f}   folds {folds.round(3)}",
          flush=True)


def main():
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_works = {row["work_id"] for row in split_rows if row["split"] == "test"}
    fold_rows = [row for row in load_rows("composer_holdout.csv") if row["work_id"] not in test_works]

    embeddings = load_embeddings()
    rhythm = load_rhythm()
    clips = load_clips()
    features = {path: np.concatenate([embeddings[path], rhythm[path]]) for path in rhythm if path in embeddings}

    settings = [(f"AST+rhythm C={c}", plain(features, c)) for c in C_GRID]
    settings += [(f"AST+rhythm PCA={size} C={c}", with_pca(features, size, c)) for size in PCA_SIZES for c in PCA_C_GRID]
    settings += [(f"AST+rhythm per clip C={c}", per_clip(clips, rhythm, c)) for c in CLIP_C_GRID]
    for name, fit_and_predict in settings:
        try_setting(name, fit_and_predict, train_rows, val_rows, fold_rows)


if __name__ == "__main__":
    main()
