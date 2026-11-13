"""Step 2: test whether changes to the training data help the AST probe.

Only the training set changes between experiments. Validation and the held-out
composers stay exactly the same works, so the scores can be compared fairly.
The random test split and the real recordings are not used here.

Usage:
    /usr/bin/python3 scripts/experiment_step2.py
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA
from train_baseline import (CLASSES, FOLDS, load_rows, logistic_model, log_result, make_xy,
                            print_scores, score)
from train_probe import EMBEDDING_SIZE, load_embeddings

C_VALUE = 10.0  # kept from the original probe, so that only the data changes


def load_render_log(name):
    """Successful renders from one log, with the column names the other code expects."""
    path = DATA / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ok"]
    for row in rows:
        row["audio_path"] = row["flac_path"]
    return rows


def fold_of_composer(composer, known_folds):
    """Use the composer's existing fold; a new composer gets a fixed fold from a hash."""
    if composer in known_folds:
        return known_folds[composer]
    return int(hashlib.sha256(composer.encode()).hexdigest(), 16) % FOLDS


def validation_score(train_rows, val_rows, embeddings):
    X_train, y_train = make_xy(train_rows, embeddings, EMBEDDING_SIZE)
    X_val, y_val = make_xy(val_rows, embeddings, EMBEDDING_SIZE)
    return score(logistic_model(C_VALUE).fit(X_train, y_train), X_val, y_val)


def composer_scores(fold_rows, extra_rows, known_folds, embeddings):
    """Macro F1 per fold. Extra pieces only ever join training, never the held-out set."""
    results = []
    for fold in range(FOLDS):
        train = [row for row in fold_rows if int(row["fold"]) != fold]
        train += [row for row in extra_rows if fold_of_composer(row["composer"], known_folds) != fold]
        held_out = [row for row in fold_rows if int(row["fold"]) == fold]
        X_train, y_train = make_xy(train, embeddings, EMBEDDING_SIZE)
        X_test, y_test = make_xy(held_out, embeddings, EMBEDDING_SIZE)
        results.append(score(logistic_model(C_VALUE).fit(X_train, y_train), X_test, y_test))
    return results


def describe(results):
    values = np.array([result["macro_f1"] for result in results])
    return values, f"macro F1 {values.mean():.3f} ± {values.std():.3f}"


def main():
    embeddings = load_embeddings()
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_works = {row["work_id"] for row in split_rows if row["split"] == "test"}
    fold_rows = [row for row in load_rows("composer_holdout.csv") if row["work_id"] not in test_works]
    known_folds = {row["composer"]: int(row["fold"]) for row in fold_rows}
    extra_works = [row for row in load_render_log("extra_render_log.csv") if row["form"] in CLASSES]

    experiments = [("original training data", []), ("plus extra marches and fugues", extra_works)]
    reference = None
    for name, extras in experiments:
        print(f"\n{name}: {len(train_rows) + len(extras)} training works")
        validation = validation_score(train_rows + extras, val_rows, embeddings)
        print_scores("  validation", validation)
        log_result(f"AST probe, {name}", "step2_validation", validation, f"C={C_VALUE}")

        results = composer_scores(fold_rows, extras, known_folds, embeddings)
        values, text = describe(results)
        print(f"  unseen composers: {text}, per fold {values.round(3)}")
        averages = {metric: float(np.mean([result[metric] for result in results])) for metric in results[0]}
        log_result(f"AST probe, {name}", "step2_composer_holdout_mean", averages, f"C={C_VALUE}")

        if reference is None:
            reference = values
        else:
            better = int((values > reference).sum())
            print(f"  change from original: {(values - reference).round(3)}, "
                  f"mean {(values - reference).mean():+.3f}, better in {better} of {FOLDS} folds")


if __name__ == "__main__":
    main()
