"""Step 2: do rhythm features help, on their own or added to the AST embeddings?

Everything else matches the linear probe: same works, same folds, same C. The test
sets are not used.

Usage:
    /usr/bin/python3 scripts/experiment_rhythm.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_step2 import C_VALUE
from extract_rhythm import OUTPUT_PATH
from train_baseline import FOLDS, load_rows, log_result, logistic_model, make_xy, print_scores, score
from train_probe import EMBEDDING_SIZE, load_embeddings

RHYTHM_SIZE = 18


def load_rhythm():
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def validation_score(train_rows, val_rows, features, columns):
    X_train, y_train = make_xy(train_rows, features, columns)
    X_val, y_val = make_xy(val_rows, features, columns)
    return score(logistic_model(C_VALUE).fit(X_train, y_train), X_val, y_val)


def composer_scores(fold_rows, features, columns):
    results = []
    for fold in range(FOLDS):
        train = [row for row in fold_rows if int(row["fold"]) != fold]
        held_out = [row for row in fold_rows if int(row["fold"]) == fold]
        X_train, y_train = make_xy(train, features, columns)
        X_test, y_test = make_xy(held_out, features, columns)
        results.append(score(logistic_model(C_VALUE).fit(X_train, y_train), X_test, y_test))
    return results


def main():
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_works = {row["work_id"] for row in split_rows if row["split"] == "test"}
    fold_rows = [row for row in load_rows("composer_holdout.csv") if row["work_id"] not in test_works]

    ast = load_embeddings()
    rhythm = load_rhythm()
    both = {path: np.concatenate([ast[path], rhythm[path]]) for path in rhythm if path in ast}

    experiments = [("AST only", ast, EMBEDDING_SIZE),
                   ("rhythm only", rhythm, RHYTHM_SIZE),
                   ("AST plus rhythm", both, EMBEDDING_SIZE + RHYTHM_SIZE)]
    reference = None
    for name, features, columns in experiments:
        print(f"\n{name}")
        validation = validation_score(train_rows, val_rows, features, columns)
        print_scores("  validation", validation)
        log_result(f"probe, {name}", "step2_validation", validation, f"C={C_VALUE}")

        results = composer_scores(fold_rows, features, columns)
        values = np.array([result["macro_f1"] for result in results])
        print(f"  unseen composers: macro F1 {values.mean():.3f} ± {values.std():.3f}, per fold {values.round(3)}")
        averages = {metric: float(np.mean([result[metric] for result in results])) for metric in results[0]}
        log_result(f"probe, {name}", "step2_composer_holdout_mean", averages, f"C={C_VALUE}")

        if reference is None:
            reference = values
        else:
            better = int((values > reference).sum())
            print(f"  change from AST only: {(values - reference).round(3)}, "
                  f"mean {(values - reference).mean():+.3f}, better in {better} of {FOLDS} folds")


if __name__ == "__main__":
    main()
