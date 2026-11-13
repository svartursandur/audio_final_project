"""Linear probe: logistic regression on frozen AST embeddings.

The same evaluation as train_baseline.py, so only one thing changes: the MFCC
summaries are replaced by AST embeddings. The test sets are only read with --final.

Usage:
    python3 scripts/train_probe.py            # validation and unseen-composer results
    python3 scripts/train_probe.py --final    # also the test sets; do this once, at the end
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_ast import OUTPUT_PATH
from train_baseline import (load_rows, logistic_model, make_xy, print_confusion, print_scores,
                            log_result, run_composer_holdout, run_final, score)

EMBEDDING_SIZE = 768
# With many more columns than training works, stronger regularization (a small C) is usually needed.
C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def load_embeddings():
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def choose_c(train_rows, val_rows, embeddings):
    """Keep the regularization strength with the best validation macro F1."""
    X_train, y_train = make_xy(train_rows, embeddings, EMBEDDING_SIZE)
    X_val, y_val = make_xy(val_rows, embeddings, EMBEDDING_SIZE)
    best = None
    print("Tuning on validation:")
    for c_value in C_VALUES:
        model = logistic_model(c_value).fit(X_train, y_train)
        result = score(model, X_val, y_val)
        print_scores(f"  C={c_value}", result)
        if best is None or result["macro_f1"] > best["macro_f1"]:
            best = {"columns": EMBEDDING_SIZE, "c_value": c_value, **result}
    print(f"Best: C={best['c_value']}\n")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="also evaluate on the test sets")
    args = parser.parse_args()

    embeddings = load_embeddings()
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_rows = [row for row in split_rows if row["split"] == "test"]

    # Until the final run, works in the test split are kept out of the composer folds too.
    test_works = {row["work_id"] for row in test_rows}
    fold_rows = [row for row in load_rows("composer_holdout.csv")
                 if args.final or row["work_id"] not in test_works]

    best = choose_c(train_rows, val_rows, embeddings)
    name = f"AST linear probe (C={best['c_value']})"
    log_result(name, "validation", best)

    X_train, y_train = make_xy(train_rows, embeddings, EMBEDDING_SIZE)
    X_val, y_val = make_xy(val_rows, embeddings, EMBEDDING_SIZE)
    model = logistic_model(best["c_value"]).fit(X_train, y_train)
    print("Validation confusion matrix for the best model:")
    print_confusion(model, X_val, y_val)
    print()

    run_composer_holdout(fold_rows, embeddings, best, name)

    if args.final:
        run_final(train_rows, test_rows, load_rows("real_test.csv"), embeddings, best, name)
    else:
        print("Test sets not used. Run with --final once, when all model decisions are done.")


if __name__ == "__main__":
    main()
