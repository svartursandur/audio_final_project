"""Baselines: always guessing the most common class, and logistic regression on MFCCs.

Decisions are made on the validation split only. The random test split and the
real recordings are read only when this script is run with --final, because the
project rules say the test set should be touched once, at the very end.

Usage:
    python3 scripts/train_baseline.py            # validation and unseen-composer results
    python3 scripts/train_baseline.py --final    # also the test sets; do this once, at the end
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT
from extract_mfcc import N_MFCC, OUTPUT_PATH

CLASSES = ["waltz", "nocturne", "march", "fugue"]
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # smaller C means a simpler model
FEATURE_SETS = {"mean": N_MFCC, "mean+std": 2 * N_MFCC}  # how many of the 40 columns to use
RESULTS_PATH = ROOT / "results" / "baseline_results.csv"
FOLDS = 5


def load_rows(name):
    with (DATA / "splits" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_features():
    saved = np.load(OUTPUT_PATH, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def make_xy(rows, features, columns):
    """Build the feature matrix X and the list of class labels y."""
    X = np.stack([features[row["audio_path"]][:columns] for row in rows])
    y = np.array([row["form"] for row in rows])
    return X, y


def logistic_model(c_value):
    """Scale the features, then fit logistic regression. Rare classes get extra weight."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000))


def majority_model():
    return DummyClassifier(strategy="most_frequent")


def score(model, X, y):
    """The PDF recommends macro F1 and balanced accuracy for imbalanced classes."""
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, labels=CLASSES, average="macro", zero_division=0),
    }


def print_scores(name, result):
    print(f"{name:<46} accuracy {result['accuracy']:.3f}   "
          f"balanced {result['balanced_accuracy']:.3f}   macro F1 {result['macro_f1']:.3f}")


def print_confusion(model, X, y):
    """Rows are the true class, columns the predicted class."""
    matrix = confusion_matrix(y, model.predict(X), labels=CLASSES)
    print(f"{'true / predicted':<18}" + "".join(f"{name:>10}" for name in CLASSES))
    for name, row in zip(CLASSES, matrix):
        print(f"{name:<18}" + "".join(f"{count:>10}" for count in row))


def choose_model(train_rows, val_rows, features):
    """Try every feature set and C value, and keep the best by validation macro F1."""
    best = None
    print("Tuning on validation:")
    for feature_name, columns in FEATURE_SETS.items():
        X_train, y_train = make_xy(train_rows, features, columns)
        X_val, y_val = make_xy(val_rows, features, columns)
        for c_value in C_VALUES:
            model = logistic_model(c_value).fit(X_train, y_train)
            result = score(model, X_val, y_val)
            print_scores(f"  {feature_name}, C={c_value}", result)
            if best is None or result["macro_f1"] > best["macro_f1"]:
                best = {"feature_name": feature_name, "columns": columns, "c_value": c_value, **result}
    print(f"Best: {best['feature_name']}, C={best['c_value']}\n")
    return best


def composer_holdout(rows, features, make_model, columns):
    """Train on four groups of composers and test on the fifth, five times."""
    results = []
    for fold in range(FOLDS):
        train = [row for row in rows if int(row["fold"]) != fold]
        held_out = [row for row in rows if int(row["fold"]) == fold]
        X_train, y_train = make_xy(train, features, columns)
        X_test, y_test = make_xy(held_out, features, columns)
        results.append(score(make_model().fit(X_train, y_train), X_test, y_test))
    return results


def mean_and_std(results, metric):
    values = [result[metric] for result in results]
    return float(np.mean(values)), float(np.std(values))


def log_result(model, evaluation, result, notes=""):
    """Append one line to the results log, as the PDF recommends."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["time", "model", "evaluation", "accuracy", "balanced_accuracy", "macro_f1", "notes"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), model, evaluation,
                         f"{result['accuracy']:.4f}", f"{result['balanced_accuracy']:.4f}",
                         f"{result['macro_f1']:.4f}", notes])


def run_composer_holdout(fold_rows, features, best, name_of_best):
    print("Unseen composers (5 folds, mean ± std across folds):")
    models = [("majority class", majority_model, FEATURE_SETS["mean"]),
              (name_of_best, lambda: logistic_model(best["c_value"]), best["columns"])]
    for name, make_model, columns in models:
        results = composer_holdout(fold_rows, features, make_model, columns)
        averages = {}
        text = []
        for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
            average, spread = mean_and_std(results, metric)
            averages[metric] = average
            text.append(f"{metric} {average:.3f} ± {spread:.3f}")
        print(f"  {name:<30} " + "   ".join(text))
        log_result(name, "composer_holdout_mean", averages)
    print()


def run_final(train_rows, test_rows, real_rows, features, best, name_of_best):
    """Evaluate on the untouched test sets. Keep this for the very end."""
    X_train, y_train = make_xy(train_rows, features, best["columns"])
    model = logistic_model(best["c_value"]).fit(X_train, y_train)
    majority = majority_model().fit(X_train, y_train)
    for split_name, rows in [("random test split", test_rows), ("real recordings", real_rows)]:
        X, y = make_xy(rows, features, best["columns"])
        print(f"FINAL, {split_name}:")
        for name, fitted in [("majority class", majority), (name_of_best, model)]:
            result = score(fitted, X, y)
            print_scores(f"  {name}", result)
            log_result(name, f"final_{split_name}", result)
        print_confusion(model, X, y)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="also evaluate on the test sets")
    args = parser.parse_args()

    features = load_features()
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_rows = [row for row in split_rows if row["split"] == "test"]

    # Until the final run, works in the test split are kept out of the composer folds too.
    test_works = {row["work_id"] for row in test_rows}
    fold_rows = [row for row in load_rows("composer_holdout.csv")
                 if args.final or row["work_id"] not in test_works]

    print(f"train {len(train_rows)} works, validation {len(val_rows)} works\n")

    X_train, y_train = make_xy(train_rows, features, FEATURE_SETS["mean"])
    X_val, y_val = make_xy(val_rows, features, FEATURE_SETS["mean"])
    majority = majority_model().fit(X_train, y_train)
    majority_result = score(majority, X_val, y_val)
    print_scores("Majority class, validation", majority_result)
    log_result("majority class", "validation", majority_result)
    print()

    best = choose_model(train_rows, val_rows, features)
    name_of_best = f"logistic regression ({best['feature_name']}, C={best['c_value']})"
    log_result(name_of_best, "validation", best)

    X_train, y_train = make_xy(train_rows, features, best["columns"])
    X_val, y_val = make_xy(val_rows, features, best["columns"])
    model = logistic_model(best["c_value"]).fit(X_train, y_train)
    print("Validation confusion matrix for the best model:")
    print_confusion(model, X_val, y_val)
    print()

    run_composer_holdout(fold_rows, features, best, name_of_best)

    if args.final:
        run_final(train_rows, test_rows, load_rows("real_test.csv"), features, best, name_of_best)
    else:
        print("Test sets not used. Run with --final once, when all model decisions are done.")


if __name__ == "__main__":
    main()
