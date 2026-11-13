"""The final test: score the finished model once on the two untouched test sets.

  1. The random test split: 58 rendered pieces whose works were never used before.
  2. The 160 real recordings: real instruments, never used for anything before.

Version 3 (models/v3_atepp_fugues.joblib, see final_model.py) is scored next to the
three baselines, each made exactly as before:
  - majority class and MFCC (train_baseline.py, MFCC setting chosen on validation),
  - frozen AST alone (train_probe.py, C chosen on validation).

Nothing here changes a model. The test pieces' AST clips are computed on the first run
(about 10 seconds per piece) and saved, so the scoring itself takes seconds.

Usage, once, when every model decision is done:
    /usr/bin/python3 -P scripts/final_test.py --final | tee results/final_test_log.txt

To check that the code works without touching the test sets, the same steps can be run
with the validation pieces in place of the test sets. That should give version 3's
validation macro F1 of 0.789, and it saves and logs nothing:
    /usr/bin/python3 -P scripts/final_test.py --check-on-validation
"""
import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, check_disk
from experiment_more_clips import ALL_POSITIONS
from extract_ast import embed_clips, load_model
from final_model import MODELS, load_features, piece_probabilities
from train_baseline import (CLASSES, choose_model, load_features as load_mfcc, load_rows, log_result,
                            logistic_model, majority_model, make_xy, print_scores, score)
from train_probe import choose_c, load_embeddings

VERSION_3 = MODELS / "v3_atepp_fugues.joblib"
TEST_CLIPS = DATA / "features" / "ast_test_clips.npz"
PREDICTIONS = ROOT / "results" / "final_predictions.csv"
BOOTSTRAP_ROUNDS = 2000


def test_sets():
    random_test = [row for row in load_rows("random.csv") if row["split"] == "test"]
    return {"random test split": random_test, "real recordings": load_rows("real_test.csv")}


def load_test_clips():
    if not TEST_CLIPS.exists():
        return {}
    saved = np.load(TEST_CLIPS, allow_pickle=False)
    return dict(zip(saved["paths"].tolist(), saved["features"]))


def compute_test_clips(rows):
    """AST numbers for the same five clip positions as the training pieces (only three are used)."""
    clips = load_test_clips()
    todo = [row["audio_path"] for row in rows if row["audio_path"] not in clips]
    if todo:
        print(f"Computing AST clips for {len(todo)} test pieces (saved as it goes)...", flush=True)
        feature_extractor, model = load_model()
        for index, path in enumerate(todo, 1):
            clips[path] = embed_clips(DATA / path, feature_extractor, model, ALL_POSITIONS)
            if index % 10 == 0 or index == len(todo):
                check_disk()
                np.savez(TEST_CLIPS, paths=np.array(list(clips)), features=np.stack(list(clips.values())))
                print(f"  {index}/{len(todo)}", flush=True)
    return clips


def scores_of(truth, predicted):
    """The same three scores as score() in train_baseline.py, for predictions already made."""
    return {"accuracy": accuracy_score(truth, predicted),
            "balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0)}


def macro_f1_range(truth, predicted, rounds=BOOTSTRAP_ROUNDS):
    """A 95% range for macro F1: resample the pieces with replacement many times."""
    rng = np.random.default_rng(0)
    truth, predicted = np.array(truth), np.array(predicted)
    scores = []
    for _ in range(rounds):
        sample = rng.integers(0, len(truth), len(truth))
        scores.append(f1_score(truth[sample], predicted[sample], labels=CLASSES, average="macro",
                               zero_division=0))
    return np.percentile(scores, [2.5, 97.5])


def print_details(truth, predicted):
    """Confusion matrix and F1 per form. Rows are the true form, columns the prediction."""
    matrix = confusion_matrix(truth, predicted, labels=CLASSES)
    print(f"    {'true / predicted':<18}" + "".join(f"{name:>10}" for name in CLASSES))
    for name, row in zip(CLASSES, matrix):
        print(f"    {name:<18}" + "".join(f"{count:>10}" for count in row))
    per_form = f1_score(truth, predicted, labels=CLASSES, average=None, zero_division=0)
    print("    F1 per form: " + ", ".join(f"{name} {value:.2f}" for name, value in zip(CLASSES, per_form)))


def print_by_instrumentation(rows, predicted):
    """For the real recordings: how many were right for each kind of instrument."""
    groups = {}
    for row, guess in zip(rows, predicted):
        right, total = groups.get(row["instrumentation"], (0, 0))
        groups[row["instrumentation"]] = (right + (guess == row["form"]), total + 1)
    print("    Right by instrumentation: " + ", ".join(
        f"{name} {right}/{total}" for name, (right, total) in sorted(groups.items(), key=lambda item: -item[1][1])))


def baseline_models(train_rows, val_rows):
    """Refit the three baselines exactly as their own scripts do. Returns (name, model, features, columns)."""
    mfcc, embeddings = load_mfcc(), load_embeddings()
    best_mfcc = choose_model(train_rows, val_rows, mfcc)
    best_ast = choose_c(train_rows, val_rows, embeddings)
    models = []
    for name, features, best, make in [
            ("majority class", mfcc, best_mfcc, majority_model),
            (f"MFCC logistic regression (C={best_mfcc['c_value']})", mfcc, best_mfcc,
             lambda: logistic_model(best_mfcc["c_value"])),
            (f"frozen AST alone (C={best_ast['c_value']})", embeddings, best_ast,
             lambda: logistic_model(best_ast["c_value"]))]:
        X, y = make_xy(train_rows, features, best["columns"])
        models.append((name, make().fit(X, y), features, best["columns"]))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="score the test sets (do this once)")
    parser.add_argument("--check-on-validation", action="store_true",
                        help="run the same steps on the validation pieces instead; saves nothing")
    args = parser.parse_args()
    if args.final == args.check_on_validation:
        sys.exit("Choose exactly one: --final or --check-on-validation")
    if args.final and PREDICTIONS.exists():
        sys.exit(f"The final test has already been run ({PREDICTIONS.relative_to(ROOT)} exists).")

    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]  # as the baselines were made
    val_rows = [row for row in split_rows if row["split"] == "val"]
    sets = test_sets() if args.final else {"validation (check only)": val_rows}

    features = load_features()
    if args.final:
        features["clips"].update(compute_test_clips([row for rows in sets.values() for row in rows]))
    version_3 = joblib.load(VERSION_3)
    print("\nBaselines, refitted (their validation tuning is printed below):")
    baselines = baseline_models(train_rows, val_rows)

    saved_rows = []
    for set_name, rows in sets.items():
        truth = [row["form"] for row in rows]
        print(f"\n===== {set_name}: {len(rows)} pieces =====")
        for name, model, base_features, columns in baselines:
            X, y = make_xy(rows, base_features, columns)
            result = score(model, X, y)
            print_scores(f"  {name}", result)
            if args.final:
                log_result(name, f"final_{set_name}", result)

        probabilities = piece_probabilities(version_3, features, rows)
        predicted = list(version_3.classes_[probabilities.argmax(axis=1)])
        result = scores_of(truth, predicted)
        low, high = macro_f1_range(truth, predicted)
        print_scores("  VERSION 3 (final model)", result)
        print(f"    macro F1 95% range: {low:.3f} to {high:.3f}")
        print_details(truth, predicted)
        if "instrumentation" in rows[0]:
            print_by_instrumentation(rows, predicted)
        if args.final:
            log_result("final model v3", f"final_{set_name}", result, f"95% range {low:.3f}-{high:.3f}")
        for row, guess, probs in zip(rows, predicted, probabilities):
            saved_rows.append({"set": set_name, "work_id": row["work_id"], "composer": row["composer"],
                               "title": row.get("title", ""), "instrumentation": row.get("instrumentation", ""),
                               "form": row["form"], "predicted": guess,
                               **{f"p_{name}": f"{p:.3f}" for name, p in zip(version_3.classes_, probs)},
                               "audio_path": row["audio_path"]})

    if args.final:
        with PREDICTIONS.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(saved_rows[0]))
            writer.writeheader()
            writer.writerows(saved_rows)
        print(f"\nEvery prediction saved to {PREDICTIONS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
