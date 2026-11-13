"""Calibration and mistakes of the final model, without touching the test sets.

Calibration: when the model says it is 80% sure, is it right about 80% of the time?
We group pieces by how sure the model was and compare that with how often it was right.
This uses the unseen-composer folds, where every one of the 318 pieces gets a
prediction from a model that never saw its composer, and the 56 validation pieces.

Mistakes: every wrong validation piece, with its file path so it can be listened to.

Usage:
    /usr/bin/python3 scripts/analysis_errors.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT
from final_model import load_features, load_splits, piece_probabilities, train
from train_baseline import CLASSES, FOLDS

# Ranges of the model's top probability. With four forms it is never below 0.25.
BIN_EDGES = [0.25, 0.40, 0.60, 0.80, 1.00]
CALIBRATION_PATH = ROOT / "results" / "calibration.csv"
MISTAKES_PATH = ROOT / "results" / "validation_mistakes.csv"


def calibration_table(name, rows, probabilities, classes):
    """Print and return how sure the model was against how often it was right."""
    confidence = probabilities.max(axis=1)
    correct = classes[probabilities.argmax(axis=1)] == np.array([row["form"] for row in rows])
    table, gap = [], 0.0
    print(f"\n{name}: {len(rows)} pieces")
    print("  how sure      pieces   average sureness   right")
    for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        inside = (confidence >= low) & (confidence < high) if high < 1 else (confidence >= low)
        if not inside.any():
            continue
        sureness, right = confidence[inside].mean(), correct[inside].mean()
        gap += inside.sum() / len(rows) * abs(sureness - right)
        print(f"  {low:.2f}–{high:.2f}   {inside.sum():>6}   {sureness:>16.2f}   {right:>5.2f}")
        table.append({"set": name, "low": low, "high": high, "pieces": int(inside.sum()),
                      "average_sureness": round(float(sureness), 3), "right": round(float(right), 3)})
    print(f"  expected calibration error: {gap:.3f} (0 is perfect)")
    print(f"  overall: {confidence.mean():.2f} average sureness, {correct.mean():.2f} right")
    return table


def unseen_composer_probabilities(features, fold_rows):
    """Predict every fold piece with a model trained on the other folds."""
    rows, probabilities = [], []
    for fold in range(FOLDS):
        train_rows = [row for row in fold_rows if int(row["fold"]) != fold]
        held_out = [row for row in fold_rows if int(row["fold"]) == fold]
        model = train(features, train_rows)
        rows += held_out
        probabilities.append(piece_probabilities(model, features, held_out))
    return rows, np.concatenate(probabilities), model.classes_


def save_mistakes(rows, probabilities, classes):
    predictions = classes[probabilities.argmax(axis=1)]
    mistakes = []
    for row, prediction, row_probabilities in zip(rows, predictions, probabilities):
        if prediction == row["form"]:
            continue
        mistake = {"form": row["form"], "predicted": prediction,
                   "sureness": round(float(row_probabilities.max()), 2),
                   "composer": row["composer"], "title": row["title"], "soundfont": row["soundfont"],
                   "file": str((DATA / row["audio_path"]).relative_to(ROOT))}
        mistake.update({f"p_{form}": round(float(p), 2) for form, p in zip(classes, row_probabilities)})
        mistakes.append(mistake)
    mistakes.sort(key=lambda mistake: (mistake["form"], -mistake["sureness"]))
    with MISTAKES_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mistakes[0]))
        writer.writeheader()
        writer.writerows(mistakes)

    print(f"\nWrong validation pieces: {len(mistakes)} of {len(rows)}")
    for mistake in mistakes:
        print(f"  {mistake['form']:>8} -> {mistake['predicted']:<8} {mistake['sureness']:.2f}  "
              f"{mistake['composer'][:22]:<22}  {mistake['title'][:45]}")
    print(f"saved {MISTAKES_PATH.relative_to(ROOT)}")


def main():
    features = load_features()
    train_rows, val_rows, fold_rows = load_splits()

    model = train(features, train_rows)
    val_probabilities = piece_probabilities(model, features, val_rows)
    classes = model.classes_

    print("Validation confusion matrix (rows: true form, columns: predicted)")
    print("          " + "".join(f"{form:>10}" for form in classes))
    matrix = confusion_matrix([row["form"] for row in val_rows], classes[val_probabilities.argmax(axis=1)],
                              labels=classes)
    for form, counts in zip(classes, matrix):
        print(f"{form:>10}" + "".join(f"{count:>10}" for count in counts))

    fold_rows_in_order, fold_probabilities, fold_classes = unseen_composer_probabilities(features, fold_rows)
    table = calibration_table("unseen composers", fold_rows_in_order, fold_probabilities, fold_classes)
    table += calibration_table("validation", val_rows, val_probabilities, classes)
    with CALIBRATION_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"saved {CALIBRATION_PATH.relative_to(ROOT)}")

    save_mistakes(val_rows, val_probabilities, classes)


if __name__ == "__main__":
    main()
