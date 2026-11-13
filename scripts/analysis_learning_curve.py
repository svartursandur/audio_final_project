"""Learning curve: train the final model on 25%, 50%, 75% and 100% of the training pieces.

If the scores are still rising at 100%, more data would probably help. If the training
score is far above the validation score, the model is overfitting.

Each smaller share is drawn 10 times with different random seeds, keeping the four forms
in the same proportions, and the results are averaged. The test sets are not used.

Usage:
    /usr/bin/python3 scripts/analysis_learning_curve.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from final_model import load_features, load_splits, piece_probabilities, train
from train_baseline import CLASSES, FOLDS

FRACTIONS = [0.25, 0.50, 0.75, 1.00]
SEEDS = 10
OUTPUT_PATH = ROOT / "results" / "learning_curve.csv"


def sample(rows, fraction, seed):
    """Keep a share of the pieces, the same share from every form."""
    rng = np.random.default_rng(seed)
    kept = []
    for form in CLASSES:
        form_rows = [row for row in rows if row["form"] == form]
        count = max(1, round(fraction * len(form_rows)))
        kept += [form_rows[index] for index in rng.choice(len(form_rows), size=count, replace=False)]
    return kept


def macro_f1(model, features, rows):
    predictions = model.classes_[piece_probabilities(model, features, rows).argmax(axis=1)]
    return f1_score([row["form"] for row in rows], predictions, labels=CLASSES, average="macro", zero_division=0)


def one_draw(features, train_rows, val_rows, fold_rows, fraction, seed):
    """Training score, validation score and unseen-composer score for one random draw."""
    smaller = sample(train_rows, fraction, seed)
    model = train(features, smaller)
    scores = {"train": macro_f1(model, features, smaller), "validation": macro_f1(model, features, val_rows)}

    fold_scores = []
    for fold in range(FOLDS):
        fold_train = sample([row for row in fold_rows if int(row["fold"]) != fold], fraction, seed)
        held_out = [row for row in fold_rows if int(row["fold"]) == fold]
        fold_scores.append(macro_f1(train(features, fold_train), features, held_out))
    scores["unseen"] = float(np.mean(fold_scores))
    return scores


def main():
    features = load_features()
    train_rows, val_rows, fold_rows = load_splits()
    table = []
    for fraction in FRACTIONS:
        seeds = range(SEEDS) if fraction < 1 else [0]  # all the data has nothing to draw
        draws = [one_draw(features, train_rows, val_rows, fold_rows, fraction, seed) for seed in seeds]
        row = {"fraction": fraction, "train_pieces": len(sample(train_rows, fraction, 0)), "draws": len(draws)}
        for score in ["train", "validation", "unseen"]:
            values = [draw[score] for draw in draws]
            row[f"{score}_mean"] = round(float(np.mean(values)), 3)
            row[f"{score}_std"] = round(float(np.std(values)), 3)
        table.append(row)
        print(f"{fraction:>4.0%} ({row['train_pieces']} pieces): train {row['train_mean']:.3f}   "
              f"validation {row['validation_mean']:.3f} ± {row['validation_std']:.3f}   "
              f"unseen {row['unseen_mean']:.3f} ± {row['unseen_std']:.3f}", flush=True)

    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
