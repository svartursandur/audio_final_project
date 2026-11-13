"""Fine-tuning run 3: search the learning rates on validation, then evaluate the best.

Run 2 (finetune_ast.py) barely learned: its training loss stayed near guessing, so its
learning rate was too small. Here every combination of the rates below is trained with
3 seeds, and only validation is scored. The setting with the best average validation
score then gets the full evaluation (validation and the five unseen-composer folds).
The unseen composers play no part in the choice, so their score stays honest.

Each training round takes 1-2 seconds on the GPU server, so the search is affordable.

Usage (on the GPU server):
    python scripts/finetune_search.py
"""
import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_ast import SEEDS, Settings, choose_epochs, full_evaluation, prepare

CLASSIFIER_RATES = [3e-4, 1e-3, 3e-3]
LAST_LAYER_RATES = [1e-5, 1e-4]
MAX_EPOCHS = 40
PATIENCE = 8
SEARCH_LOG = Path(__file__).resolve().parent.parent / "results" / "finetune_search.csv"


def main():
    ast_model, train_rows, val_rows, _ = prepare()
    averages = {}
    with SEARCH_LOG.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["last_layer_lr", "classifier_lr", "seed", "best_epochs", "validation_macro_f1"])
        for last_rate, classifier_rate in product(LAST_LAYER_RATES, CLASSIFIER_RATES):
            settings = Settings(last_rate, classifier_rate, MAX_EPOCHS, PATIENCE)
            print(f"\n=== {settings.describe()} ===", flush=True)
            scores = []
            for seed in SEEDS:
                best_epochs, best_score = choose_epochs(ast_model, train_rows, val_rows, seed, settings)
                writer.writerow([last_rate, classifier_rate, seed, best_epochs, f"{best_score:.4f}"])
                handle.flush()
                scores.append(best_score)
            averages[(last_rate, classifier_rate)] = np.mean(scores)
            print(f"average best validation macro F1 {np.mean(scores):.3f}", flush=True)

    print("\nSearch results (average over seeds of the best validation macro F1):")
    for (last_rate, classifier_rate), score in sorted(averages.items(), key=lambda item: -item[1]):
        print(f"  last layer lr {last_rate:g}, classifier lr {classifier_rate:g}: {score:.3f}")
    best_last, best_classifier = max(averages, key=averages.get)
    full_evaluation(Settings(best_last, best_classifier, MAX_EPOCHS, PATIENCE),
                    "AST partial fine-tuning, run 3 (best of learning-rate search)")


if __name__ == "__main__":
    main()
