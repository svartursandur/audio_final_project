"""Ablation: remove one part of the final model at a time and measure the drop.

The results table in our notes only ever adds parts. This asks the reverse question:
starting from the full model, how much does each part matter on its own? The test
sets are not used.

Usage:
    /usr/bin/python3 scripts/analysis_ablation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_tuning import try_setting
from final_model import MODEL_VERSION, fit_and_predict, load_features, load_splits

ABLATIONS = [
    ("full model", {}),
    ("without audio copies", {"versions": ()}),
    ("without bass rhythm", {"parts": ("ast", "rhythm")}),
    ("without rhythm", {"parts": ("ast", "bass")}),
    ("without either rhythm feature", {"parts": ("ast",)}),
    ("without per-clip training", {"per_clip": False}),
    # Without AST the three clips would be identical, so use one example per piece and no copies.
    ("without AST (rhythm features only)", {"parts": ("rhythm", "bass"), "versions": (), "per_clip": False}),
]


def main():
    features = load_features()
    train_rows, val_rows, fold_rows = load_splits()
    for name, options in ABLATIONS:
        try_setting(f"ablation {MODEL_VERSION}: {name}", fit_and_predict(features, **options),
                    train_rows, val_rows, fold_rows)


if __name__ == "__main__":
    main()
