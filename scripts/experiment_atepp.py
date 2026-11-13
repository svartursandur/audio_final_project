"""Step 2: do the extra ATEPP pieces improve version 2 of the model?

Same model, settings, validation pieces and folds as version 2 (final_model.py). The
only change is extra training pieces. They never join validation or a held-out fold:
in a fold, a piece only trains if its composer belongs to another fold, so a composer
is never both trained on and tested on. The test sets are not used.

Three settings:
  - no ATEPP            version 2, for comparison
  - ATEPP fugues only   checks the fugue gain on its own (this became version 3)
  - all ATEPP           fugues plus the waltzes, nocturnes and marches

Usage:
    /usr/bin/python3 scripts/experiment_atepp.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA
from experiment_step2 import fold_of_composer
from experiment_tuning import try_setting
from final_model import load_features, load_splits, piece_probabilities, train

RENDER_LOG = DATA / "atepp_render_log.csv"


def load_atepp_rows():
    """ATEPP pieces that rendered, in the same shape as rows from the split files."""
    with RENDER_LOG.open(encoding="utf-8", newline="") as handle:
        return [{"work_id": row["work_id"], "composer": row["composer"], "form": row["form"],
                 "title": row["title"], "audio_path": row["flac_path"]}
                for row in csv.DictReader(handle) if row["status"] == "ok"]


def fit_and_predict_with_extra(features, extra_rows, known_folds):
    """Like final_model.fit_and_predict, plus extra training pieces.

    When the pieces being predicted are one unseen-composer fold, extra pieces by
    composers in that fold are left out of training.
    """
    def run(train_rows, test_rows):
        held_out = {row.get("fold") for row in test_rows}
        extra = extra_rows
        if None not in held_out:  # one held-out fold, not the validation pieces
            fold = int(held_out.pop())
            extra = [row for row in extra_rows if fold_of_composer(row["composer"], known_folds) != fold]
        # add_extra=False: only the extra pieces chosen here, not version 3's built-in ATEPP fugues
        model = train(features, train_rows + extra, add_extra=False)
        probabilities = piece_probabilities(model, features, test_rows)
        return model.classes_[probabilities.argmax(axis=1)]
    return run


def main():
    features = load_features()
    train_rows, val_rows, fold_rows = load_splits()
    known_folds = {row["composer"]: int(row["fold"]) for row in fold_rows}
    atepp = [row for row in load_atepp_rows() if row["audio_path"] in features["augmented"]]
    print(f"{len(atepp)} ATEPP pieces with features", flush=True)

    experiments = [("no ATEPP", []),
                   ("ATEPP fugues only", [row for row in atepp if row["form"] == "fugue"]),
                   ("all ATEPP", atepp)]
    for name, extra in experiments:
        try_setting(f"v2 + {name}", fit_and_predict_with_extra(features, extra, known_folds),
                    train_rows, val_rows, fold_rows)


if __name__ == "__main__":
    main()
