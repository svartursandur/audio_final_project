"""Step 2: does training on altered copies of the clips improve the classifier?

Same classifier, features and C as the current best. Only the training examples change:
altered copies are added to the real clips. Validation and held-out pieces are always the
real, unaltered clips. The altered copies keep their piece's rhythm numbers.

Usage:
    /usr/bin/python3 scripts/experiment_augmentation.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_more_clips import ORIGINAL_THREE, load_existing as load_clips
from experiment_rhythm import load_rhythm
from experiment_tuning import try_setting
from extract_augmented import AUDIO_VERSIONS, SPECTROGRAM_VERSIONS, load_augmented
from extract_bass_rhythm import load_bass_rhythm
from train_baseline import load_rows, logistic_model

C_VALUE = 0.1  # the current best value, kept fixed so only the training examples change


def with_copies(clips, augmented, whole_piece, versions):
    """Train on real clips plus the chosen altered versions, then predict on real clips."""
    def real_examples(row):
        path = row["audio_path"]
        return [np.concatenate([clips[path][index], whole_piece[path]]) for index in ORIGINAL_THREE]

    def altered_examples(row):
        path = row["audio_path"]
        return [np.concatenate([clip, whole_piece[path]]) for version in versions for clip in augmented[path][version]]

    def fit_and_predict(train_rows, test_rows):
        examples, labels = [], []
        for row in train_rows:
            piece_examples = real_examples(row) + altered_examples(row)
            examples += piece_examples
            labels += [row["form"]] * len(piece_examples)
        model = logistic_model(C_VALUE).fit(np.stack(examples), np.array(labels))

        test_examples = np.stack([example for row in test_rows for example in real_examples(row)])
        probabilities = model.predict_proba(test_examples)
        by_piece = probabilities.reshape(len(test_rows), len(ORIGINAL_THREE), -1).mean(axis=1)
        return model.classes_[by_piece.argmax(axis=1)]
    return fit_and_predict


def main():
    split_rows = load_rows("random.csv")
    train_rows = [row for row in split_rows if row["split"] == "train"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    test_works = {row["work_id"] for row in split_rows if row["split"] == "test"}
    fold_rows = [row for row in load_rows("composer_holdout.csv") if row["work_id"] not in test_works]

    clips = load_clips()
    augmented = load_augmented()
    rhythm = load_rhythm()
    bass = load_bass_rhythm()
    whole_piece = {path: np.concatenate([rhythm[path], bass[path]]) for path in rhythm}

    experiments = [("no copies", []),
                   ("audio copies", AUDIO_VERSIONS),
                   ("SpecAugment copies", SPECTROGRAM_VERSIONS),
                   ("audio and SpecAugment copies", AUDIO_VERSIONS + SPECTROGRAM_VERSIONS)]
    for name, versions in experiments:
        fit_and_predict = with_copies(clips, augmented, whole_piece, versions)
        try_setting(f"augmentation: {name}, C={C_VALUE}", fit_and_predict, train_rows, val_rows, fold_rows)


if __name__ == "__main__":
    main()
