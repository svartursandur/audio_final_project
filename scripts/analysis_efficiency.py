"""Efficiency: how big is the model, and how long does one prediction take on this CPU?

The timing runs the demo's exact steps on a few validation pieces (never test pieces)
and reports the median time of each step.

Usage:
    /usr/bin/python3 scripts/analysis_efficiency.py
"""
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import probe_duration, to_normalized_flac
from common import DATA, ROOT
from extract_ast import embed_clips, load_model
from extract_bass_rhythm import bass_rhythm_summary
from extract_rhythm import rhythm_summary
from final_model import load_features, load_splits, train

PIECES_TO_TIME = 5
OUTPUT_PATH = ROOT / "results" / "efficiency.txt"


def timed(function, *arguments):
    start = time.perf_counter()
    result = function(*arguments)
    return result, time.perf_counter() - start


def main():
    lines = []

    def report(text):
        print(text, flush=True)
        lines.append(text)

    start = time.perf_counter()
    feature_extractor, ast_model = load_model()
    load_seconds = time.perf_counter() - start
    features = load_features()
    train_rows, val_rows, _ = load_splits()
    classifier, train_seconds = timed(train, features, train_rows)

    ast_parameters = sum(parameter.numel() for parameter in ast_model.parameters())
    logistic = classifier[-1]
    classifier_parameters = logistic.coef_.size + logistic.intercept_.size
    report("Size")
    report(f"  AST (frozen):            {ast_parameters / 1e6:.1f} million parameters, "
           f"{ast_parameters * 4 / 1e6:.0f} MB as 32-bit numbers")
    report(f"  classifier on top:       {classifier_parameters:,} parameters "
           f"({logistic.coef_.shape[1]} inputs x {logistic.coef_.shape[0]} forms, plus 4 offsets)")
    report(f"  start-up: loading AST {load_seconds:.1f} s, training the classifier {train_seconds:.1f} s")

    steps = {"prepare audio (ffmpeg)": [], "AST, 3 clips": [], "rhythm": [], "bass rhythm": [], "classifier": []}
    lengths = []
    for row in val_rows[:PIECES_TO_TIME]:
        source = DATA / row["audio_path"]
        lengths.append(probe_duration(source))
        with tempfile.TemporaryDirectory() as folder:
            normalized = Path(folder) / "input.flac"
            _, seconds = timed(to_normalized_flac, source, normalized)
            steps["prepare audio (ffmpeg)"].append(seconds)
            clips, seconds = timed(embed_clips, normalized, feature_extractor, ast_model)
            steps["AST, 3 clips"].append(seconds)
            rhythm, seconds = timed(rhythm_summary, normalized)
            steps["rhythm"].append(seconds)
            bass, seconds = timed(bass_rhythm_summary, normalized)
            steps["bass rhythm"].append(seconds)
        examples = [list(clip) + list(rhythm) + list(bass) for clip in clips]
        _, seconds = timed(classifier.predict_proba, examples)
        steps["classifier"].append(seconds)

    report(f"\nOne prediction, median over {PIECES_TO_TIME} validation pieces "
           f"(median length {statistics.median(lengths) / 60:.1f} minutes), {torch.get_num_threads()} CPU threads")
    total = 0.0
    for step, seconds in steps.items():
        median = statistics.median(seconds)
        total += median
        report(f"  {step:<24} {median:6.2f} s")
    report(f"  {'total':<24} {total:6.2f} s")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
