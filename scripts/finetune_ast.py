"""Partial fine-tuning: retrain AST's last layer and a new classifier in PyTorch.

AST has 12 layers. The first 11 stay frozen. Their output never changes, so it is
computed once and saved to disk. Training then only has to run the last layer and
the classifier, which is what makes this practical on a laptop CPU.

The evaluation matches the linear probe: validation chooses the number of training
rounds, and the composer folds use that number. The test sets are not used.

Run 1 (results/finetune_run1_log.txt) used a classifier learning rate of 1e-3, a fixed
learning rate and up to 8 rounds. Validation jumped between 0.36 and 0.57 from round to
round, which looks like unstable training. Run 2 changes three things:
  - a 10 times smaller classifier learning rate (1e-4),
  - a learning rate that shrinks smoothly over the rounds (a cosine schedule),
  - up to 12 rounds, stopping early when validation hasn't improved for 3 rounds.
It also uses the same training data as version 3 (final_model.py): the two wrong labels
are removed and the 33 ATEPP fugues are added to training, by the same fold rule.
It repeats everything for 3 seeds and reports the mean and spread.

The comparison is with frozen AST: logistic regression on the average of the three
clips' AST numbers, trained on exactly the same pieces.

Run 2 barely learned: training loss only fell from 1.5 to about 1.15 (guessing gives 1.39),
so its learning rate was too small. Run 3 searches the learning rates on validation
(finetune_search.py) and evaluates the best setting here.

Usage (run on the rented GPU server; on a laptop CPU it takes many hours):
    python scripts/finetune_ast.py                     # run 2's settings
    python scripts/finetune_search.py                  # run 3: search, then evaluate the best
"""
import copy
import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, check_disk
from experiment_more_clips import ORIGINAL_THREE, load_existing as load_clips
from experiment_step2 import C_VALUE, composer_scores, validation_score
from extract_ast import AST_SAMPLE_RATE, DEVICE, load_model, take_clips
from extract_mfcc import load_audio
from final_model import extra_pieces, load_splits, with_extra_pieces
from train_baseline import CLASSES, FOLDS, log_result

SEEDS = [1000, 2000, 3000]  # far apart, because each fold adds 1 to 5 to the seed
CACHE_DIR = DATA / "features" / "ast_before_last_layer"
EMBEDDING_SIZE = 768
CLIPS_PER_PIECE = 3
BATCH_SIZE = 8
SMALLEST_RATE_FACTOR = 0.1  # the schedule ends at a tenth of the starting rate
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
NUMBER_OF_THREADS = 10  # leave part of the CPU free for other work


@dataclass
class Settings:
    """The training choices that differ between runs. The defaults are run 2's."""
    last_layer_learning_rate: float = 1e-5  # small, because these weights are already good
    classifier_learning_rate: float = 1e-4  # run 1 used 1e-3
    max_epochs: int = 12
    patience: int = 3  # stop after this many rounds without a better validation score

    def describe(self):
        return (f"last layer lr {self.last_layer_learning_rate:g}, classifier lr "
                f"{self.classifier_learning_rate:g}, up to {self.max_epochs} rounds")


def cache_path(audio_path):
    return CACHE_DIR / f"{hashlib.sha256(audio_path.encode()).hexdigest()[:24]}.npy"


def frozen_layers_output(input_values, ast_model):
    """Run AST up to, but not including, its last layer."""
    with torch.no_grad():
        hidden = ast_model.embeddings(input_values.to(DEVICE))
        for layer in ast_model.layers[:-1]:
            hidden = layer(hidden)
    return hidden.cpu()


def build_cache(rows, feature_extractor, ast_model):
    """Save the frozen layers' output for each piece. Half precision halves the disk use."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [row["audio_path"] for row in rows if not cache_path(row["audio_path"]).exists()]
    print(f"{len(rows) - len(todo)} cached, {len(todo)} to compute", flush=True)
    for index, path in enumerate(todo, 1):
        clips = take_clips(load_audio(DATA / path, sample_rate=AST_SAMPLE_RATE))
        inputs = feature_extractor(clips, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
        hidden = frozen_layers_output(inputs["input_values"], ast_model).numpy().astype(np.float16)
        np.save(cache_path(path), hidden)
        if index % 25 == 0:
            check_disk()
            print(f"cached {index}/{len(todo)}", flush=True)


def load_clip(audio_path, clip_index):
    """Read one clip's saved output, without loading the whole file into memory."""
    table = np.load(cache_path(audio_path), mmap_mode="r")
    return torch.from_numpy(table[clip_index].astype(np.float32))


class PartlyFineTunedAST(nn.Module):
    """AST's last layer, followed by a new classifier for our four forms."""

    def __init__(self, ast_model):
        super().__init__()
        self.last_layer = copy.deepcopy(ast_model.layers[-1])
        self.layernorm = copy.deepcopy(ast_model.layernorm)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(EMBEDDING_SIZE, len(CLASSES))

    def embed(self, hidden):
        """The same final steps as AST: average its two summary tokens."""
        output = self.layernorm(self.last_layer(hidden))
        return (output[:, 0] + output[:, 1]) / 2

    def forward(self, hidden):
        return self.classifier(self.dropout(self.embed(hidden)))


def class_weights(rows):
    """Rare classes get a larger weight, like class_weight='balanced' in the baseline."""
    counts = np.array([sum(row["form"] == form for row in rows) for form in CLASSES], dtype=np.float32)
    return torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)


def make_optimizer(model, settings):
    return torch.optim.AdamW([
        {"params": list(model.last_layer.parameters()) + list(model.layernorm.parameters()),
         "lr": settings.last_layer_learning_rate},
        {"params": model.classifier.parameters(), "lr": settings.classifier_learning_rate},
    ], weight_decay=WEIGHT_DECAY)


def rate_factor(epoch, max_epochs):
    """How much of the starting learning rate to use in a round (counted from 0).

    A cosine curve: full size at the start, shrinking smoothly to SMALLEST_RATE_FACTOR
    by the last possible round. Big steps early, careful small steps at the end.
    """
    progress = epoch / max(max_epochs - 1, 1)
    return SMALLEST_RATE_FACTOR + (1 - SMALLEST_RATE_FACTOR) * 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, rows, optimizer, loss_function):
    """Show every piece once, using a randomly chosen clip of each."""
    model.train()
    order = torch.randperm(len(rows)).tolist()
    total_loss = 0.0
    for start in range(0, len(order), BATCH_SIZE):
        batch = [rows[index] for index in order[start:start + BATCH_SIZE]]
        hidden = torch.stack([load_clip(row["audio_path"], int(torch.randint(CLIPS_PER_PIECE, (1,))))
                              for row in batch]).to(DEVICE)
        labels = torch.tensor([CLASSES.index(row["form"]) for row in batch]).to(DEVICE)
        loss = loss_function(model(hidden), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(rows)


def predict_form(model, audio_path):
    """Average the classifier's scores over a piece's three clips, then take the best form."""
    hidden = torch.stack([load_clip(audio_path, index) for index in range(CLIPS_PER_PIECE)]).to(DEVICE)
    with torch.no_grad():
        scores = model(hidden).mean(dim=0)
    return CLASSES[int(scores.argmax())]


def evaluate(model, rows):
    model.eval()
    truth = [row["form"] for row in rows]
    predicted = [predict_form(model, row["audio_path"]) for row in rows]
    return {
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0),
    }


def train_model(ast_model, train_rows, epochs, seed, settings, val_rows=None):
    """Train for up to a set number of rounds.

    With val_rows, validation is scored after each round, and training stops early
    once it hasn't improved for settings.patience rounds. The learning-rate schedule
    always spans settings.max_epochs, so a retrain for fewer rounds follows the same curve.
    """
    torch.manual_seed(seed)
    model = PartlyFineTunedAST(ast_model).to(DEVICE)
    optimizer = make_optimizer(model, settings)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: rate_factor(epoch, settings.max_epochs))
    loss_function = nn.CrossEntropyLoss(weight=class_weights(train_rows).to(DEVICE))
    history = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_rows, optimizer, loss_function)
        schedule.step()
        if val_rows:
            macro_f1 = evaluate(model, val_rows)["macro_f1"]
            history.append(macro_f1)
            print(f"  epoch {epoch}: training loss {train_loss:.3f}, validation macro F1 {macro_f1:.3f} "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
            if epoch - (int(np.argmax(history)) + 1) >= settings.patience:
                print(f"  no improvement for {settings.patience} rounds, stopping", flush=True)
                break
    return model, history


def choose_epochs(ast_model, train_rows, val_rows, seed, settings):
    """Train while scoring validation each round. Return the best round and its score."""
    _, history = train_model(ast_model, with_extra_pieces(train_rows), settings.max_epochs, seed,
                             settings, val_rows)
    return int(np.argmax(history)) + 1, max(history)


def run_one_seed(ast_model, train_rows, val_rows, fold_rows, seed, settings, run_name):
    """Choose the number of rounds on validation, then score validation and the five folds."""
    print(f"\nSeed {seed}. Choosing the number of training rounds on validation:", flush=True)
    best_epochs, best_score = choose_epochs(ast_model, train_rows, val_rows, seed, settings)
    print(f"Best: {best_epochs} rounds, validation macro F1 {best_score:.3f}", flush=True)
    model, _ = train_model(ast_model, with_extra_pieces(train_rows), best_epochs, seed, settings)
    validation = evaluate(model, val_rows)
    notes = f"seed={seed} epochs={best_epochs} {settings.describe()}"
    log_result(run_name, "validation", validation, notes)

    print("Unseen composers:", flush=True)
    folds = []
    for fold in range(FOLDS):
        train = with_extra_pieces([row for row in fold_rows if int(row["fold"]) != fold])
        held_out = [row for row in fold_rows if int(row["fold"]) == fold]
        model, _ = train_model(ast_model, train, best_epochs, seed + fold + 1, settings)
        folds.append(evaluate(model, held_out))
        print(f"  fold {fold}: macro F1 {folds[-1]['macro_f1']:.3f}", flush=True)
    averages = {metric: float(np.mean([result[metric] for result in folds])) for metric in folds[0]}
    log_result(run_name, "composer_holdout_mean", averages, notes)
    return validation["macro_f1"], np.array([result["macro_f1"] for result in folds])


def frozen_ast_features():
    """Frozen AST's numbers for every piece: the average over its three clips.

    This equals ast_embeddings.npz for the original pieces, and also covers the ATEPP pieces.
    """
    return {path: clips[ORIGINAL_THREE].mean(axis=0) for path, clips in load_clips().items()}


def prepare():
    """Load the pieces and AST, and make sure every piece's frozen-layer output is cached."""
    torch.set_num_threads(NUMBER_OF_THREADS)
    train_rows, val_rows, fold_rows = load_splits()  # wrong labels removed, test pieces left out
    extra = extra_pieces()  # the ATEPP fugues, added to training only
    feature_extractor, ast_model = load_model()
    build_cache(train_rows + val_rows + extra, feature_extractor, ast_model)
    return ast_model, train_rows, val_rows, fold_rows


def full_evaluation(settings, run_name):
    """Validation and the five unseen-composer folds for every seed, then the frozen-AST comparison."""
    ast_model, train_rows, val_rows, fold_rows = prepare()
    extra = extra_pieces()
    print(f"\n{run_name}: {settings.describe()}", flush=True)
    validation_scores, fold_scores = [], []
    for seed in SEEDS:
        validation, folds = run_one_seed(ast_model, train_rows, val_rows, fold_rows, seed,
                                         settings, run_name)
        validation_scores.append(validation)
        fold_scores.append(folds)
    validation_scores, fold_scores = np.array(validation_scores), np.array(fold_scores)
    unseen = fold_scores.mean(axis=1)  # one unseen-composer score per seed

    print(f"\nFine-tuning over {len(SEEDS)} seeds:")
    print(f"  validation        macro F1 {validation_scores.mean():.3f} ± {validation_scores.std():.3f}")
    print(f"  unseen composers  macro F1 {unseen.mean():.3f} ± {unseen.std():.3f}")
    print(f"  per fold, averaged over seeds: {fold_scores.mean(axis=0).round(3)}")

    features = frozen_ast_features()
    known_folds = {row["composer"]: int(row["fold"]) for row in fold_rows}
    probe_validation = validation_score(train_rows + extra, val_rows, features)["macro_f1"]
    probe = np.array([result["macro_f1"] for result in composer_scores(fold_rows, extra, known_folds, features)])
    print(f"Frozen AST (C={C_VALUE}), same pieces: validation {probe_validation:.3f}, "
          f"unseen composers {probe.mean():.3f}")
    change = fold_scores.mean(axis=0) - probe
    print(f"Fine-tuning minus frozen AST, per fold: {change.round(3)}, mean {change.mean():+.3f}, "
          f"better in {int((change > 0).sum())} of {FOLDS} folds")


if __name__ == "__main__":
    full_evaluation(Settings(), "AST partial fine-tuning, run 2")
