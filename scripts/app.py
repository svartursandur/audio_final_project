"""Demo: play or upload some music and see which form the model thinks it is.

The demo prepares audio with exactly the same code that prepared the training
data. The project PDF warns that most broken demos skip this step.

Usage:
    /usr/bin/python3 scripts/app.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import AudioError, to_normalized_flac
from extract_ast import embed_clips, load_model
from extract_bass_rhythm import bass_rhythm_summary
from extract_rhythm import rhythm_summary
from final_model import load_features, load_splits, train

print("Loading the AST model, which takes about 15 seconds...")
FEATURE_EXTRACTOR, AST_MODEL = load_model()


def train_classifier():
    """Fit the current best model (see final_model.py) on the training works only."""
    train_rows, _, _ = load_splits()
    return train(load_features(), train_rows)


CLASSIFIER = train_classifier()


def predict(audio_path):
    """Return {form: probability} for one audio file."""
    if audio_path is None:
        raise ValueError("Please record or upload some audio first.")
    with tempfile.TemporaryDirectory() as folder:
        normalized = Path(folder) / "input.flac"
        try:
            to_normalized_flac(Path(audio_path), normalized)
        except AudioError as error:
            raise ValueError(f"Could not read this audio: {error}") from error
        clip_embeddings = embed_clips(normalized, FEATURE_EXTRACTOR, AST_MODEL)
        whole_piece = np.concatenate([rhythm_summary(normalized), bass_rhythm_summary(normalized)])
    # Same column order as final_model.piece_examples(). The piece's answer is the average over its clips.
    X = np.stack([np.concatenate([clip, whole_piece]) for clip in clip_embeddings])
    probabilities = CLASSIFIER.predict_proba(X).mean(axis=0)
    return {form: float(probability) for form, probability in zip(CLASSIFIER.classes_, probabilities)}


def build_demo():
    # Imported here so that predict() can be tested without Gradio installed.
    import gradio as gr

    return gr.Interface(
        fn=predict,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Play or upload a piece"),
        outputs=gr.Label(num_top_classes=4, label="Predicted form"),
        title="What musical form is this?",
        description="Waltz, nocturne, march or fugue. Trained on synthesized piano, so piano recordings work best. "
                    "It listens to three 10-second clips and measures the rhythm, "
                    "so give it at least 30 seconds if you can.",
        flagging_mode="never",
    )


if __name__ == "__main__":
    build_demo().launch()
