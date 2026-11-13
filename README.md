# Learning from Sound: recognizing musical form

A model that listens to a piece of classical music and says whether it is a **waltz, nocturne, march or fugue**.

It combines a frozen pretrained audio model (AST) with hand-made rhythm features and a logistic-regression classifier. It is trained on piano MIDI rendered to audio, and tested on works it never saw, composers it never heard, and 160 real recordings.

This repository has the code, the data lists, the cached features and every logged result. The report (PDF, handed in separately) explains the method and the results.

## Results

<!-- Write the main results here, or point to the report. All numbers are in results/. -->

## What is in the repository

```
scripts/          every step, from collecting data to the final test (see scripts/README.md)
data/README.md    where the data comes from, licenses, label quality, known problems
data/splits/      the train/validation/test lists, the 5 composer folds, the real test set
data/*.csv        render plans and logs, download logs with licenses and checksums, label audits
data/features/    cached features (MFCC, AST, rhythm, bass rhythm), a few MB in total
models/           the trained classifiers, versions 1 to 3 (version 3 is final)
results/          every run's scores and the analysis outputs
requirements.txt  Python packages
```

**Not included**, because of size or license terms:
- the audio (about 5 GB)
- the soundfonts
- the GiantMIDI and ATEPP downloads, which need you to agree to their terms yourself

The steps below recreate them.

## Setup

You need:
- **Python 3.12 or newer** with the packages in `requirements.txt`. A CPU is enough for everything except the optional fine-tuning.
- **`ffmpeg`**, for decoding audio, and **`fluidsynth`**, for rendering MIDI. FluidSynth is only needed to rebuild the audio.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run of a script that uses AST downloads the model `MIT/ast-finetuned-audioset-10-10-0.4593` from Hugging Face (about 350 MB).

Run every script from the project root, for example `python scripts/train_baseline.py`.

## Reproducing the main results

There are three levels, from quickest to most complete.

### Level 1: from the cached features (minutes, no data download)

The features in `data/features/` are enough to retrain every classifier and rerun every comparison. Each script prints its scores and adds a row to `results/baseline_results.csv`.

| Result in the report | Command | Output |
|---|---|---|
| Majority-class and MFCC baselines | `python scripts/train_baseline.py` | printed, logged |
| Frozen AST alone (linear probe) | `python scripts/train_probe.py` | printed, logged |
| + rhythm features | `python scripts/experiment_rhythm.py` | printed, logged |
| Per-clip training, C values, PCA | `python scripts/experiment_tuning.py` | printed, logged |
| + bass-rhythm features | `python scripts/experiment_bass_rhythm.py` | printed, logged |
| Audio copies and SpecAugment | `python scripts/experiment_augmentation.py` | printed, logged |
| Version 2 against version 3 (ATEPP fugues) | `python scripts/experiment_atepp.py` | printed, logged |
| Train and save the final model | `python scripts/final_model.py` | `models/*.joblib` |
| Ablation | `python scripts/analysis_ablation.py` | printed, logged |
| Learning curve | `python scripts/analysis_learning_curve.py` | `results/learning_curve.csv` |
| Calibration and validation mistakes | `python scripts/analysis_errors.py` | `results/calibration.csv`, `results/validation_mistakes.csv` |
| Final test on both test sets | `python scripts/final_test.py --check-on-validation`, then see below | `results/final_test_log.txt`, `results/final_predictions.csv` |

**About the final test:**
- `final_test.py --final` was run once, and its output is in `results/`. It refuses to run again while `results/final_predictions.csv` exists, so move that file away to rerun it.
- `--check-on-validation` runs the same steps on the validation pieces instead, and should print version 3's validation macro F1 of 0.789.

**Two scripts need the audio:** `analysis_efficiency.py`, which times predictions on real files, and the demo below. Everything else in Level 1 needs only the features.

### Level 2: rebuilding the data and features from scratch (several hours)

1. **GiantMIDI-Piano** (CC BY 4.0).
   - Agree to the disclaimer at <https://github.com/bytedance/GiantMIDI-Piano>.
   - Download `midis_v1.2.zip` and `surname_checked_midis_v1.2.zip` into `data/raw/giantmidi/`.
2. **ATEPP v1.1** (CC BY 4.0).
   - Agree to the terms at <https://github.com/BetsyTang/ATEPP>.
   - Download `ATEPP-1.1.zip` into the project root.
3. **Soundfonts.** Install your system's `fluid-soundfont-gm` package, which provides FluidR3, then run `python scripts/fetch_soundfonts.py`. That downloads the other two soundfonts, about 1.3 GB.
4. **Render the training audio** from the plan in `data/render_plan.csv`. It needs about 3 GB:
   ```bash
   python scripts/render_midi.py
   ```
5. **Download the 160 real recordings,** about 2.3 GB. Downloads are deliberately slow: one request per second, and one file per six seconds from Wikimedia. Every file's URL, license and checksum is logged in `data/real_download_log.csv`.
   ```bash
   python scripts/fetch_real.py
   ```
6. **Compute the features:**
   ```bash
   python scripts/extract_mfcc.py
   python scripts/extract_ast.py
   python scripts/experiment_more_clips.py   # the per-clip AST numbers
   python scripts/extract_rhythm.py
   python scripts/extract_bass_rhythm.py
   python scripts/extract_augmented.py       # audio copies for training (slow, about 2 hours)
   ```
7. **Add the ATEPP fugues:**
   ```bash
   python scripts/select_atepp.py
   python scripts/render_midi.py --plan atepp_plan.csv --log atepp_render_log.csv
   python scripts/extract_atepp_features.py
   ```
8. Then run Level 1.

**The split files in `data/splits/` are part of the repository,** so steps 1–7 rebuild exactly the same train, validation and test sets.

**Optional, to rebuild the lists themselves:** steps 1–11 in `scripts/README.md`. That means searching the sources, labeling titles and making the splits. Search results on Wikimedia Commons and the Internet Archive may have changed since, so the lists could come out slightly different.

**A disk-space guard in `scripts/common.py`** stops any run if free disk falls below 8 GB or `data/` grows past 12 GB.

### Level 3: fine-tuning AST (optional, GPU)

This is the experiment comparing fine-tuned AST with frozen AST. It's in the report, and it doesn't change the final model.
- It needs the training and validation audio from Level 2.
- It was run on one rented H100 for about an hour. On a laptop CPU it would take many hours.

```bash
bash scripts/gpu_setup.sh            # installs ffmpeg and the Python packages on the server
python scripts/finetune_ast.py       # run 2: fixed settings, 3 seeds
python scripts/finetune_search.py    # run 3: learning-rate search on validation, then the best setting with 3 seeds
```

The logs from the runs in the report are `results/finetune_run1_log.txt`, `finetune_run2_log.txt`, `finetune_run3_log.txt` and `finetune_search.csv`.

## The demo

```bash
python scripts/app.py
```

Open the link it prints, then upload or record at least 30 seconds of music. It shows the probability of each form.

The demo prepares audio with exactly the same code as training: the same loudness normalization, clip positions, AST and rhythm features. It needs the training features in `data/features/`, but no audio.

## How the evaluation is kept honest

- **Splits are by work,** so different recordings of the same piece never end up on both sides.
- **Unseen composers:** five folds where each composer is in exactly one fold. The model is tested on composers it never trained on.
- **Decisions were made on validation and unseen composers only.** A change was adopted only if it improved both.
- **The test split and the real recordings were used once, at the very end.** `final_test.py` refuses a second run.
- **Every run is logged** with a timestamp in `results/baseline_results.csv`.

## Licenses and credits

- **GiantMIDI-Piano:** Kong, Li, Chen and Wang, "GiantMIDI-Piano: A large-scale MIDI dataset for classical piano music" (CC BY 4.0).
- **ATEPP:** Zhang et al., "ATEPP: A dataset of automatically transcribed expressive piano performance" (CC BY 4.0).
- **Real recordings:** public domain, CC0, CC BY or CC BY-SA. Each file's author, source and license is in `data/splits/real_test.csv`.
- **Soundfonts:** FluidR3 GM (MIT), Salamander Grand Piano (CC BY 3.0) and Upright Piano KW (CC0).
- **AST:** Gong, Chung and Glass, "AST: Audio Spectrogram Transformer" (2021), with the model from Hugging Face.

`data/README.md` has the full details.
