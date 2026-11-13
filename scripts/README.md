# How the dataset scripts work

The goal: turn messy public metadata into a clean, honest dataset with no work appearing on both sides of a split. Every script uses only the Python standard library, plus the `ffmpeg` and `fluidsynth` programs.

## The pipeline, in run order

| Step | Script | What it does |
|---|---|---|
| 1 | `fetch_static.py` | Downloads the MusicNet, MAESTRO and GiantMIDI metadata tables |
| 2 | `fetch_commons.py` | Searches Wikimedia Commons by title and saves each file's license |
| 3 | `fetch_ia_search.py` | Searches the Internet Archive for audio with a license |
| 4 | `fetch_ia_items.py` | Gets the file list for each promising Internet Archive item |
| 5 | `build_manifest.py` | Combines everything into one table of labelled candidates |
| 6 | `report_counts.py` | Prints how many works, composers and top-composer share each class has |
| 7 | `fetch_soundfonts.py` | Gets the piano soundfonts and logs their licenses |
| 8 | `select_phase2.py` | Chooses what to render and ranks real recordings for the test set |
| 9 | `render_midi.py` | Turns MIDI into audio |
| 10 | `fetch_real.py` | Downloads the real test recordings |
| 11 | `make_splits.py` | Creates the train, validation and test lists |
| 12 | `extract_mfcc.py` | Summarizes each audio file as 40 MFCC numbers and caches them |
| 13 | `train_baseline.py` | Trains the baselines and logs results to `results/baseline_results.csv` |
| 14 | `extract_ast.py` | Turns each audio file into 768 numbers using the frozen AST model |
| 15 | `train_probe.py` | Trains a linear probe on those numbers, evaluated the same way as the baseline |
| 16 | `app.py` | The live demo, using the best classifier: run it after steps 19, 21, 24 and 26, then open the link it prints |
| 17 | `select_extra_giantmidi.py` | Picks hand-checked extra marches and fugues from the full GiantMIDI set |
| 18 | `experiment_step2.py` | Tests whether those extra pieces improve the probe |
| 19 | `experiment_more_clips.py` | Tests whether 5 clips per piece beat 3 |
| 20 | `finetune_ast.py` | Retrains AST's last layer and a classifier in PyTorch |
| 21 | `extract_rhythm.py` | Measures each piece's rhythm as 18 numbers |
| 22 | `experiment_rhythm.py` | Tests rhythm features alone and added to the AST embeddings |
| 23 | `experiment_tuning.py` | Tries other C values, PCA, and training on each clip separately |
| 24 | `extract_bass_rhythm.py` | Measures how the bass and upper notes move against each other as 24 numbers |
| 25 | `experiment_bass_rhythm.py` | Tests whether the bass-rhythm numbers improve the per-clip classifier |
| 26 | `extract_augmented.py` | Makes altered copies of the non-test clips (audio changes and SpecAugment) and embeds them with AST |
| 27 | `experiment_augmentation.py` | Tests whether adding those copies to the training examples helps |
| 28 | `analysis_ablation.py` | Removes one part of the final model at a time to see what each part adds |
| 29 | `analysis_learning_curve.py` | Trains on 25–100% of the training pieces, saved to `results/learning_curve.csv` |
| 30 | `analysis_errors.py` | Confusion matrix, calibration (`results/calibration.csv`) and wrong validation pieces (`results/validation_mistakes.csv`) |
| 31 | `analysis_efficiency.py` | Model size and prediction time on the CPU, saved to `results/efficiency.txt` |
| 32 | `select_atepp.py` | Picks extra pieces from ATEPP, leaving out every work already in the data or the real test set |
| 33 | `render_midi.py --plan atepp_plan.csv --log atepp_render_log.csv` | Renders those ATEPP pieces the same way as the rest |
| 34 | `extract_atepp_features.py` | Computes every feature the model needs for the ATEPP pieces |
| 35 | `experiment_atepp.py` | Tests version 2 against adding the ATEPP fugues (version 3) or all ATEPP pieces |
| 36 | `gpu_setup.sh`, `finetune_ast.py`, `finetune_search.py` | Fine-tuning runs 2 and 3 on a rented GPU, compared with frozen AST on the same pieces |
| 37 | `final_test.py --final` | The one-time final test on the random test split and the 160 real recordings |

Run each one from the project root, for example `python3 scripts/fetch_static.py`. Steps 12 to 15 need PyTorch and scikit-learn, which are installed for your system Python, so run them as `/usr/bin/python3 scripts/train_baseline.py`.

**Linear probe ideas worth explaining.** AST learned to describe sounds from millions of AudioSet clips. We freeze it and reuse that knowledge, so only a small classifier needs training, which suits a dataset of a few hundred works and a laptop. Using three clips per piece samples different parts of it, because AST can only listen to 10 seconds at a time.

**Baseline ideas worth explaining.** MFCCs describe timbre, and averaging them over a whole piece throws away the order of events, so this model can't really hear rhythm or structure. The majority class shows what a useless model scores. Tuning uses only the validation split, and `--final` is kept for one last run on the test sets, as the PDF asks.

## The helper files

- **`final_model.py`** defines the current best model once, so the demo, the analyses and the final test all use the same one. Running it saves both versions to `models/`. Version 2 leaves out the two wrongly labelled training pieces listed in `WRONG_LABELS`.
- **`labeling.py`** decides a piece's form from its title. It is mostly lists of patterns.
- **`composers.py`** recognizes composer names in messy titles.
- **`work_ids.py`** gives each work a single ID, however its title is spelled.
- **`audio_utils.py`** converts audio to 32 kHz mono FLAC at a fixed loudness.
- **`common.py`** handles polite downloading, caching and the disk-space limit.

## The ideas worth being able to explain

**Why label from titles?** Nobody has labelled these recordings by form, and titles usually say it directly. The cost is label noise, so I measured it: 5.7% on a sample of 210 that was checked by hand.

**Why `work_id` matters.** The PDF warns that clips from the same song must stay in one split. Different recordings of the same piece are the same song. Without one shared ID, a model could be tested on a piece it trained on, and the score would be too optimistic. See `work_id()` in `work_ids.py`: it tries a catalog number first, because "Op. 64 No. 2" is the most reliable name a piece has.

**Why some titles are excluded.** "Prelude and Fugue" contains a prelude, so a clip of it might not be a fugue at all. That would be a wrong label, so those are excluded in `label_title()`.

**Why render MIDI at all?** Openly licensed recordings of many forms are scarce and are dominated by a few composers. MIDI gave far more composers per class. The cost is that training audio sounds synthetic, which is exactly what the real test set measures.

**Why three soundfonts?** So the model learns the music and not one piano's sound. A hash of the `work_id` picks the soundfont, which keeps the choice unrelated to the class.

**Why loudness normalization?** Otherwise a model could learn that one source is quieter than another instead of learning the form. See `to_normalized_flac()`.

**The three tests, from easiest to hardest:**
1. `random.csv`: new works, composers possibly seen in training.
2. `composer_holdout.csv`: composers never seen in training.
3. `real_test.csv`: real recordings instead of rendered MIDI.

The gap between these results shows how far the model really generalizes.

## Regular expressions, briefly

Most patterns in `labeling.py` use only these pieces:

| Symbol | Meaning | Example |
|---|---|---|
| `\b` | edge of a word | `\bpolka\b` won't match `polkadot` |
| `s?` | an optional letter | `valses?` matches `valse` and `valses` |
| `\s` | a space | `funeral\s+march` |
| `\d` | a digit | `march\s+\d{1,2}` matches `march 22` |
| `\w*` | any letters | `\w*walzer` matches `kaiserwalzer` |
| `(a\|b)` | either one | `(flat\|sharp)` |

## Safety features

- `common.py` stops any run if `data/` passes 12 GB or free disk falls below 8 GB.
- Requests are limited to one per second, and Wikimedia's file server to one per six seconds, after it asked us to slow down.
- Finished work is logged, so a script can be stopped and restarted without redoing it.
