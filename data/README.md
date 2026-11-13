# Classical form dataset

Audio for classifying a piano piece's musical form: **waltz, nocturne, march or fugue**.

- **Training data:** GiantMIDI-Piano MIDI files rendered to audio with FluidSynth.
- **Held-out test data:** 160 real recordings with open licenses.
- **Audio format:** 32 kHz, mono, 16-bit FLAC, normalized to -23 LUFS, limited to about -1 dBFS.

Rendering at 32 kHz suits EfficientAT and PANNs. For AST (16 kHz), resample when loading.

## Counts

### Rendered MIDI (training side)

381 of 382 planned works rendered. `Golden Bell Waltz` came out silent and was left out. Five more works were removed because they also appear in the real test set, leaving 376.

| Class | Rendered | Used | train | val | test |
|---|---|---|---|---|---|
| waltz | 193 | 193 | 135 | 29 | 29 |
| nocturne | 99 | 96 | 67 | 14 | 15 |
| march | 52 | 52 | 36 | 8 | 8 |
| fugue | 37 | 35 | 24 | 5 | 6 |

Classes are unbalanced, about 5.5 waltzes per fugue, so use class balancing and macro F1.

### Real recordings (held-out test)

| Class | Recordings | Distinct works | Composers | Top composer's share of works | Instrumentation found in text |
|---|---|---|---|---|---|
| waltz | 40 | 40 | 24 | 8% | unknown 13, orchestra 11, band 10, piano 6 |
| nocturne | 40 | 24 | 3 | 92% (Chopin) | unknown 33, piano 7 |
| march | 40 | 40 | 32 | 5% | band 23, unknown 9, piano 5, orchestra 3 |
| fugue | 40 | 40 | 7 | 38% (Bach) | unknown 29, piano 4, organ 4, guitar 1 |

Nocturne has only 24 distinct works with open-licensed recordings. Reaching 40 took 16 further performances of those works, marked `repeat_of_work=1` in `real_test.csv`. Exclude them to score on distinct works only.

## Files

| Path | Contents |
|---|---|
| `manifest_candidates.csv` | Every labelled candidate, with `needs_review` |
| `review.csv` | Ambiguous or flagged titles, with reasons |
| `maestro_nc_reference.csv` | MAESTRO matches, for reference only (non-commercial license) |
| `counts_report.txt` | Per-source and combined tables |
| `audit_round1.csv`, `audit_round2.csv` | Hand-checked labels with verdicts |
| `render_plan.csv`, `render_log.csv` | What was rendered, with which soundfont |
| `real_test_candidates.csv`, `real_download_log.csv` | Ranked candidates; URL, license and SHA-256 per file |
| `splits/random.csv` | 70/15/15 split by work |
| `splits/composer_holdout.csv` | 5 folds; each composer is in exactly one fold |
| `splits/real_test.csv` | Real audio only |
| `audio/midi_render/`, `audio/real/` | Processed audio |
| `audio/real_original/` | Real recordings as downloaded |
| `midi/` | The MIDI files that were rendered |
| `raw/`, `cache/` | Source metadata and API responses |
| `soundfonts/soundfonts.csv` | Each soundfont, its license and checksum |

## Sources and attribution

| Source | Used for | License |
|---|---|---|
| [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano), `surname_checked_midis_v1.2.zip` | Training MIDI | CC BY 4.0; used after agreeing to its disclaimer |
| [Wikimedia Commons](https://commons.wikimedia.org) | 101 test recordings | Per file, see `splits/real_test.csv` |
| [Internet Archive](https://archive.org), including Musopen items | 59 test recordings | Per file, see `splits/real_test.csv` |
| [MusicNet](https://zenodo.org/records/5120004) metadata | Candidate counts only | CC BY 4.0 for the dataset |
| [MAESTRO v3](https://magenta.withgoogle.com/datasets/maestro) metadata | Reference counts only | CC BY-NC-SA 4.0, so excluded |

**GiantMIDI citation:** Qiuqiang Kong, Bochen Li, Jitong Chen and Yuxuan Wang, "GiantMIDI-Piano: A large-scale MIDI dataset for classical piano music."

**Real recordings:** the author credited by the source, the URL and the license for each file are in `splits/real_test.csv`. CC BY and CC BY-SA files require attribution if redistributed. Some sources credit the composer rather than the recording's author, so check the file's source page before redistributing it. License counts: Public domain 80, CC0 36, CC BY-SA 35, CC BY 9.

### Soundfonts

Each work uses one soundfont, chosen by a hash of its `work_id`, so the choice doesn't depend on its class.

| Soundfont | License | Author | Waltz / nocturne / march / fugue |
|---|---|---|---|
| FluidR3 GM | MIT | Frank Wen and contributors | 62 / 37 / 24 / 19 |
| Salamander Grand Piano V3 | CC BY 3.0 | Alexander Holm; SF2 by FreePats | 63 / 28 / 12 / 11 |
| Upright Piano KW | CC0 1.0 | FreePats | 69 / 34 / 16 / 7 |

These planned counts are uneven by chance, since small classes split unevenly. FluidR3 is used from the system package `fluid-soundfont-gm`; the other two are in `soundfonts/`.

## How labels were made

Labels come only from titles, using multilingual keywords: waltz/valse/walzer, nocturne/notturno, march/marche/marsch, fugue/fuga, and similar. See `scripts/labeling.py`.

- **Excluded:** paired pieces such as "Prelude and Fugue" (unless the source separates the movements, as MusicNet does), canons, and titles with no recognized composer on Commons and Internet Archive.
- **Sent to review, not guessed:** hybrids such as Polonaise-Fantaisie, funeral marches, "March" as the month, "in the style of", movements of larger works, excerpts.
- **Flagged and not counted:** multi-piece sets, arrangements, medleys, unresolved work IDs, and public-domain Internet Archive recordings from 1926 or later or undated.

A `work_id` is composer + form + catalog number, or else number and key, or else title words. Splits are made per `work_id`.

## Known noise

- **Label errors:** a hand check of 210 labels found 12 wrong (5.7%) on a sample not used to write the rules: waltz 1, mazurka 1, polonaise 2, nocturne 2, march 3, polka 3, fugue 0, each out of 30. Composer errors: 3 of 210. These were judged from titles, not by listening, and 30 per class gives only a rough rate.
- **Errors known to remain:** hybrids such as `Impromptu-valse` and `Rondo-polka`, "Turkish March" (really a rondo), and some unflagged arrangements.
- **Training and test sound different:** training is all synthesized piano, while the real marches are mostly band and many waltzes are orchestral or band. Most other real recordings have unknown instrumentation. Report piano and non-piano results separately.
- **Composer confound in the nocturne test set:** 92% of its works are by Chopin.
- **Transcription quality:** GiantMIDI was transcribed automatically from YouTube audio, so it contains note errors. Some "piano" pieces are arrangements, such as waltzes on operetta themes.
- **Work IDs are imperfect:** a work spelled two ways may still get two IDs. Only exact ID matches were removed from training, so a few test works could remain there under another ID.
- **Licenses rest on what sources state:** Commons "Public domain" tags were not checked file by file and may sometimes cover only the composition. MusicNet's license is dataset-level.

## Shortfalls

- MusicNet audio (11.1 GB) was not downloaded, so its 18 Bach fugues are not in the test set.
- Fugue training has only 35 works and march only 52.
- Mazurka, polonaise and polka were left out: mazurka fails the composer-share rule, polonaise has few works and is 76% Chopin in real audio, and polka has only 13 real-audio works.

## Extra pieces tested in step 2 (not used in the final training set)

17 marches and 11 fugues from outside GiantMIDI's curated set are in `extra_render_plan.csv` and `audio/midi_render/`. All 28 were checked by hand. They did not improve the probe (macro F1 on unseen composers 0.579 before, 0.570 after), so `splits/` does not include them.

Waltzes and nocturnes from outside the curated set were rejected: in a hand check, 11 of 30 waltzes had the wrong composer or piece, and some were really Chopin works from the real test set. See `audit_extra_round1.csv` and `audit_extra_round2.csv`.

## Reproducing

Run from the project root. GiantMIDI's zip must be downloaded by hand after agreeing to its disclaimer, into `data/raw/giantmidi/`. See `scripts/README.md` for the order.
