"""Choose extra training pieces from the full GiantMIDI set, skipping doubtful ones.

The curated set only keeps pieces whose YouTube title names the composer. The
pieces outside it are less certain, and some are simply the wrong recording: a
"Nocturne by Sahla" whose YouTube title says Chopin. Those would add wrong
composers and could leak Chopin's real test pieces into training.

A hand check of the first version found about 1 piece in 5 had the wrong composer
or piece, so a piece is now kept only if it passes all the checks below. Every
piece that remains was then checked by hand (data/audit_extra_round1.csv).
"""
import collections
import csv
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manifest import canonical_composer
from common import DATA, RAW
from composers import find_composers
from labeling import fold, label_title
from select_phase2 import FINAL_CLASSES, assign_soundfont, soundfont_names
from work_ids import catalog_key, work_id

CURATED_ZIP = RAW / "giantmidi" / "surname_checked_midis_v1.2.zip"
FULL_ZIP = RAW / "giantmidi" / "midis_v1.2.zip"
METADATA = RAW / "giantmidi" / "full_music_pieces_youtube_similarity_pianosoloprob.csv"
PLAN_PATH = DATA / "extra_render_plan.csv"
# Waltzes and nocturnes outside the curated set failed the hand check (11 of 30 had the wrong
# composer or piece), and waltz already has plenty of works. Only these two classes are added.
FORMS_TO_ADD = ["march", "fugue"]
PLAN_COLUMNS = ["work_id", "composer", "form", "title", "source_id", "soundfont", "youtube_title"]
# Words that carry no identity: a title made only of these could be anyone's piece.
GENERIC_WORDS = {"waltz", "valse", "walzer", "valzer", "nocturne", "notturno", "march", "marche", "marsch", "marcia",
                 "fugue", "fuga", "fuge", "no", "in", "op", "flat", "sharp", "major", "minor",
                 "a", "b", "c", "d", "e", "f", "g"}


def zip_stems(path):
    """The piece names inside a zip, without unpacking it."""
    with zipfile.ZipFile(path) as archive:
        return {unicodedata.normalize("NFC", Path(name).stem) for name in archive.namelist() if name.endswith(".mid")}


def known_work_ids():
    """Works already in the dataset: rendered works and the real test works."""
    known = set()
    for path in [DATA / "render_plan.csv", DATA / "splits" / "real_test.csv"]:
        with path.open(encoding="utf-8", newline="") as handle:
            known |= {row["work_id"] for row in csv.DictReader(handle)}
    return known


def youtube_names_someone_else(youtube_title, composer):
    """The YouTube title names a different composer. Hyphens become spaces so 'Chopin-' is found."""
    named_composers = find_composers(youtube_title.replace("-", " "))
    return bool(named_composers) and composer not in named_composers


def youtube_names_another_form(youtube_title, form):
    youtube_form, _, _ = label_title(youtube_title)
    return youtube_form is not None and youtube_form != form


def opus_of(title):
    """The catalog number of a title without any 'No.' part: 'op64n2' becomes 'op64'."""
    key = catalog_key(title)
    return re.sub(r"n\d+$", "", key) if key else None


def youtube_has_another_catalog_number(youtube_title, title):
    """'Grande valse, Op.54' on a video of 'Op. 18' is a different piece."""
    youtube_opus = opus_of(youtube_title)
    return youtube_opus is not None and youtube_opus != opus_of(title)


def title_is_generic(title):
    """'Nocturne No.12' could be anyone's piece, so it can't be verified."""
    words = re.split(r"[^a-z]+", fold(title))
    return all(word in GENERIC_WORDS or not word for word in words)


def title_is_paired_or_hybrid(title):
    """'Menuet et valse' or 'Etude-Nocturne' is only partly the labelled form."""
    folded = fold(title)
    return bool(re.search(r"\band\b|\bet\b|\bund\b|\betude\b|\bfantaisie\b|\bfantasie\b", folded))


def looks_doubtful(row, composer, form):
    youtube_title, title = row["youtube_title"], row["music"]
    return (youtube_names_someone_else(youtube_title, composer)
            or youtube_names_another_form(youtube_title, form)
            or youtube_has_another_catalog_number(youtube_title, title)
            or title_is_generic(title)
            or title_is_paired_or_hybrid(title))


def select_pieces():
    new_stems = zip_stems(FULL_ZIP) - zip_stems(CURATED_ZIP)
    known = known_work_ids()
    fonts = soundfont_names()
    chosen, dropped = {}, collections.Counter()
    csv.field_size_limit(10_000_000)
    with METADATA.open(encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda row: row["audio_name"])
    for row in rows:
        name = unicodedata.normalize("NFC", row["audio_name"].strip())
        if name not in new_stems:
            continue
        form, reasons, _ = label_title(row["music"])
        if form not in FINAL_CLASSES or reasons:
            continue
        composer = canonical_composer(row["surname"], row["firstname"])
        identifier = work_id(composer, form, row["music"])
        if identifier in known or identifier in chosen:
            continue
        if looks_doubtful(row, composer, form):
            dropped[form] += 1
            continue
        chosen[identifier] = {"work_id": identifier, "composer": composer, "form": form, "title": row["music"],
                              "source_id": row["audio_name"], "youtube_title": row["youtube_title"],
                              "soundfont": assign_soundfont(identifier, fonts)}
    return list(chosen.values()), dropped


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def known_wrong_work_ids():
    """Works that a person marked as wrong in either hand check of these pieces."""
    wrong = set()
    for name in ["audit_extra_round1.csv", "audit_extra_round2.csv"]:
        with (DATA / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["form_correct"] == "0" or row["composer_and_piece_correct"] == "0":
                    wrong.add(row["work_id"])
    return wrong


def main():
    candidates, dropped = select_pieces()
    wrong = known_wrong_work_ids()
    plan = [row for row in candidates if row["form"] in FORMS_TO_ADD and row["work_id"] not in wrong]
    write_csv(PLAN_PATH, plan)
    for form in FORMS_TO_ADD:
        print(f"{form:<10}{sum(row['form'] == form for row in plan):>5} works added")
    print(f"wrote {len(plan)} works to {PLAN_PATH.name}")


if __name__ == "__main__":
    main()
