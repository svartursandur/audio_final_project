"""Pick extra training pieces from ATEPP, a second source of transcribed piano performances.

ATEPP (CC BY 4.0) is MIDI transcribed from real piano recordings, like GiantMIDI, so it
sounds the same once rendered. Our fugue class is the smallest, so it helps most there.
Other forms are added too, so the model can't learn "ATEPP means fugue".

Its track names look like "work: movement", for example
"The Well-Tempered Clavier, Book I, Prelude and Fugue No. 16 in G Minor, BWV 861: Fugue".

Rules, in order:
  1. Anything with "variations" is left out (Diabelli's "Variations on a Waltz" are not waltzes).
  2. If the movement (after the last colon) names a form, that is the form. A fugue only
     counts if the movement starts with the word, so "Adagio - Fuga" is left out as mixed.
  3. If the movement is only a number ("4 Fugen, Op. 72: No. 2"), the collection's title
     gives the form. Any other movement name ("I. Grave") is left out.
  4. Titles that our usual rules exclude or send to review are left out.
  5. A work already in our dataset (any split) or among the real test recordings is
     skipped, so no work crosses into validation or test.
  6. One performance per work, and at most 12 works per composer per form.

Extra pieces only ever join training (see experiment_step2.composer_scores).

Usage:
    python3 scripts/select_atepp.py
"""
import csv
import io
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT
from labeling import fold, label_title
from render_midi import MIDI_DIR, stem_id
from select_phase2 import assign_soundfont, soundfont_names
from work_ids import work_id

ATEPP_ZIP = ROOT / "ATEPP-1.1.zip"
METADATA = "ATEPP-metadata-1.1.csv"
FORMS = ["waltz", "nocturne", "march", "fugue"]
MAX_PER_COMPOSER_AND_FORM = 12
PLAN_PATH = DATA / "atepp_plan.csv"
LICENSE = "CC BY 4.0"

STARTS_WITH_NUMBER = r"^\s*(no\.?\s*)?[ivxlc\d]+\b"
STARTS_WITH_FUGUE = r"^\s*((no\.?\s*)?[ivxlc\d]+\.?\s*)?(fugue|fuga|fuge)\b"


def read_metadata():
    with zipfile.ZipFile(ATEPP_ZIP) as archive, archive.open(METADATA) as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))


def form_of(track):
    """Return (form, None) or (None, reason it was left out), following rules 1 to 4."""
    if re.search(r"\bvariation|\bvar\.", fold(track)):
        return None, "variations"
    if re.search(r"\bseasons\b", fold(track)):
        return None, "march_is_the_month"  # Tchaikovsky's The Seasons: "March" is the month
    movement = track.rsplit(":", 1)[-1] if ":" in track else track
    if re.search(r"^\s*\d+\s*-\s*\d+\.", movement) or re.search(r"[a-z]+\s*-\s*[a-z]+\s*-\s*[a-z]+", fold(movement)):
        return None, "several_pieces_in_one_track"  # "Carnaval: 16-18. Valse allemande-Paganini-Aveu"
    form, reasons, excluded = label_title(movement)
    if form in FORMS and not excluded:
        if form == "fugue" and not re.search(STARTS_WITH_FUGUE, fold(movement)):
            return None, "fugue_inside_movement"
    elif ":" in track and not re.search(STARTS_WITH_NUMBER, fold(movement)):
        return None, "movement_not_named_form"
    else:
        form, reasons, excluded = label_title(track)
    if excluded or form not in FORMS:
        return None, excluded or "no_form"
    if reasons:
        return None, "needs_review"
    return form, None


def same_work_id(identifier, title):
    """Fix IDs that a hand check found wrong: the same work under another name.

    Without this, a work could be added twice, or slip past the check against
    validation, test and the real test recordings.
    """
    text = fold(title)
    composer, form, _ = identifier.split("|", 2)
    number = re.search(r"no\.?\s*(\d+)", text)
    # The same works as real test recordings, which carry other names there.
    if composer == "chopin-frederic" and form == "fugue":
        return IN_REAL_TEST  # "Fugue in A minor B. 144"
    if composer == "ravel-maurice" and (form == "fugue" or "la valse" in text):
        return IN_REAL_TEST  # "Le Tombeau de Couperin. II. Fugue" and "La Valse for two pianos"
    if composer == "beethoven-ludwig-van" and form == "fugue" and "110" in text:
        return IN_REAL_TEST  # "Klaviersonate Nr. 31 op. 110 - IV. Fuga"
    # The same work under several names within ATEPP or our data.
    if composer == "chopin-frederic" and form == "waltz" and number and 14 <= int(number.group(1)) <= 19:
        # The posthumous waltzes are numbered 14 to 19 in some editions and by B. numbers in ours.
        return f"chopin-frederic|waltz|{CHOPIN_WALTZ_NUMBERS[int(number.group(1))]}"
    if composer == "franck-cesar" and form == "fugue":
        return "franck-cesar|fugue|fwv21"  # Prelude, Chorale and Fugue, under six spellings
    if composer == "shostakovich-dmitri" and "op87" in identifier:
        fugue_number = re.search(r"fugue\s*no\.?\s*(\d+)", text) or re.search(r"op\.?\s*87\s*/\s*(\d+)", text)
        return f"shostakovich-dmitri|fugue|op87n{fugue_number.group(1)}" if fugue_number else UNCLEAR_WHICH_WORK
    if composer == "liszt-franz" and "oubli" in text:
        numbers = re.findall(r"no\.?\s*(\d)", text)
        return f"liszt-franz|waltz|s215n{numbers[-1]}" if numbers else UNCLEAR_WHICH_WORK
    if composer == "liszt-franz" and form == "nocturne" and "impromptu" in text:
        return "liszt-franz|nocturne|s191"
    if composer == "schumann-robert" and "davidsbundler" in text:
        return "schumann-robert|march|op9n21"
    if composer == "schumann-robert" and "bunte" in text and form == "march":
        # Bunte Blätter has two marches: No. 11 "Marsch" and No. 14 "Geschwindmarsch".
        return "schumann-robert|march|" + ("op99n14" if "geschwind" in text else "op99n11")
    return identifier


IN_REAL_TEST = "same work as a real test recording"
UNCLEAR_WHICH_WORK = "unclear which work"


# Chopin's posthumous waltzes: edition number -> the B. number our data uses.
CHOPIN_WALTZ_NUMBERS = {14: "b56", 15: "b44", 16: "b21", 17: "b46", 18: "b133", 19: "b150"}

# Left out on purpose. The real test has a recording titled only "Bach-Fuga-C-Moll",
# which might be this fugue, so it is safer not to train on it.
LEFT_OUT_BY_HAND = {"bach-johann-sebastian|fugue|bwv847"}


def our_composer_names():
    """Composer names already in our data, so ATEPP's names can be written the same way."""
    names = set()
    for split in ["random.csv", "real_test.csv"]:
        with (DATA / "splits" / split).open(encoding="utf-8", newline="") as handle:
            names |= {row["composer"] for row in csv.DictReader(handle)}
    return names


def to_our_name(atepp_name, known):
    """'Johann Sebastian Bach' -> 'Bach, Johann Sebastian', reusing our spelling when it matches."""
    words = atepp_name.split()
    surname, given = words[-1], " ".join(words[:-1])
    for name in known:
        known_surname, _, known_given = name.partition(", ")
        if fold(known_surname) == fold(surname) and fold(known_given).startswith(fold(words[0])):
            return name
    return f"{surname}, {given}"


def works_to_avoid():
    """Every work_id already in our dataset or among the real test recordings."""
    avoid = set()
    for split in ["random.csv", "real_test.csv"]:
        with (DATA / "splits" / split).open(encoding="utf-8", newline="") as handle:
            avoid |= {row["work_id"] for row in csv.DictReader(handle)}
    return avoid


def main():
    rows = read_metadata()
    known = our_composer_names()
    avoid = works_to_avoid()
    left_out = Counter()
    performances = defaultdict(list)
    for row in rows:
        form, reason = form_of(row["track"])
        if reason:
            left_out[reason] += 1
            continue
        composer = to_our_name(row["composer"], known)
        identifier = same_work_id(work_id(composer, form, row["track"]), row["track"])
        if identifier in (IN_REAL_TEST, UNCLEAR_WHICH_WORK):
            left_out[identifier] += 1
            continue
        if identifier in avoid:
            left_out["already_in_our_data"] += 1
            continue
        if identifier in LEFT_OUT_BY_HAND:
            left_out["left_out_by_hand"] += 1
            continue
        performances[identifier].append({**row, "form": form, "our_composer": composer})

    chosen, per_composer = [], Counter()
    for identifier in sorted(performances):
        performance = sorted(performances[identifier], key=lambda row: row["midi_path"])[0]
        key = (performance["our_composer"], performance["form"])
        if per_composer[key] >= MAX_PER_COMPOSER_AND_FORM:
            left_out["over_composer_limit"] += 1
            continue
        per_composer[key] += 1
        chosen.append((identifier, performance))

    fonts = soundfont_names()
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    if PLAN_PATH.exists():  # remove MIDI files from an earlier run, so only this selection is left
        with PLAN_PATH.open(encoding="utf-8", newline="") as handle:
            for old in csv.DictReader(handle):
                (MIDI_DIR / f"{stem_id(old['work_id'])}.mid").unlink(missing_ok=True)
    plan = []
    with zipfile.ZipFile(ATEPP_ZIP) as archive:
        # Folder names in the zip are spelled differently from the metadata ("Frederic" vs
        # "Frédéric"), but every MIDI file has a unique numbered name, so look it up by that.
        by_file_name = {Path(name).name: name for name in archive.namelist() if name.endswith((".mid", ".midi"))}
        for identifier, performance in chosen:
            member = by_file_name.get(Path(performance["midi_path"]).name)
            if member is None:
                left_out["midi_missing_from_zip"] += 1
                continue
            data = archive.read(member)
            if data[:4] != b"MThd":
                left_out["not_a_midi_file"] += 1
                continue
            (MIDI_DIR / f"{stem_id(identifier)}.mid").write_bytes(data)
            plan.append({"work_id": identifier, "composer": performance["our_composer"], "form": performance["form"],
                         "title": performance["track"], "source_id": f"atepp:{performance['midi_path']}",
                         "soundfont": assign_soundfont(identifier, fonts), "artist": performance["artist"],
                         "license": LICENSE})

    with PLAN_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan[0]))
        writer.writeheader()
        writer.writerows(plan)

    print_pieces_to_check(plan)
    print(f"\n{len(rows)} ATEPP performances; left out: {dict(left_out.most_common())}")
    print(f"chose {len(plan)} works, one performance each, saved to {PLAN_PATH.relative_to(ROOT)}")
    for form in FORMS:
        form_rows = [row for row in plan if row["form"] == form]
        composers = Counter(row["composer"] for row in form_rows)
        print(f"  {form:<9} {len(form_rows):>3} works, {len(composers)} composers: {dict(composers.most_common())}")


def print_pieces_to_check(plan):
    """List chosen pieces whose composer and form also appear outside training.

    IDs catch most overlaps, but a work can hide under another name, so these are
    the ones to compare by eye.
    """
    outside_training = defaultdict(list)
    for split in ["random.csv", "real_test.csv"]:
        with (DATA / "splits" / split).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split", "real") != "train":
                    outside_training[(row["composer"], row["form"])].append(row["title"][:60])
    print("Chosen pieces to compare by eye with validation, test and real test pieces:")
    for row in plan:
        others = outside_training.get((row["composer"], row["form"]))
        if others:
            print(f"  {row['title'][:70]}\n      vs {others}")


if __name__ == "__main__":
    main()
