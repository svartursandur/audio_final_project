"""Print per-source and combined viability tables; draw the label audit sample."""
import collections
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA
from labeling import CLASSES

MIN_WORKS, MIN_COMPOSERS, MAX_TOP_SHARE = 30, 5, 0.40
AUDIT_PER_CLASS, AUDIT_SEED = 30, 20261109


def load(name):
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stats(rows):
    """Distinct works per class, with each work credited to one composer."""
    works = {}
    for row in rows:
        works.setdefault(row["work_id"], row)
    composers = collections.Counter(row["composer"] for row in works.values())
    total = sum(composers.values())
    top_name, top_count = composers.most_common(1)[0] if composers else ("-", 0)
    return total, len(composers), (top_count / total if total else 0.0), top_name


def table(title, rows):
    print(f"\n{title}")
    print(f"{'class':<10}{'works':>7}{'composers':>11}{'top share':>11}  {'top composer':<30}viable")
    for form in CLASSES:
        total, composers, share, name = stats([row for row in rows if row["form"] == form])
        viable = total >= MIN_WORKS and composers >= MIN_COMPOSERS and share <= MAX_TOP_SHARE
        print(f"{form:<10}{total:>7}{composers:>11}{share:>10.0%}  {name:<30}{'YES' if viable else 'no'}")


def main():
    manifest = load("manifest_candidates.csv")
    clean = [row for row in manifest if row["needs_review"] == "0"]
    for source, label in [("giantmidi", "GiantMIDI curated zip"), ("musicnet", "MusicNet"),
                          ("commons", "Wikimedia Commons"), ("internet_archive", "Internet Archive")]:
        table(label, [row for row in clean if row["source"] == source])
    maestro = [row for row in load("maestro_nc_reference.csv") if not row["reasons"]]
    table("MAESTRO (CC BY-NC-SA: reference only, excluded from combined)", maestro)
    table("COMBINED, all allowed sources", clean)
    table("COMBINED, without GiantMIDI", [row for row in clean if row["source"] != "giantmidi"])
    real = [row for row in clean if row["is_midi"] == "0"]
    table("REAL AUDIO ONLY (candidates for the held-out test set)", real)

    midi_works = {row["work_id"] for row in clean if row["is_midi"] == "1"}
    print("\nReal-audio works also present as MIDI (these must be held out of training):")
    for form in CLASSES:
        real_works = {row["work_id"] for row in real if row["form"] == form}
        print(f"  {form:<10}{len(real_works & midi_works):>5} of {len(real_works)}")

    print(f"\nFlagged rows (needs_review=1, not counted): {len(manifest) - len(clean)}")

    if (DATA / "audit_round2_sample.csv").exists():
        return  # Keep the sample that was hand-audited; redrawing it would orphan the verdicts.
    audited = {(row["source"], row["title"]) for row in load("audit_round1.csv")}
    random.seed(AUDIT_SEED)
    sample = []
    for form in CLASSES:
        pool = [row for row in clean if row["form"] == form and (row["source"], row["title"]) not in audited]
        sample += random.sample(pool, min(AUDIT_PER_CLASS, len(pool)))
    with (DATA / "audit_round2_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["form", "composer", "title", "source", "instrumentation", "work_id"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample)
    print(f"wrote data/audit_round2_sample.csv with {len(sample)} rows")


if __name__ == "__main__":
    main()
