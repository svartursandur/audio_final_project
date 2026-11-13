"""Write work-level random splits, leave-composers-out folds, and the real-audio test list."""
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RAW
from select_phase2 import FINAL_CLASSES

SPLITS = DATA / "splits"
SEED = 20261109
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15
FOLDS = 5
BASE_COLUMNS = ["work_id", "composer", "form", "title", "soundfont", "audio_path", "midi_path"]
REAL_COLUMNS = ["work_id", "composer", "form", "title", "author", "source", "source_id", "url", "license", "license_url",
                "instrumentation", "recording_year", "audio_path", "original_path", "sha256", "duration_s", "repeat_of_work"]


def load(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def author_lookup():
    """Map each real recording to the author its source credits, for license attribution."""
    authors = {}
    with (RAW / "commons" / "files.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            without_html = re.sub(r"<[^>]+>", " ", record["meta"].get("Artist", ""))
            authors[("commons", str(record["pageid"]))] = " ".join(without_html.split())
    with (RAW / "ia" / "items.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            creator = item.get("creator", "")
            if isinstance(creator, list):
                creator = "; ".join(creator)
            for entry in item["files"]:
                authors[("internet_archive", f"{item['identifier']}/{entry['name']}")] = creator
    return authors


def random_split(rows):
    rng = random.Random(SEED)
    output = []
    for form in FINAL_CLASSES:
        works = sorted((row for row in rows if row["form"] == form), key=lambda row: row["work_id"])
        rng.shuffle(works)
        train_end = round(len(works) * TRAIN_FRACTION)
        val_end = train_end + round(len(works) * VAL_FRACTION)
        for index, row in enumerate(works):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            output.append({**row, "split": split})
    return output


def composer_folds(rows):
    """Greedily place whole composers so that each fold has a similar class mix."""
    rng = random.Random(SEED)
    per_composer = collections.defaultdict(collections.Counter)
    for row in rows:
        per_composer[row["composer"]][row["form"]] += 1
    composers = sorted(per_composer)
    rng.shuffle(composers)
    composers.sort(key=lambda name: sum(per_composer[name].values()), reverse=True)
    totals = [collections.Counter() for _ in range(FOLDS)]
    assignment = {}
    for name in composers:
        def cost(fold):
            return sum((totals[fold][form] + count) ** 2 for form, count in per_composer[name].items())
        best = min(range(FOLDS), key=lambda fold: (cost(fold), sum(totals[fold].values())))
        assignment[name] = best
        totals[best].update(per_composer[name])
    return [{**row, "fold": assignment[row["composer"]]} for row in rows], totals


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    rendered = [row for row in load(DATA / "render_log.csv") if row["status"] == "ok"]
    real = [row for row in load(DATA / "real_download_log.csv") if row["status"] == "ok"]
    held_out = {row["work_id"] for row in real}
    overlap = [row for row in rendered if row["work_id"] in held_out]
    usable = [{**row, "audio_path": row["flac_path"]} for row in rendered if row["work_id"] not in held_out]

    random_rows = random_split(usable)
    fold_rows, totals = composer_folds(usable)
    authors = author_lookup()
    real_rows = [{**row, "audio_path": row["flac_path"], "repeat_of_work": int(row["work_pass"] != "0"),
                  "author": authors.get((row["source"], row["source_id"]), "")} for row in real]
    write(SPLITS / "random.csv", random_rows, BASE_COLUMNS + ["split"])
    write(SPLITS / "composer_holdout.csv", fold_rows, BASE_COLUMNS + ["fold"])
    write(SPLITS / "real_test.csv", real_rows, REAL_COLUMNS)

    # Invariants: a work has one split, a composer has one fold, no test work is trained on.
    assert len({row["work_id"] for row in random_rows}) == len(random_rows), "work appears twice in random split"
    fold_of = collections.defaultdict(set)
    for row in fold_rows:
        fold_of[row["composer"]].add(row["fold"])
    assert all(len(folds) == 1 for folds in fold_of.values()), "composer crosses folds"
    assert not ({row["work_id"] for row in random_rows} & held_out), "real-test work present in MIDI split"

    print(f"MIDI works rendered {len(rendered)}; removed for real-test overlap {len(overlap)}; usable {len(usable)}")
    print(f"{'class':<10}{'train':>7}{'val':>6}{'test':>6}" + "".join(f"{'fold' + str(i):>7}" for i in range(FOLDS)))
    for form in FINAL_CLASSES:
        counts = collections.Counter(row["split"] for row in random_rows if row["form"] == form)
        print(f"{form:<10}{counts['train']:>7}{counts['val']:>6}{counts['test']:>6}" + "".join(f"{totals[i][form]:>7}" for i in range(FOLDS)))
    print(f"composers per fold: {collections.Counter(next(iter(f)) for f in fold_of.values())}")
    print("\nreal test set")
    print(f"{'class':<10}{'recordings':>11}{'works':>7}{'composers':>11}{'top share':>11}  instrumentation")
    for form in FINAL_CLASSES:
        rows = [row for row in real_rows if row["form"] == form]
        works = {}
        for row in rows:
            works.setdefault(row["work_id"], row["composer"])
        composers = collections.Counter(works.values())
        share = composers.most_common(1)[0][1] / len(works) if works else 0
        print(f"{form:<10}{len(rows):>11}{len(works):>7}{len(composers):>11}{share:>10.0%}  "
              f"{collections.Counter(row['instrumentation'] for row in rows).most_common(4)}")


if __name__ == "__main__":
    main()
