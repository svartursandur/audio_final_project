"""Plan the MIDI renders and rank real recordings for the held-out test set."""
import collections
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RAW
from labeling import fold

FINAL_CLASSES = ["waltz", "nocturne", "march", "fugue"]
SEED = 20261109
MIN_SECONDS, MAX_SECONDS = 30, 1200
MAX_FILE_MB = 150
INSTRUMENT_PRIORITY = {"piano": 0, "solo piano": 0, "unknown": 1}
PLAN_COLUMNS = ["work_id", "composer", "form", "title", "source_id", "soundfont"]
REAL_COLUMNS = ["rank", "work_pass", "work_id", "composer", "form", "title", "source", "source_id", "url", "license",
                "license_url", "instrumentation", "recording_year", "size_mb", "duration_s"]


def load_clean():
    with (DATA / "manifest_candidates.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["needs_review"] == "0" and row["form"] in FINAL_CLASSES]
    return rows


def soundfont_names():
    with (DATA / "soundfonts" / "soundfonts.csv").open(encoding="utf-8", newline="") as handle:
        return [row["name"] for row in csv.DictReader(handle)]


def assign_soundfont(work_id, names):
    """Depends only on the work_id hash, so it is independent of class."""
    return names[int(hashlib.sha256(f"soundfont-v1:{work_id}".encode()).hexdigest(), 16) % len(names)]


def plan_renders(rows):
    names = soundfont_names()
    works = {}
    for row in sorted((r for r in rows if r["source"] == "giantmidi"), key=lambda r: r["source_id"]):
        works.setdefault(row["work_id"], row)
    plan = [{**row, "soundfont": assign_soundfont(work, names)} for work, row in works.items()]
    with (DATA / "render_plan.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plan)
    table = collections.Counter((row["form"], row["soundfont"]) for row in plan)
    print(f"render plan: {len(plan)} works")
    for form in FINAL_CLASSES:
        print(f"  {form:<10}" + "  ".join(f"{name}={table[(form, name)]}" for name in names))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def file_details():
    details = {}
    with (RAW / "commons" / "files.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            details[("commons", str(record["pageid"]))] = (number(record.get("size")) / 1e6, number(record.get("duration")))
    with (RAW / "ia" / "items.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            for entry in item["files"]:
                key = ("internet_archive", f"{item['identifier']}/{entry['name']}")
                details[key] = (number(entry.get("size")) / 1e6, number(entry.get("length")))
    return details


def recording_order(row):
    return INSTRUMENT_PRIORITY.get(row["instrumentation"], 2), row["size_mb"] or 1e9, row["source_id"]


def rank_class(rows, rng):
    """One recording per work first, cycling through composers; further recordings only afterwards."""
    by_work = collections.defaultdict(list)
    for row in rows:
        by_work[row["work_id"]].append(row)
    for work, recordings in by_work.items():
        unique, seen = [], set()
        for row in sorted(recordings, key=recording_order):
            key = re.sub(r"[^a-z0-9]+", "", fold(row["title"]))
            if key not in seen:
                seen.add(key)
                unique.append(row)
        by_work[work] = unique
    ranked, work_pass = [], 0
    while any(len(recordings) > work_pass for recordings in by_work.values()):
        by_composer = collections.defaultdict(list)
        for work, recordings in sorted(by_work.items()):
            if len(recordings) > work_pass:
                by_composer[recordings[0]["composer"]].append(recordings[work_pass])
        composers = sorted(by_composer)
        rng.shuffle(composers)
        for items in by_composer.values():
            rng.shuffle(items)
        while any(by_composer[name] for name in composers):
            for name in composers:
                if by_composer[name]:
                    ranked.append({**by_composer[name].pop(), "work_pass": work_pass})
        work_pass += 1
    return ranked


def plan_real(rows):
    details = file_details()
    rng = random.Random(SEED)
    output, dropped = [], collections.Counter()
    for form in FINAL_CLASSES:
        usable = []
        for row in rows:
            if row["form"] != form or row["is_midi"] != "0" or row["source"] == "musicnet":
                continue
            size, duration = details.get((row["source"], row["source_id"]), (0.0, 0.0))
            if duration and not MIN_SECONDS <= duration <= MAX_SECONDS:
                dropped["duration_out_of_range"] += 1
                continue
            if size > MAX_FILE_MB:
                dropped["file_too_large"] += 1
                continue
            usable.append({**row, "size_mb": round(size, 2), "duration_s": round(duration, 1)})
        ranked = rank_class(usable, rng)
        output += [{**row, "rank": rank} for rank, row in enumerate(ranked, 1)]
        print(f"{form:<10}{len(ranked):>5} ranked recordings, {len({row['work_id'] for row in ranked})} works")
    with (DATA / "real_test_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    print("dropped before ranking:", dict(dropped))


def main():
    rows = load_clean()
    plan_renders(rows)
    plan_real(rows)


if __name__ == "__main__":
    main()
