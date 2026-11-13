"""Build data/manifest_candidates.csv and data/review.csv from the raw metadata."""
import collections
import csv
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RAW
from composers import COMPOSER_MAP, find_composers
from labeling import clean_filename, fold, instrumentation_from_text, label_title, normalize_license
from work_ids import work_id

COLUMNS = ["work_id", "composer", "form", "title", "source", "source_id", "url", "license", "license_url",
           "instrumentation", "recording_year", "is_midi", "needs_review"]
# Flagged rows stay in the manifest. Ambiguous rows have no trustworthy form, so they go to review only.
FLAG_ONLY = {"multi_piece", "arrangement", "medley", "pd_recording_uncertain", "composer_unspecified", "multiple_composers",
             "work_id_unresolved"}
GIANTMIDI_LICENSE = ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/")
MUSICNET_LICENSE = ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/")
MAESTRO_LICENSE = ("CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/")
CURRENT_PD_RECORDING_YEAR = 1926  # US sound recordings published before 1926 are public domain as of 2026.

excluded = collections.defaultdict(collections.Counter)


def unaccent(text):
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).strip()


def canonical_composer(surname, first):
    """Merge spellings of a known composer only when the first initial agrees (Clara is not Robert)."""
    surname = re.sub(r"\s*(?:Jr\.?|Sr\.?|II|I)$", "", unaccent(surname)).strip(" ,")
    first = unaccent(first)
    mapped = COMPOSER_MAP.get(fold(surname))
    if mapped and fold(first)[:1] == fold(mapped.split(", ")[1])[:1]:
        return mapped
    return f"{surname}, {first}".strip(", ")


def year_of(text):
    match = re.search(r"\b(1[6-9]\d\d|20\d\d)\b", str(text or ""))
    return match.group(1) if match else ""


def strip_html(text):
    return re.sub(r"<[^>]+>", " ", str(text or ""))


def make_row(composer, form, title, source, source_id, url, license_pair, instrumentation, year, is_midi, reasons):
    return {
        "work_id": work_id(composer, form, title), "composer": composer, "form": form, "title": title,
        "source": source, "source_id": source_id, "url": url, "license": license_pair[0],
        "license_url": license_pair[1], "instrumentation": instrumentation, "recording_year": year,
        "is_midi": int(is_midi), "needs_review": int(bool(reasons)), "reasons": ";".join(sorted(set(reasons))),
    }


def giantmidi_zip_stems():
    """Piece names actually present in the curated zip, which the zip is never unpacked to read."""
    with zipfile.ZipFile(RAW / "giantmidi" / "surname_checked_midis_v1.2.zip") as archive:
        names = [Path(name).stem for name in archive.namelist() if name.endswith(".mid")]
    return {unicodedata.normalize("NFC", name) for name in names}


def giantmidi_rows():
    path = RAW / "giantmidi" / "full_music_pieces_youtube_similarity_pianosoloprob.csv"
    in_zip = giantmidi_zip_stems()
    matched = set()
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = unicodedata.normalize("NFC", row["audio_name"].strip())
            if name not in in_zip:
                continue
            matched.add(name)
            form, reasons, reason_out = label_title(row["music"])
            if not form:
                excluded["giantmidi"][reason_out] += 1
                continue
            composer = canonical_composer(row["surname"], row["firstname"])
            yield make_row(composer, form, row["music"], "giantmidi", row["audio_name"],
                           "https://github.com/bytedance/GiantMIDI-Piano", GIANTMIDI_LICENSE, "piano", "", True, reasons)
    print(f"GiantMIDI zip: {len(in_zip)} pieces, {len(matched)} matched to metadata, {len(in_zip - matched)} unmatched")


def musicnet_rows():
    with (RAW / "musicnet" / "musicnet_metadata.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            movement = re.sub(r"^\d+\.\s*", "", row["movement"])
            form, reasons, reason_out = label_title(movement)
            if not form:
                excluded["musicnet"][reason_out] += 1
                continue
            composition = row["composition"]
            if not re.search(r"\bWTK\b", composition):
                reasons = reasons + ["movement_of_larger_work"]
            title = f"{composition}: {movement}"
            found = find_composers(row["composer"])
            composer = found[0] if found else unaccent(row["composer"])
            identifier = work_id(composer, form, f"{row['catalog_name']} {movement}")
            result = make_row(composer, form, title, "musicnet", row["id"], "https://zenodo.org/records/5120004",
                              MUSICNET_LICENSE, row["ensemble"].lower(), "", False, reasons)
            result["work_id"] = identifier
            yield result


def maestro_rows():
    with (RAW / "maestro" / "maestro-v3.0.0.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            form, reasons, reason_out = label_title(row["canonical_title"])
            if not form:
                excluded["maestro"][reason_out] += 1
                continue
            names = row["canonical_composer"].split(" / ")
            if len(names) > 1:
                reasons = reasons + ["arrangement"]
            parts = names[0].split()
            composer = canonical_composer(parts[-1], " ".join(parts[:-1]))
            yield make_row(composer, form, row["canonical_title"], "maestro", row["midi_filename"],
                           "https://magenta.withgoogle.com/datasets/maestro", MAESTRO_LICENSE, "piano", row["year"], True, reasons)


def pick_composer(primary, secondary, reasons, creator=""):
    found = find_composers(primary) or find_composers(secondary) or find_composers(creator, creator_style=True)
    if not found:
        return None
    if len(found) > 1:
        reasons.append("multiple_composers")
    if found[0].endswith("(unspecified)"):
        reasons.append("composer_unspecified")
    return found[0]


def commons_rows():
    with (RAW / "commons" / "files.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            meta = record["meta"]
            license_name = normalize_license(meta.get("LicenseShortName", ""), meta.get("LicenseUrl", ""))
            if not license_name:
                excluded["commons"]["license_not_allowed"] += 1
                continue
            title = clean_filename(record["title"])
            form, reasons, reason_out = label_title(title)
            if not form:
                excluded["commons"][reason_out] += 1
                continue
            secondary = " ; ".join([strip_html(meta.get("ObjectName")), strip_html(meta.get("Categories"))])
            composer = pick_composer(title, secondary, reasons)
            if not composer:
                excluded["commons"]["no_known_composer"] += 1
                continue
            if record.get("duration") and record["duration"] < 30:
                reasons.append("excerpt_or_demo")
            is_midi = "midi" in record.get("mime", "")
            instrument = instrumentation_from_text(f"{title} {secondary}")
            yield make_row(composer, form, title, "commons", str(record["pageid"]), record["url"],
                           (license_name, meta.get("LicenseUrl", "")), instrument,
                           year_of(strip_html(meta.get("DateTimeOriginal"))), is_midi, reasons)


def first(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def ia_rows():
    path = RAW / "ia" / "items.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            link = first(item.get("licenseurl_item")) or first(item.get("licenseurl"))
            license_name = normalize_license("", link)
            if not license_name:
                excluded["ia"]["license_not_allowed"] += 1
                continue
            identifier, item_title = item["identifier"], str(first(item.get("title")))
            creator = str(first(item.get("creator")))
            files = item.get("files", [])
            if not files:
                excluded["ia"]["no_audio_files"] += 1
                continue
            year = year_of(item.get("date"))
            for entry in files:
                file_title = str(entry.get("title") or clean_filename(entry["name"]))
                title = item_title if len(files) == 1 else file_title
                form, reasons, reason_out = label_title(title)
                if not form:
                    excluded["ia"][reason_out] += 1
                    continue
                composer = pick_composer(title, item_title, reasons, f"{creator} ; {entry.get('creator', '')}")
                if not composer:
                    excluded["ia"]["no_known_composer"] += 1
                    continue
                if license_name == "Public domain" and (not year or int(year) >= CURRENT_PD_RECORDING_YEAR):
                    reasons.append("pd_recording_uncertain")
                url = f"https://archive.org/download/{identifier}/{entry['name']}"
                instrument = instrumentation_from_text(f"{title} {item_title} {creator}")
                yield make_row(composer, form, title, "internet_archive", f"{identifier}/{entry['name']}", url,
                               (license_name, link), instrument, year, False, reasons)


def flag_unresolved_works(rows):
    """Flag IDs that may split one work in two, which would leak a work across splits."""
    numbered = {row["work_id"].rsplit("n", 1)[0] for row in rows if re.search(r"\|op\d+[a-z]?n\d+$", row["work_id"])}
    for row in rows:
        catalog = row["work_id"].split("|")[2]
        no_catalog = row["composer"].startswith("Chopin") and not re.match(r"(?:op\d|b\d|kk)", catalog)
        if no_catalog or row["work_id"] in numbered:
            reasons = set(filter(None, row["reasons"].split(";"))) | {"work_id_unresolved"}
            row["reasons"] = ";".join(sorted(reasons))
            row["needs_review"] = 1


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_rows = list(giantmidi_rows()) + list(musicnet_rows()) + list(commons_rows()) + list(ia_rows())
    maestro = list(maestro_rows())
    flag_unresolved_works(all_rows)
    manifest, review = [], []
    for row in all_rows:
        reasons = set(filter(None, row["reasons"].split(";")))
        if reasons:
            review.append(row)
        if reasons <= FLAG_ONLY:
            manifest.append(row)
    review += [{**row, "source": "maestro_nc_reference"} for row in maestro if row["reasons"]]
    write_csv(DATA / "manifest_candidates.csv", manifest, COLUMNS)
    write_csv(DATA / "review.csv", review, COLUMNS + ["reasons"])
    write_csv(DATA / "maestro_nc_reference.csv", maestro, COLUMNS + ["reasons"])
    summary = {source: dict(counter) for source, counter in excluded.items()}
    (DATA / "excluded_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"manifest {len(manifest)} rows; review {len(review)} rows; maestro reference {len(maestro)} rows")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
