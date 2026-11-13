"""Download the real recordings for the held-out test set, logging URL, license and checksum."""
import collections
import csv
import hashlib
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import AudioError, probe_duration, to_normalized_flac
from common import DATA, DiskGuardError, RateLimitedError, check_disk, download_stream
from select_phase2 import FINAL_CLASSES, MAX_FILE_MB, MAX_SECONDS, MIN_SECONDS

PER_CLASS = 40
ORIGINAL_DIR = DATA / "audio" / "real_original"
OUTPUT_DIR = DATA / "audio" / "real"
LOG_PATH = DATA / "real_download_log.csv"
COLUMNS = ["rank", "work_pass", "work_id", "composer", "form", "title", "source", "source_id", "url", "license", "license_url",
           "instrumentation", "recording_year", "original_path", "sha256", "bytes", "flac_path", "duration_s",
           "input_lufs", "status", "error"]
SUFFIXES = {".ogg", ".oga", ".flac", ".wav", ".mp3", ".m4a", ".opus", ".aif", ".aiff", ".webm"}


def recording_id(row):
    return hashlib.sha256(f"{row['source']}:{row['source_id']}".encode()).hexdigest()[:20]


def fetch_one(row):
    identifier = recording_id(row)
    suffix = Path(row["url"].split("?")[0]).suffix.lower()
    original = ORIGINAL_DIR / f"{identifier}{suffix if suffix in SUFFIXES else '.audio'}"
    flac = OUTPUT_DIR / f"{identifier}.flac"
    result = {**row, "original_path": str(original.relative_to(DATA)), "flac_path": str(flac.relative_to(DATA)),
              "sha256": "", "bytes": "", "duration_s": "", "input_lufs": "", "status": "failed", "error": ""}
    try:
        digest, size, _ = download_stream(row["url"], original, int(MAX_FILE_MB * 1e6))
        result.update(sha256=digest, bytes=size)
        duration = probe_duration(original)
        if not MIN_SECONDS <= duration <= MAX_SECONDS:
            raise AudioError(f"duration out of range: {duration:.0f}s")
        loudness, final_duration = to_normalized_flac(original, flac)
        result.update(duration_s=f"{final_duration:.2f}", input_lufs=f"{loudness:.2f}", status="ok")
    except (AudioError, urllib.error.URLError, RuntimeError, OSError) as error:
        if isinstance(error, DiskGuardError):
            raise
        original.unlink(missing_ok=True)
        flac.unlink(missing_ok=True)
        result["error"] = str(error)[:300]
    return result


def write_log(rows):
    temporary = LOG_PATH.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())
    temporary.replace(LOG_PATH)


def main():
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA / "real_test_candidates.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    log = {}
    if LOG_PATH.exists():
        with LOG_PATH.open(encoding="utf-8", newline="") as handle:
            log = {(row["source"], row["source_id"]): row for row in csv.DictReader(handle)}
    for form in FINAL_CLASSES:
        ranked = sorted((row for row in candidates if row["form"] == form), key=lambda row: int(row["rank"]))
        count = 0
        for row in ranked:
            if count >= PER_CLASS:
                break
            key = (row["source"], row["source_id"])
            if log.get(key, {}).get("status") == "ok":
                count += 1
                continue
            if key in log:
                continue  # A recorded failure is not retried; the next-ranked candidate replaces it.
            check_disk()
            try:
                result = fetch_one(row)
            except RateLimitedError as error:
                print(f"STOPPED: {error}. Run this script again later to continue.", flush=True)
                sys.exit(2)
            log[key] = result
            write_log(log)
            if result["status"] == "ok":
                count += 1
                print(f"{form} {count}/{PER_CLASS}: {row['title'][:60]}", flush=True)
            else:
                print(f"{form} FAILED: {row['title'][:50]}: {result['error'][:120]}", flush=True)
        print(f"== {form}: {count} of {PER_CLASS}", flush=True)
    ok = [row for row in log.values() if row["status"] == "ok"]
    print(f"ok {len(ok)}; failed {len(log) - len(ok)}; originals {sum(int(row['bytes']) for row in ok)/1e9:.2f} GB")
    print(collections.Counter(row["form"] for row in ok))


if __name__ == "__main__":
    main()
