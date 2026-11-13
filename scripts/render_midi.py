"""Render planned GiantMIDI works with FluidSynth to loudness-normalized 32 kHz mono FLAC."""
import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import SAMPLE_RATE, AudioError, to_normalized_flac
from common import DATA, RAW, DiskGuardError, check_disk

ZIP_PATH = RAW / "giantmidi" / "midis_v1.2.zip"  # the full set, which includes the curated pieces
MIDI_DIR = DATA / "midi"
OUTPUT_DIR = DATA / "audio" / "midi_render"
TEMP_DIR = DATA / "tmp_render"
LOG_COLUMNS = ["work_id", "composer", "form", "title", "source_id", "soundfont", "midi_path", "flac_path",
               "duration_s", "input_lufs", "flac_bytes", "status", "error"]
MIN_DURATION = 5.0
RENDER_TIMEOUT = 1800
BATCH_SIZE = 25


def stem_id(work_id):
    return hashlib.sha256(work_id.encode()).hexdigest()[:20]


def zip_members():
    """MIDI files in the GiantMIDI zip, by name. Empty when the zip isn't here (e.g. on the GPU server)."""
    if not ZIP_PATH.exists():
        return {}
    with zipfile.ZipFile(ZIP_PATH) as archive:
        return {unicodedata.normalize("NFC", Path(name).stem): name for name in archive.namelist() if name.endswith(".mid")}


def extract_midi(member, destination):
    if destination.exists():
        return
    with zipfile.ZipFile(ZIP_PATH) as archive, archive.open(member) as source:
        data = source.read()
    if data[:4] != b"MThd":
        raise AudioError("not a MIDI file")
    destination.write_bytes(data)


def render_one(row, members, fonts):
    identifier = stem_id(row["work_id"])
    midi = MIDI_DIR / f"{identifier}.mid"
    flac = OUTPUT_DIR / f"{identifier}.flac"
    wav = TEMP_DIR / f"{identifier}.wav"
    result = {**row, "midi_path": str(midi.relative_to(DATA)), "flac_path": str(flac.relative_to(DATA)),
              "duration_s": "", "input_lufs": "", "flac_bytes": "", "status": "failed", "error": ""}
    try:
        if not midi.exists():  # ATEPP pieces were already extracted by select_atepp.py
            member = members.get(unicodedata.normalize("NFC", row["source_id"]))
            if not member:
                raise AudioError("MIDI not found in zip")
            extract_midi(member, midi)
        completed = subprocess.run(
            ["nice", "-n", "15", "fluidsynth", "-ni", "-q", "-F", str(wav), "-T", "wav", "-O", "s16", "-r", str(SAMPLE_RATE),
             "-g", "0.6", "-R", "0", "-C", "0", "-o", "synth.polyphony=1024", "-o", "synth.dynamic-sample-loading=1",
             fonts[row["soundfont"]], str(midi)],
            capture_output=True, text=True, timeout=RENDER_TIMEOUT)
        if completed.returncode != 0 or not wav.exists():
            raise AudioError(f"fluidsynth failed: {completed.stderr.strip()[-200:]}")
        loudness, duration = to_normalized_flac(wav, flac)
        if duration < MIN_DURATION:
            flac.unlink(missing_ok=True)
            raise AudioError(f"too short: {duration:.1f}s")
        result.update(duration_s=f"{duration:.2f}", input_lufs=f"{loudness:.2f}", flac_bytes=flac.stat().st_size, status="ok")
    except (AudioError, subprocess.TimeoutExpired, OSError, zipfile.BadZipFile) as error:
        result["error"] = str(error)[:300]
    finally:
        wav.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="render only the first N works, as a test")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--plan", default="render_plan.csv", help="file in data/ listing what to render")
    parser.add_argument("--log", default="render_log.csv", help="file in data/ recording the results")
    args = parser.parse_args()

    for directory in (MIDI_DIR, OUTPUT_DIR, TEMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    with (DATA / "soundfonts" / "soundfonts.csv").open(encoding="utf-8", newline="") as handle:
        fonts = {row["name"]: row["path"] for row in csv.DictReader(handle)}
    log_path = DATA / args.log
    with (DATA / args.plan).open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    if args.limit:
        plan = plan[:args.limit]

    finished = {}
    if log_path.exists():
        with log_path.open(encoding="utf-8", newline="") as handle:
            finished = {row["work_id"]: row for row in csv.DictReader(handle)}
    todo = [row for row in plan if finished.get(row["work_id"], {}).get("status") != "ok"]
    members = zip_members()
    print(f"{len(plan)} planned, {len(todo)} to render", flush=True)

    def render(row):
        return render_one(row, members, fonts)

    # Work through the pieces in small batches. Between batches the progress is saved
    # and the disk is checked, so a problem can never cost more than one batch.
    stopped_by_disk_guard = None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(todo), BATCH_SIZE):
            try:
                used, free = check_disk()
            except DiskGuardError as error:
                stopped_by_disk_guard = error
                break
            print(f"{start}/{len(todo)} done; data {used/1e9:.2f} GB; free {free/1e9:.1f} GB", flush=True)
            for result in pool.map(render, todo[start:start + BATCH_SIZE]):
                finished[result["work_id"]] = result
                if result["status"] != "ok":
                    print(f"FAILED {result['title'][:50]}: {result['error'][:120]}", flush=True)
            write_log(finished, log_path)

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    results = [finished[row["work_id"]] for row in plan if row["work_id"] in finished]
    ok = [row for row in results if row["status"] == "ok"]
    print(f"ok {len(ok)}; failed {len(results) - len(ok)}; flac {sum(int(row['flac_bytes']) for row in ok)/1e9:.2f} GB")
    if stopped_by_disk_guard:
        print(f"STOPPED by disk guard: {stopped_by_disk_guard}")
        sys.exit(1)


def write_log(finished, log_path):
    temporary = log_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(finished.values())
    temporary.replace(log_path)


if __name__ == "__main__":
    main()
