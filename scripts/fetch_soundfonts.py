"""Download the piano soundfonts, unpack only their .sf2 files, and log licenses and checksums."""
import csv
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, check_disk, download_stream, sha256_file

SOUNDFONT_DIR = DATA / "soundfonts"
BASE = "https://freepats.zenvoid.org/Piano"
MAX_ARCHIVE_BYTES = 400_000_000
REMOTE = [
    ("salamander_grand", f"{BASE}/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz",
     "CC BY 3.0", "https://creativecommons.org/licenses/by/3.0/", "Alexander Holm (samples); FreePats (SF2 build)"),
    ("upright_kw", f"{BASE}/UprightPianoKW/UprightPianoKW-SF2-20220221.7z",
     "CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/", "FreePats"),
]
LOCAL = [("fluidr3_gm", "/usr/share/soundfonts/FluidR3_GM.sf2", "MIT", "https://opensource.org/license/mit",
          "Frank Wen and contributors (Fedora package fluid-soundfont-gm)")]
COLUMNS = ["name", "path", "bytes", "sha256", "license", "license_url", "source_url", "author", "archive_sha256"]


def unpack(archive, directory):
    """Extract into an empty directory, refusing archive members that would escape it."""
    if archive.name.endswith(".7z"):
        listing = subprocess.run(["7z", "l", "-ba", "-slt", str(archive)], capture_output=True, text=True, check=True).stdout
        paths = [line[7:] for line in listing.splitlines() if line.startswith("Path = ")]
        if any(path.startswith("/") or ".." in Path(path).parts for path in paths):
            raise RuntimeError(f"unsafe path in {archive}")
        subprocess.run(["7z", "x", "-y", f"-o{directory}", str(archive)], capture_output=True, check=True)
    else:
        with tarfile.open(archive) as handle:
            handle.extractall(directory, filter="data")


def fetch_remote(name, url):
    target = SOUNDFONT_DIR / f"{name}.sf2"
    archive_name = url.rsplit("/", 1)[1]
    with tempfile.TemporaryDirectory(dir=DATA) as temporary:
        temporary = Path(temporary)
        archive = temporary / archive_name
        archive_hash, size, _ = download_stream(url, archive, MAX_ARCHIVE_BYTES)
        print(f"{name}: downloaded {size/1e6:.0f} MB", flush=True)
        unpacked = temporary / "unpacked"
        unpacked.mkdir()
        unpack(archive, unpacked)
        fonts = sorted(unpacked.rglob("*.sf2"), key=lambda path: path.stat().st_size, reverse=True)
        if not fonts:
            raise RuntimeError(f"no .sf2 found in {archive_name}")
        print(f"{name}: using {fonts[0].name} of {[font.name for font in fonts]}", flush=True)
        fonts[0].replace(target)
    return archive_hash


def main():
    check_disk()
    SOUNDFONT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path, license_name, license_url, author in LOCAL:
        rows.append({"name": name, "path": path, "bytes": Path(path).stat().st_size, "sha256": sha256_file(path),
                     "license": license_name, "license_url": license_url, "source_url": "local system package",
                     "author": author, "archive_sha256": ""})
    for name, url, license_name, license_url, author in REMOTE:
        target = SOUNDFONT_DIR / f"{name}.sf2"
        archive_hash = "" if target.exists() else fetch_remote(name, url)
        rows.append({"name": name, "path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target),
                     "license": license_name, "license_url": license_url, "source_url": url, "author": author,
                     "archive_sha256": archive_hash})
    with (SOUNDFONT_DIR / "soundfonts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['name']}: {row['bytes']/1e6:.0f} MB, {row['license']}")


if __name__ == "__main__":
    main()
