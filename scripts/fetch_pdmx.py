"""Download the PDMX metadata table (one row per score, no music files)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, check_disk, download_stream

URL = "https://zenodo.org/api/records/15571083/files/PDMX.csv/content"
MAX_BYTES = 300_000_000


def main():
    check_disk()
    destination = RAW / "pdmx" / "PDMX.csv"
    digest, size, already_there = download_stream(URL, destination, MAX_BYTES)
    state = "already present" if already_there else "downloaded"
    print(f"{destination.name}: {state}, {size/1e6:.0f} MB, sha256 {digest}")


if __name__ == "__main__":
    main()
