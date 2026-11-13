"""Download the three small metadata CSVs. Never fetches audio or MIDI archives."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, check_disk, download_file

FILES = {
    "musicnet/musicnet_metadata.csv": "https://zenodo.org/records/5120004/files/musicnet_metadata.csv?download=1",
    "maestro/maestro-v3.0.0.csv": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv",
    "giantmidi/full_music_pieces_youtube_similarity_pianosoloprob.csv": "https://raw.githubusercontent.com/bytedance/GiantMIDI-Piano/master/resources/full_music_pieces_youtube_similarity_pianosoloprob.csv",
}


def main():
    check_disk()
    for relative, url in FILES.items():
        destination = RAW / relative
        fetched = download_file(url, destination)
        state = "downloaded" if fetched else "already present"
        print(f"{relative}: {state}, {destination.stat().st_size} bytes")


if __name__ == "__main__":
    main()
