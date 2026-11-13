"""Fetch per-file lists for Internet Archive items that could enter the manifest."""
import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, cached_json, check_disk
from composers import find_composers
from labeling import label_title, normalize_license

AUDIO_FORMATS = ("mp3", "flac", "ogg", "wave", "wav", "aiff", "m4a", "opus")


def first(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def is_musopen(item):
    return "musopen" in str(item.get("collection", "")).lower() or any(m.startswith("musopen") for m in item["matched"])


def selected(item):
    if is_musopen(item):
        return True
    title, creator = str(item.get("title", "")), str(item.get("creator", ""))
    form, _, _ = label_title(title)
    return bool(normalize_license("", first(item.get("licenseurl"))) and form and find_composers(f"{title} ; {creator}"))


def audio_files(metadata):
    files = []
    for entry in metadata.get("files", []):
        kind = str(entry.get("format", "")).lower()
        if entry.get("source") == "original" and any(name in kind for name in AUDIO_FORMATS):
            files.append({
                "name": entry.get("name", ""), "title": entry.get("title", ""), "format": entry.get("format", ""),
                "size": entry.get("size"), "length": entry.get("length"), "creator": entry.get("creator", ""),
            })
    return files


def main():
    check_disk()
    items = [json.loads(line) for line in (RAW / "ia" / "search.jsonl").open(encoding="utf-8")]
    chosen = [item for item in items if selected(item)]
    print(f"fetching file lists for {len(chosen)} items", flush=True)
    destination = RAW / "ia" / "items.jsonl"
    missing = 0
    with destination.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(chosen, 1):
            quoted = urllib.parse.quote(item["identifier"], safe="")
            metadata = cached_json(f"https://archive.org/metadata/{quoted}", "ia_items")
            if not metadata or "metadata" not in metadata:
                missing += 1
                continue
            handle.write(json.dumps({
                **item, "licenseurl_item": metadata["metadata"].get("licenseurl", ""),
                "files": audio_files(metadata),
            }, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                print(f"{index}/{len(chosen)}", flush=True)
    print(f"done; {missing} items unavailable", flush=True)


if __name__ == "__main__":
    main()
