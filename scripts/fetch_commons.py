"""Search Wikimedia Commons by title keyword and store file and license metadata."""
import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, cached_json, check_disk

API = "https://commons.wikimedia.org/w/api.php?"
KEYWORDS = [
    "waltz", "valse", "walzer", "mazurka", "mazurek", "polonaise", "polonez",
    "nocturne", "notturno", "march", "marche", "marsch", "polka", "fugue", "fuga",
]
EXTMETA = "LicenseShortName|LicenseUrl|License|Artist|ObjectName|DateTimeOriginal|ImageDescription|Categories"
PAGE_SIZE = 500
MAX_OFFSET = 9500  # CirrusSearch refuses offsets at or beyond 10,000


def call(params):
    url = API + urllib.parse.urlencode({"action": "query", "format": "json", **params})
    return cached_json(url, "commons")


def search_keyword(keyword):
    titles, offset = [], 0
    total = None
    while offset <= MAX_OFFSET:
        reply = call({
            "list": "search", "srnamespace": 6, "srlimit": PAGE_SIZE, "sroffset": offset,
            "srsearch": f"intitle:{keyword} filetype:audio", "srinfo": "totalhits", "srprop": "",
        })
        if "query" not in reply:
            raise RuntimeError(f"Commons error for {keyword}: {reply.get('error')}")
        total = reply["query"]["searchinfo"]["totalhits"]
        titles += [hit["title"] for hit in reply["query"]["search"]]
        next_offset = reply.get("continue", {}).get("sroffset")
        if next_offset is None:
            break
        offset = next_offset
    print(f"{keyword}: {len(titles)} retrieved of {total}", flush=True)
    return titles, total


def file_info(titles):
    records = []
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        reply = call({
            "prop": "imageinfo", "titles": "|".join(batch),
            "iiprop": "url|size|mime|extmetadata", "iiextmetadatafilter": EXTMETA,
            "iiextmetadatalanguage": "en",
        })
        for page in reply.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = {key: value.get("value", "") for key, value in info.get("extmetadata", {}).items()}
            records.append({
                "title": page.get("title", ""), "pageid": page.get("pageid"),
                "url": info.get("url", ""), "descriptionurl": info.get("descriptionurl", ""),
                "size": info.get("size"), "duration": info.get("duration"), "mime": info.get("mime", ""),
                "meta": meta,
            })
    return records


def main():
    check_disk()
    seen, totals = {}, {}
    for keyword in KEYWORDS:
        titles, total = search_keyword(keyword)
        totals[keyword] = {"retrieved": len(titles), "total": total}
        for title in titles:
            seen.setdefault(title, keyword)
    print(f"{len(seen)} distinct files; fetching license metadata", flush=True)
    records = file_info(sorted(seen))
    destination = RAW / "commons" / "files.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (RAW / "commons" / "search_totals.json").write_text(json.dumps(totals, indent=1))
    print(f"wrote {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
