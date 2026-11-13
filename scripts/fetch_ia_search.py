"""Search the Internet Archive for licensed audio items; item file lists are fetched separately."""
import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, cached_json, check_disk

API = "https://archive.org/advancedsearch.php?"
FIELDS = ["identifier", "title", "creator", "licenseurl", "collection", "date"]
FORMS = {
    "waltz": "waltz OR valse OR walzer",
    "mazurka": "mazurka OR mazurek",
    "polonaise": "polonaise OR polonez",
    "nocturne": "nocturne OR notturno",
    "march": "march OR marche OR marsch",
    "polka": "polka",
    "fugue": "fugue OR fuga",
}
QUERIES = {name: f"mediatype:audio AND title:({terms}) AND licenseurl:[* TO *]" for name, terms in FORMS.items()}
QUERIES["musopen_collection"] = "collection:musopen AND mediatype:audio"
QUERIES["musopen_any"] = "mediatype:audio AND musopen"
ROWS = 1000
MAX_ITEMS_PER_QUERY = 30000


def search(query):
    docs, page, total = [], 1, None
    while True:
        pairs = [("q", query), ("rows", ROWS), ("page", page), ("output", "json")]
        pairs += [("fl[]", field) for field in FIELDS]
        reply = cached_json(API + urllib.parse.urlencode(pairs), "ia_search")
        if not reply or "response" not in reply:
            raise RuntimeError(f"Archive search failed: {query}: {reply}")
        total = reply["response"]["numFound"]
        batch = reply["response"]["docs"]
        docs += batch
        if not batch or len(docs) >= total or len(docs) >= MAX_ITEMS_PER_QUERY:
            return docs, total
        page += 1


def main():
    check_disk()
    items, totals = {}, {}
    for name, query in QUERIES.items():
        docs, total = search(query)
        totals[name] = {"retrieved": len(docs), "total": total}
        print(f"{name}: {len(docs)} retrieved of {total}", flush=True)
        for doc in docs:
            entry = items.setdefault(doc["identifier"], {**doc, "matched": []})
            entry["matched"].append(name)
    destination = RAW / "ia" / "search.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for item in items.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (RAW / "ia" / "search_totals.json").write_text(json.dumps(totals, indent=1))
    print(f"wrote {len(items)} distinct items", flush=True)


if __name__ == "__main__":
    main()
