"""Give every musical work one ID, however its title happens to be spelled.

A work_id looks like "chopin-frederic|waltz|op64n2": composer, form, catalog
number. Splits are made per work_id, so if one piece got two different IDs it
could land in both training and testing. That is why this matters.
"""
import re

from labeling import fold

# Catalog systems, tried in this order after opus numbers. Example: "BWV 851" gives "bwv851".
CATALOG_PATTERNS = [
    ("bwvanh", r"\bbwv\.?\s*anh\.?\s*(\d+)"),        # Bach appendix
    ("bwv", r"\bbwv\.?\s*(\d+[a-z]?)"),              # Bach
    ("woo", r"\bwoo\.?\s*(\d+)"),                    # Beethoven, without opus
    ("hob", r"\bhob\.?\s*([ivxl]+[:.]\s*\d+)"),      # Haydn
    ("kv", r"\bk(v|\.)\s*(\d+)"),                    # Mozart
    ("d", r"\bd\.\s*(\d+)"),                         # Schubert
    ("s", r"\bs\.\s*(\d+)"),                         # Liszt
    ("b", r"\bb\.\s*(\d+)"),                         # Chopin (Brown catalog)
    ("kk", r"\bkk\.?\s*([ivxl]+[a-z]?[\s/]*\d+)"),   # Chopin (Kobylanska catalog)
]

# Finds "Op. 64 No. 2", "op-64-no-2" or "opus.64.2". Group 1 is the opus, group 2 the word
# "no" if present, and group 3 the number inside the opus.
OPUS_PATTERN = (r"\bop(?:us)?[\s.\-_]*(?:post\w*[\s.\-_]*)?(\d+[a-z]?)"
                r"(?:[\s,/:.\-_]*(?:(no|nr|n|number)[\s.\-_]*)?(\d+))?")
# Without the word "no", a large number after the opus is probably a year, not a piece number.
LARGEST_NUMBER_WITHOUT_NO = 12
NUMBER_PATTERN = r"\b(?:no|nr)[\s.\-_]*(\d+)"                 # "No. 7"
KEY_PATTERN = r"\bin\s+([a-g])(?:[\s-]*(flat|sharp|b|#))?[\s-]*(major|minor|dur|moll)?\b"  # "in E-flat major"
NOT_ALPHANUMERIC = r"[^a-z0-9]+"
TITLE_WORDS_TO_DROP = {"in", "the", "of", "a", "an", "for", "and", "piano", "major", "minor", "no", "op"}

# Chopin's popular numbers ("Nocturne No. 7") mapped to the opus they stand for.
CHOPIN_NOCTURNES = {
    1: "op9n1", 2: "op9n2", 3: "op9n3", 4: "op15n1", 5: "op15n2", 6: "op15n3", 7: "op27n1",
    8: "op27n2", 9: "op32n1", 10: "op32n2", 11: "op37n1", 12: "op37n2", 13: "op48n1", 14: "op48n2",
    15: "op55n1", 16: "op55n2", 17: "op62n1", 18: "op62n2", 19: "op72n1", 20: "b49", 21: "b108",
}
CHOPIN_WALTZES = {
    1: "op18", 2: "op34n1", 3: "op34n2", 4: "op34n3", 5: "op42", 6: "op64n1", 7: "op64n2",
    8: "op64n3", 9: "op69n1", 10: "op69n2", 11: "op70n1", 12: "op70n2", 13: "op70n3",
}


def slug(text):
    """Make a short ID-safe string: 'Chopin, Frederic' becomes 'chopin-frederic'."""
    return re.sub(NOT_ALPHANUMERIC, "-", fold(text)).strip("-")


def remove_bracketed_credits(title):
    """Drop performer credits, which are usually in brackets."""
    return re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", title)


def find_opus(folded_title):
    """Return (opus, number_within_opus) from a title. Either can be None."""
    match = re.search(OPUS_PATTERN, folded_title)
    if not match:
        return None, None
    opus, word_no, number = match.group(1), match.group(2), match.group(3)
    if number and not word_no and int(number) > LARGEST_NUMBER_WITHOUT_NO:
        number = None
    return opus, number


def chopin_key(form, folded_title, opus, number_in_opus):
    """Resolve a Chopin waltz or nocturne to one catalog key, or None if unsure."""
    if form == "nocturne":
        table = CHOPIN_NOCTURNES
    elif form == "waltz":
        table = CHOPIN_WALTZES
    else:
        return None

    match = re.search(NUMBER_PATTERN, folded_title) or re.search(rf"\b{form}[\s.\-_]*(\d+)\b", folded_title)
    popular_number = int(match.group(1)) if match else None

    if form == "nocturne" and opus == "72" and not number_in_opus:
        return "op72n1"  # Op. 72 has only one nocturne
    if opus and not number_in_opus and popular_number:
        entry = table.get(popular_number, "")
        if entry.startswith(f"op{opus}"):
            return entry
        return f"op{opus}n{popular_number}"
    if not opus and popular_number in table:
        return table[popular_number]
    if form == "nocturne" and not opus and "posth" in folded_title:
        if re.search(r"\bc[\s-]*sharp\s+minor|\bcis[\s-]*moll", folded_title):
            return "b49"
        if re.search(r"\bc\s+minor\b|\bc[\s-]*moll", folded_title):
            return "b108"
    return None


def catalog_key(title, form="", composer=""):
    """Return a catalog key like 'op64n2' or 'bwv851', or None if the title has none."""
    folded = fold(title)
    opus, number = find_opus(folded)

    if composer.startswith("Chopin"):
        special = chopin_key(form, folded, opus, number)
        if special:
            return special
        if form == "polonaise" and "militar" in folded:
            return "op40n1"  # the "Military" polonaise
        if form == "polonaise" and re.search(r"\bh[eé]ro", folded):
            return "op53"    # the "Heroic" polonaise
    if composer.startswith("Bach, Johann") and "art of fugue" in folded:
        return "bwv1080"

    if opus:
        return f"op{opus}n{number}" if number else f"op{opus}"

    for prefix, pattern in CATALOG_PATTERNS:
        found = re.search(pattern, folded)
        if found:
            key = prefix + re.sub(NOT_ALPHANUMERIC, "", found.group(found.lastindex))
            # B. 16 is the one Chopin number covering two different mazurkas, so the key tells them apart.
            if composer.startswith("Chopin") and form == "mazurka" and key == "b16":
                musical_key = re.search(KEY_PATTERN, folded)
                if musical_key:
                    return f"{key}-{musical_key.group(1)}{musical_key.group(2) or ''}"
            return key
    return None


def number_and_key_id(folded_title):
    """Fallback ID from the piece number and musical key, e.g. 'no3-eflatmajor'. None if neither is present."""
    number = re.search(NUMBER_PATTERN, folded_title)
    key = re.search(KEY_PATTERN, folded_title)
    if not number and not key:
        return None
    parts = []
    if number:
        parts.append(f"no{number.group(1)}")
    if key:
        accidental = {"b": "flat", "#": "sharp"}.get(key.group(2), key.group(2) or "")
        mode = {"dur": "major", "moll": "minor"}.get(key.group(3), key.group(3) or "")
        parts.append(f"{key.group(1)}{accidental}{mode}")
    return "-".join(parts)


def title_words_id(folded_title, composer):
    """Last-resort ID built from the words of the title."""
    without_track_number = re.sub(r"^\s*\d+\s*[:.\-]\s*", "", folded_title)
    without_performer = re.sub(r"\b(?:played|performed)\b.*$", " ", without_track_number)
    surname_words = fold(composer).split(",")[0].split()
    words = [word for word in re.split(NOT_ALPHANUMERIC, without_performer)
             if word and word not in TITLE_WORDS_TO_DROP and word not in surname_words]
    return "-".join(words)[:60]


def work_id(composer, form, title):
    """Build the ID, preferring a catalog number, then number and key, then title words."""
    without_credits = remove_bracketed_credits(title)
    piece = catalog_key(without_credits, form, composer)
    if piece is None:
        folded = fold(without_credits)
        piece = number_and_key_id(folded)
        if piece is None:
            piece = title_words_id(folded, composer)
    return f"{slug(composer)}|{form}|{piece}"
