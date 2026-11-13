"""Decide a piece's musical form from its title.

Every rule below is a list of small regular expressions. A rule "fires" when
any one of its patterns is found in the title. In the patterns, \\b means a
word boundary and \\s means a space, so r"\\bpolkas?\\b" finds "polka" or
"polkas" as a whole word but not "polkadot".

Titles are always passed through fold() first, which lowercases them and
removes accents, so "Marché" and "marche" look the same to the rules.
"""
import re
import unicodedata

CLASSES = ["waltz", "mazurka", "polonaise", "nocturne", "march", "polka", "fugue"]

# A title containing any of these words is labelled with that form.
FORM_KEYWORDS = {
    "waltz": [r"\bwaltz(es)?\b", r"\bvalses?\b", r"\bvalzer\b", r"\b\w*walzer\b"],  # \w* catches "Kaiserwalzer"
    "mazurka": [r"\bmazurkas?\b", r"\bmazurek\b", r"\bmazurki\b"],
    "polonaise": [r"\bpolonaises?\b", r"\bpolonez\b", r"\bpolonezy\b"],
    "nocturne": [r"\bnocturnes?\b", r"\bnotturno\b", r"\bnotturni\b"],
    "march": [r"\bmarch(es)?\b", r"\bmarche?s?\b", r"\bmarcia\b", r"\b\w*marsch(e|en)?\b"],  # catches "Radetzkymarsch"
    "polka": [r"\bpolkas?\b"],
    "fugue": [r"\bfugues?\b", r"\bfuga\b", r"\bfughe\b", r"\bfuge\b", r"\bfugen\b"],
}

# Titles that are clearly not classical pieces, despite containing a form word.
NOT_CLASSICAL = [r"\bpolka\s+dots?\b", r"\bjam\b", r"\bpodcast\b", r"\bsermon\b", r"\bradio\b", r"\bepisode\b", r"\bprotest\b"]

# "Prelude and Fugue" and similar pieces are only half a fugue, so they are excluded.
# There is deliberately no \b at the start, so run-together names like "JSBachPrelude" also match.
FUGUE_PARTNERS = ("prelude|preludes|praeludium|preludio|toccata|fantasia|fantasie|fantaisie|fantasy|"
                  "chaconne|passacaglia|introduction|adagio|allegro|variations?")
JOINING_WORD = r"(and|&|et|und|e|y|,)?"
PAIRED_FUGUE = (rf"({FUGUE_PARTNERS})\b[\s,&-]*{JOINING_WORD}[\s,&-]*"  # the first half, then an optional "and"
                r"(\w+[\s,&-]+){0,2}?(fugue|fuga|fuge)\b")               # up to two other words, then "fugue"

# When "march" probably means the month, or another non-musical sense.
MARCH_NOT_THE_FORM = [
    r"\bmarch\s+\d{1,2}\b",                                  # March 22
    r"\b\d{1,2}(st|nd|rd|th)?\s+march\b",                    # 22nd March
    r"\bmarch\s*,?\s*(1[89]|20)\d\d\b",                      # March 2019
    r"\b(in|of|during|early|late|mid|since|until)[\s-]+march\b",  # in March
    r"\b\d{4}[-_.]03[-_.]\d{2}\b",                           # 2019-03-22
    r"\bseasons\b",                                          # Tchaikovsky's The Seasons: "March"
    r"\bje\s+marche\b",                                      # French "I walk"
]

# Titles matching these need a person's judgement. Each key becomes a reason in review.csv.
REVIEW_RULES = {
    # Two forms fused together, such as a "Valse-Caprice".
    "hybrid_form": [
        r"\bpolonaise[\s-]*fant[a]\w*",
        r"\bandante\s+spianato\b",
        r"\bvalse[\s-]*(caprice|impromptu|scherzo|etude|fantaisie|fantas\w*|ballet)",
        r"\bcaprice[\s-]*valse",
        r"\bscherzo[\s-]*valse",
        r"\bfantaisie[\s-]*(valse|polonaise|mazurka)",
        r"\bpolka[\s-]*(mazur\w*|redowa)",
        r"\bmazurka[\s-]*polka",
        r"\betude[\s-]*(valse|mazurka)",
        r"\bmarche[\s-]*fantaisie",
        # A paired title such as "Introduction and Polonaise".
        r"\b(introduction|impromptu|andante|adagio|fantaisie|fantasia|prelude|scherzo)\b(\s+\w+){0,2}[\s,]*"
        r"(and|et|und|e|&)\s+(\w+\s+){0,2}(valse|waltz|polonaise|march|marche|marsch|mazurka|polka|nocturne)\b",
    ],
    "funeral_march": [r"\bfuneral\s+march", r"\bmarche\s+funebre", r"\bmarcia\s+funebre", r"\btrauermarsch\b"],
    "in_the_style_of": [
        r"\bin\s+the\s+style\s+of\b", r"\ben\s+forme\s+de\b", r"\bin\s+form\s+of\b", r"\bstyle\s+of\b",
        r"\bdans\s+le\s+style\b", r"\balla\s+(marcia|polacca|turca)\b", r"\btempo\s+di\b", r"\ba\s+la\s+",
    ],
    "arrangement": [
        r"\barr(\.|anged|angement|ang\.)?\b", r"\btranscri(bed|ption|pt)\b", r"\bparaphrase\b", r"\bafter\b",
        r"\bon\s+themes?\b", r"\breminiscences\b", r"\bfantas(y|ia)\s+on\b", r"\bvariations?\s+on\b",
        r"\bbearbeitung\b", r"\bboogie\b", r"\bjazz\b",
    ],
    "medley": [r"\bmedley\b", r"\bpotpourri\b", r"\bselections?\b", r"\bmixed\b"],
    "multi_piece": [
        r"\b\d+\s+(\w+\s+){0,2}walzer\b",                                # 12 Walzer
        r"\b\w*marsche\b",                                               # German plural
        r"\b(waltzes|valses|mazurkas|polonaises|nocturnes|marches|polkas|fugues)\b",  # any plural form word
        r"\bcomplete\b", r"\bcollection\b", r"\bset\s+of\b", r"\bsuite\b",
        r"\bnos\.?\s*\d+\s*(-|to|and|&)\s*\d+\b",                        # Nos. 1-3
        r"\b\d+\s+and\s+\d+\b",                                          # 4 and 5
    ],
    "movement_of_larger_work": [
        r"\bsonat[ae]\b", r"\bsymphon\w+\b", r"\bconcerto\b", r"\bopera\b", r"\bballet\b", r"\boperetta\b",
        r"\bmovement\b", r"\bmvt\b", r"\bacto?\s+[ivx\d]+\b", r"\bduo\b", r"\bduet\b", r"\baria\b",
        r"\bquartet\b", r"\bquintet\b", r"\btrio\b",
    ],
    "excerpt_or_demo": [
        r"\bexcerpt\b", r"\bextract\b", r"\bpart\s+\d\b", r"\bscale\s+durations\b", r"\bdemo\b",
        r"\bringtone\b", r"\bsample\b", r"\bfragment\b",
    ],
}

# Searched in this order; the first instrument found is the one recorded.
INSTRUMENT_KEYWORDS = [
    ("orchestra", [r"\borchest\w*", r"\bsymphon\w+\b", r"\bphilharmoni\w+"]),
    ("band", [r"\bband\b", r"\bmilitary\b", r"\bwind\s+ensemble\b", r"\bbrass\b", r"\bfanfare\b"]),
    ("organ", [r"\borgan\b", r"\borgel\b", r"\borgue\b"]),
    ("harpsichord", [r"\bharpsichord\b", r"\bcembalo\b", r"\bclavecin\b"]),
    ("guitar", [r"\bguitar\b", r"\bgitarre\b", r"\bguitare\b"]),
    ("accordion", [r"\baccordion\b"]),
    ("string_ensemble", [r"\bstring\s+(quartet|quintet|trio|orchestra)\b", r"\bquartet\b"]),
    ("violin", [r"\bviolin\b"]),
    ("cello", [r"\bcello\b"]),
    ("voice", [r"\bsoprano\b", r"\btenor\b", r"\bbaritone\b", r"\bchoir\b", r"\bchorus\b", r"\bvocal\b"]),
    ("music_box", [r"\bpolyphon\b", r"\bmusic\s*box\b"]),
    ("piano", [r"\bpiano\b", r"\bpianoforte\b", r"\bklavier\b"]),
]


def fold(text):
    """Lowercase text and remove accents, so spelling differences don't matter."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return without_accents.lower().replace("ß", "ss").replace("ø", "o").replace("ł", "l")


def matches_any(patterns, text):
    """True if at least one of the regular expressions is found in the text."""
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def clean_filename(title):
    """Turn a file name like 'File:Chopin_Waltz.ogg' into 'Chopin Waltz'."""
    text = re.sub(r"^file:", "", str(title), flags=re.IGNORECASE)
    text = re.sub(r"\.(ogg|oga|flac|wav|mp3|mid|midi|opus|m4a|webm)$", "", text, flags=re.IGNORECASE)
    return re.sub(r"[_]+", " ", text).strip()


def label_title(title):
    """Label one title and return (form, review_reasons, excluded_reason).

    A title that is excluded returns form=None plus the reason it was excluded.
    A title that needs a person's judgement returns its form plus review reasons.
    """
    text = fold(title)
    if matches_any(NOT_CLASSICAL, text):
        return None, [], "non_classical_title"

    forms = [form for form in CLASSES if matches_any(FORM_KEYWORDS[form], text)]
    if not forms:
        return None, [], "no_form_keyword"

    # A title has two ways of failing to be a true fugue.
    if "fugue" in forms and re.search(PAIRED_FUGUE, text):
        return None, [], "paired_fugue"
    if "fugue" in forms and re.search(r"\bcanon\b", text):
        return None, [], "canon_not_fugue"

    reasons = [name for name, patterns in REVIEW_RULES.items() if matches_any(patterns, text)]
    # This check uses the original title, because fold() would turn "fugué" (fugal) into "fugue".
    if "fugue" in forms and matches_any([r"fugué\b", r"\bfugato\b", r"\bfughetta\b"], str(title).lower()):
        reasons.append("fugal_not_fugue")
    if "march" in forms and matches_any(MARCH_NOT_THE_FORM, text):
        reasons.append("march_as_month")
    if len(forms) > 1:
        reasons.append("multiple_forms")
    return forms[0], sorted(set(reasons)), None


def instrumentation_from_text(text):
    """Return the first instrument named in the text, or 'unknown'."""
    folded = fold(text)
    for instrument, patterns in INSTRUMENT_KEYWORDS:
        if matches_any(patterns, folded):
            return instrument
    return "unknown"


def normalize_license(short_name="", url=""):
    """Return a clean license name, or None if the license isn't one we allow.

    Only public domain, CC0, CC BY and CC BY-SA are allowed. Anything
    non-commercial (NC), no-derivatives (ND) or unrecognized returns None.
    """
    short = fold(short_name).strip()
    link = fold(url).strip()
    if re.search(r"\bnc\b|by-nc|\bnd\b|by-nd|nd-nc", f"{short} {link}"):
        return None
    if "publicdomain/zero" in link or short in ("cc0", "cc-zero") or short.startswith("cc0"):
        return "CC0"
    if "publicdomain" in link or short in ("public domain", "pdm-owner", "pdm"):
        return "Public domain"
    version = re.search(r"licenses/by-sa/(\d\.\d)", link) or re.search(r"cc by-sa (\d\.\d)", short)
    if version:
        return f"CC BY-SA {version.group(1)}"
    version = re.search(r"licenses/by/(\d\.\d)", link) or re.search(r"cc by (\d\.\d)", short)
    if version:
        return f"CC BY {version.group(1)}"
    return None
