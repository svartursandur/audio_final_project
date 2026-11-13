"""Find which composer a title or credit refers to, using a hand-made list of names.

Only sources with messy titles (Commons, Internet Archive) need this. GiantMIDI
already has clean composer columns. A piece with no recognized composer is
dropped, which is how pop songs and radio shows named "Waltz" get filtered out.
"""
import re

from labeling import fold

# Each entry is spelling=Canonical name. Several spellings can point to one composer.
COMPOSER_LIST = """
chopin=Chopin, Frederic; liszt=Liszt, Franz; schubert=Schubert, Franz; brahms=Brahms, Johannes
schumann=Schumann, Robert; mendelssohn=Mendelssohn, Felix; beethoven=Beethoven, Ludwig van
mozart=Mozart, Wolfgang Amadeus; haydn=Haydn, Joseph; handel=Handel, George Frideric; haendel=Handel, George Frideric
tchaikovsky=Tchaikovsky, Pyotr; tschaikowsky=Tchaikovsky, Pyotr; tchaikowsky=Tchaikovsky, Pyotr; chaikovsky=Tchaikovsky, Pyotr
dvorak=Dvorak, Antonin; grieg=Grieg, Edvard; faure=Faure, Gabriel; saint-saens=Saint-Saens, Camille; debussy=Debussy, Claude
ravel=Ravel, Maurice; satie=Satie, Erik; sibelius=Sibelius, Jean; rachmaninoff=Rachmaninoff, Sergei; rachmaninov=Rachmaninoff, Sergei
scriabin=Scriabin, Alexander; skryabin=Scriabin, Alexander; glazunov=Glazunov, Alexander; glazounov=Glazunov, Alexander
granados=Granados, Enrique; albeniz=Albeniz, Isaac; szymanowski=Szymanowski, Karol; moniuszko=Moniuszko, Stanislaw
paderewski=Paderewski, Ignacy; wieniawski=Wieniawski, Henryk; moszkowski=Moszkowski, Moritz; scharwenka=Scharwenka, Xaver
godard=Godard, Benjamin; chabrier=Chabrier, Emmanuel; waldteufel=Waldteufel, Emile; lanner=Lanner, Joseph; lehar=Lehar, Franz
offenbach=Offenbach, Jacques; suppe=Suppe, Franz von; sousa=Sousa, John Philip; elgar=Elgar, Edward; ives=Ives, Charles
joplin=Joplin, Scott; gottschalk=Gottschalk, Louis Moreau; czerny=Czerny, Carl; hummel=Hummel, Johann Nepomuk; clementi=Clementi, Muzio
kuhlau=Kuhlau, Friedrich; dussek=Dussek, Jan Ladislav; pachelbel=Pachelbel, Johann; buxtehude=Buxtehude, Dieterich
frescobaldi=Frescobaldi, Girolamo; reicha=Reicha, Anton; reger=Reger, Max; franck=Franck, Cesar; widor=Widor, Charles-Marie
vierne=Vierne, Louis; dupre=Dupre, Marcel; guilmant=Guilmant, Alexandre; gounod=Gounod, Charles; bizet=Bizet, Georges
berlioz=Berlioz, Hector; verdi=Verdi, Giuseppe; wagner=Wagner, Richard; rossini=Rossini, Gioachino; meyerbeer=Meyerbeer, Giacomo
bruckner=Bruckner, Anton; mahler=Mahler, Gustav; prokofiev=Prokofiev, Sergei; shostakovich=Shostakovich, Dmitri; shostakovitch=Shostakovich, Dmitri
khachaturian=Khachaturian, Aram; mussorgsky=Mussorgsky, Modest; moussorgsky=Mussorgsky, Modest; rimsky-korsakov=Rimsky-Korsakov, Nikolai
borodin=Borodin, Alexander; balakirev=Balakirev, Mily; arensky=Arensky, Anton; lyadov=Lyadov, Anatoly; liadov=Lyadov, Anatoly
medtner=Medtner, Nikolai; glinka=Glinka, Mikhail; dargomyzhsky=Dargomyzhsky, Alexander; kabalevsky=Kabalevsky, Dmitry
tarrega=Tarrega, Francisco; barrios=Barrios, Agustin; giuliani=Giuliani, Mauro; carulli=Carulli, Ferdinando
czibulka=Czibulka, Alphons; ivanovici=Ivanovici, Iosif; fucik=Fucik, Julius; teike=Teike, Carl; ganne=Ganne, Louis
gungl=Gungl, Josef; ziehrer=Ziehrer, Carl Michael; komzak=Komzak, Karel; kalman=Kalman, Emmerich; fahrbach=Fahrbach, Philipp
hindemith=Hindemith, Paul; bartok=Bartok, Bela; stravinsky=Stravinsky, Igor; gershwin=Gershwin, George; holst=Holst, Gustav
sullivan=Sullivan, Arthur; ponce=Ponce, Manuel; villa-lobos=Villa-Lobos, Heitor; piazzolla=Piazzolla, Astor; lecuona=Lecuona, Ernesto
turina=Turina, Joaquin; mompou=Mompou, Federico; poulenc=Poulenc, Francis; milhaud=Milhaud, Darius; honegger=Honegger, Arthur
lefebure-wely=Lefebure-Wely, Louis; thalberg=Thalberg, Sigismond; henselt=Henselt, Adolf; alkan=Alkan, Charles-Valentin
raff=Raff, Joachim; gade=Gade, Niels; sinding=Sinding, Christian; svendsen=Svendsen, Johan; heller=Heller, Stephen
burgmuller=Burgmuller, Friedrich; pierne=Pierne, Gabriel; massenet=Massenet, Jules; delibes=Delibes, Leo; gliere=Gliere, Reinhold
tcherepnin=Tcherepnin, Alexander; cimarosa=Cimarosa, Domenico; scarlatti=Scarlatti, Domenico; weber=Weber, Carl Maria von
"""


def parse_composer_list():
    """Turn the text list above into a dictionary of {spelling: canonical name}."""
    composers = {}
    for entry in re.split(r"[;\n]", COMPOSER_LIST):
        if "=" in entry:
            spelling, name = entry.strip().split("=", 1)
            composers[spelling.strip()] = name.strip()
    return composers


COMPOSER_MAP = parse_composer_list()

# Surnames shared by several composers, or that are also common words, must appear with a first name.
FULL_NAME_COMPOSERS = {
    "john field": "Field, John", "field, john": "Field, John",
    "johann strauss": "Strauss, Johann", "strauss, johann": "Strauss, Johann",
    "josef strauss": "Strauss, Josef", "strauss, josef": "Strauss, Josef",
    "eduard strauss": "Strauss, Eduard", "strauss, eduard": "Strauss, Eduard",
    "richard strauss": "Strauss, Richard", "strauss, richard": "Strauss, Richard",
    "anton rubinstein": "Rubinstein, Anton", "rubinstein, anton": "Rubinstein, Anton",
    "cesar cui": "Cui, Cesar", "cui, cesar": "Cui, Cesar",
    "manuel de falla": "Falla, Manuel de", "vaughan williams": "Vaughan Williams, Ralph",
    "carl nielsen": "Nielsen, Carl", "nielsen, carl": "Nielsen, Carl",
    "samuel barber": "Barber, Samuel", "barber, samuel": "Barber, Samuel",
    "c.p.e. bach": "Bach, Carl Philipp Emanuel", "cpe bach": "Bach, Carl Philipp Emanuel",
    "w.f. bach": "Bach, Wilhelm Friedemann", "wilhelm friedemann bach": "Bach, Wilhelm Friedemann",
    "carl philipp emanuel bach": "Bach, Carl Philipp Emanuel",
}

# Longest spellings go first so that "rimsky-korsakov" is tried as a whole. The two
# lookaround checks stop a name from matching inside a longer word or hyphenated name.
ALL_SPELLINGS = "|".join(re.escape(spelling) for spelling in sorted(COMPOSER_MAP, key=len, reverse=True))
COMPOSER_WORD = rf"(?<![\w-])({ALL_SPELLINGS})(?![\w-])"

# "Haydn Quartet" or "Sousa's Band" names performers, not a composer, so it is removed first.
PERFORMER_GROUP = r"[\w'’.-]+\s+(?:string\s+)?(?:quartet|quintet|trio|orchestra|band|ensemble)\b"
BACH_SONS = r"c\.?\s*p\.?\s*e\.?\s+bach|w\.?\s*f\.?\s+bach|friedemann\s+bach|emanuel\s+bach"


def first_name_conflicts(text_after_name, canonical_name, creator_style):
    """True for 'Wagner, J.' when the list means Richard Wagner: a different person."""
    given = re.match(r"\s*,\s*([a-z])(\.?)", text_after_name)
    if not given:
        return False
    initial, has_dot = given.group(1), given.group(2)
    expected_initial = fold(canonical_name.split(", ")[1])[:1]
    # In a title, "Chopin, Nocturne" is not a first name, so only an initial with a dot counts there.
    return initial != expected_initial and (creator_style or bool(has_dot))


def find_composers(text, creator_style=False):
    """Return every composer named in the text, as 'Surname, First'.

    Set creator_style=True for credit fields written as 'surname, first name'.
    """
    folded = re.sub(PERFORMER_GROUP, " ", fold(text))
    found = []

    for full_name, canonical in FULL_NAME_COMPOSERS.items():
        if full_name in folded and canonical not in found:
            found.append(canonical)

    # A plain "Bach" or a BWV number means J.S. Bach, once his sons' names are removed.
    if re.search(r"\bbach\b|\bbwv\b", re.sub(BACH_SONS, " ", folded)):
        found.append("Bach, Johann Sebastian")

    for match in re.finditer(COMPOSER_WORD, folded):
        canonical = COMPOSER_MAP[match.group(1)]
        if first_name_conflicts(folded[match.end():], canonical, creator_style):
            continue
        if canonical not in found:
            found.append(canonical)

    # A bare "Strauss" could be any of four composers, so it is marked for review.
    if re.search(r"\bstrauss\b", folded) and not any(name.startswith("Strauss") for name in found):
        found.append("Strauss, (unspecified)")
    return found
