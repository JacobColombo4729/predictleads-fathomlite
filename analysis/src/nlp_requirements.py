"""How many years of experience, and what degree, a requirement asks for - read from the sentence's grammar.

**No regular expressions** (`EDA_SDA_3_Lean`, decided 2026-10-25). A pattern over characters can see that "5" sits
near "years"; it cannot see *what the 5 modifies* or *what the years are years of*, so it has to approximate both
with a window ("`years?`, then up to 50 characters, then `experience`"). spaCy's dependency parse states the
relations outright:

    We need 5+ years of experience with Python
            |    |       |
            |    |       +-- pobj of "of", whose head is "years"   -> the years are years OF EXPERIENCE
            |    +---------- the noun the number modifies
            +--------------- nummod of "years"                     -> the number belongs to "years"

So the rule here is a statement about grammar: **a number that modifies a noun whose lemma is "year", where that
noun's own subtree (or its head's) mentions experience.** That rejects "founded in 1998, we have 25 years of
history" without a stop-list, because *history* is not an experience word and the parse says the 25 belongs to it.

The degree rule works the same way. "Bachelor's degree or equivalent experience" parses as

    degree --conj--> experience <--amod-- equivalent

so the equivalence is a coordination between the degree and an alternative, not two keywords that happen to share
a line.

Lemma sets (`YEAR_LEMMAS`, `DEGREE_LEVELS`, ...) are vocabulary, not patterns: they list the *words* a concept is
expressed with, and the parse decides what those words are doing. The sets are small and visible on purpose.

`may_state_requirement` is the screen that makes this affordable over millions of blocks: a text with no number
and no degree word in it cannot state either requirement, so it is never parsed. It is a superset, and the grammar
rules above do all the deciding.

NER is loaded too, and used only as a cross-check: `date_entities` returns the spans the statistical model tags as
DATE, which the notebook compares with what the parse rule found. Two independent readings agreeing is evidence;
the parse is what the numbers come from.

Teaching doc: `docs/nlp_methods.md` (part 3).
"""
from collections import namedtuple

import pandas as pd

MODEL = "en_core_web_sm"

# Vocabulary. Each set lists the words a concept is written with; the parse decides what they modify.
YEAR_LEMMAS = frozenset({"year", "yr", "yrs"})
EXPERIENCE_LEMMAS = frozenset({"experience", "background", "exp", "expertise", "practice", "tenure", "work",
                               "working", "industry", "professional", "hands-on"})
DEGREE_LEMMAS = frozenset({"degree", "diploma", "bachelor", "baccalaureate", "master", "mba", "phd", "ph.d.",
                           "doctorate", "doctoral", "associate", "bs", "ba", "bsc", "msc", "ms", "ma", "undergraduate",
                           "graduate", "postgraduate"})
DEGREE_LEVELS = {  # a degree word -> its level, so "the highest asked for" has a meaning
    "associate": "associate", "diploma": "associate",
    "bachelor": "bachelor", "baccalaureate": "bachelor", "bs": "bachelor", "ba": "bachelor", "bsc": "bachelor",
    "undergraduate": "bachelor",
    "master": "master", "mba": "master", "msc": "master", "ms": "master", "ma": "master", "postgraduate": "master",
    "phd": "phd", "ph.d.": "phd", "doctorate": "phd", "doctoral": "phd",
}
LEVEL_ORDER = ["associate", "bachelor", "master", "phd"]           # lowest to highest
EQUIVALENT_LEMMAS = frozenset({"equivalent", "comparable", "commensurate", "lieu", "equivalency"})
# What an equivalence can be offered instead of, or alongside: an alternative to the degree, or the degree word
# itself ("High school diploma or equivalent" coordinates the equivalence straight onto "diploma").
ALTERNATIVE_LEMMAS = frozenset({"experience", "combination", "work", "background", "training", "education",
                                "qualification"}) | DEGREE_LEMMAS
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
MAX_YEARS = 20  # "50 years of experience" is the company's history, not something it is asking an applicant for

# The screen's vocabulary (see `may_state_requirement`). A years requirement must contain a number, which the
# tokenizer can see on its own; a degree requirement must contain a degree word. Tokenizing loads no lemmatizer, so
# the plural of each degree word is listed too - the screen is meant to over-include, and the parse decides.
SCREEN_WORDS = frozenset({form for word in DEGREE_LEMMAS for form in (word, word + "s")})

Facts = namedtuple("Facts", "years degree_level degree_equivalent")


def pipeline(model=MODEL):
    """The spaCy pipeline these rules need: the tagger and lemmatizer for word identity, the parser for the
    relations, NER for the cross-check. Nothing is disabled, because every component is used."""
    import spacy

    return spacy.load(model)


def tokenizer():
    """Tokenization and nothing else: no tagger, lemmatizer, parser or NER. This is what the screen below runs on,
    and it is why the screen can afford to look at every block of every description."""
    import spacy

    return spacy.blank("en")


def may_state_requirement(texts, nlp, batch_size=2_000):
    """Which of these texts could state a years or degree requirement at all: one boolean per text, in order.

    A screen, not a decision. Parsing a sentence costs an order of magnitude more than tokenizing it, and most
    blocks of a description - what the job involves, the benefits, the company's mission - cannot state either
    requirement: a years requirement needs a number in it and a degree requirement needs a degree word in it. A
    block with neither is skipped without being read.

    The screen over-includes on purpose. "A 401k match", "5 direct reports" and "graduate of our academy" all pass
    it, and `years_asked` and `degree_asked` then throw them out on the grammar. Nothing a rule here could find is
    screened out, which is what `tests/test_nlp_requirements.py` pins: every positive case in that file passes.
    """
    return [any(token.like_num or token.lower_ in SCREEN_WORDS for token in doc)
            for doc in nlp.pipe(texts, batch_size=batch_size)]


def _as_number(token):
    """A number token as an int: digits directly, spelled-out numbers from the small vocabulary above."""
    text = token.text.replace(",", "")
    if text.isdigit():
        return int(text)
    return WORD_NUMBERS.get(token.lower_)


def _mentions(token, lemmas):
    """True when the token's own subtree, or the subtree of its head, uses one of these words. One hop up is enough
    to cross "a minimum of 5 years' experience", where the experience hangs off the head rather than the years."""
    for scope in (token, token.head):
        if any(t.lemma_.lower() in lemmas or t.lower_ in lemmas for t in scope.subtree):
            return True
    return False


def _numbers_modifying(token):
    """Every number attached to this noun, the far end of a range included. A range hangs its low end off its high
    end rather than off the noun ("3-5 years" parses as 3 -quantmod-> 5 -nummod-> years), so a numeric modifier's
    own subtree is searched too."""
    numeric = []
    for child in token.children:
        if child.like_num:
            numeric += [t for t in child.subtree if t.like_num]
    return [n for n in (_as_number(t) for t in numeric) if n is not None]


def years_asked(doc):
    """The most years any sentence of `doc` asks for, or None. Within one mention a range takes its lower end
    ("3-5 years" asks for 3); across mentions the largest wins, because that is the bar an applicant has to clear."""
    found = []
    for token in doc:
        if token.lemma_.lower() not in YEAR_LEMMAS and token.lower_ not in YEAR_LEMMAS:
            continue
        if not _mentions(token, EXPERIENCE_LEMMAS):
            continue                                   # years of something else: history, a contract, a degree
        numbers = [n for n in _numbers_modifying(token) if 0 < n <= MAX_YEARS]
        if numbers:
            found.append(min(numbers))                 # the lower end of a range is what is actually required
    return max(found) if found else None


def mentions_degree(doc):
    """True when the text names a degree at all, with or without a level. "A degree is required" names no level,
    but it is still a degree requirement, so the two questions are kept apart."""
    return any(t.lemma_.lower() in DEGREE_LEMMAS or t.lower_ in DEGREE_LEMMAS for t in doc)


def degree_asked(doc):
    """(level, equivalent_accepted): the highest degree level mentioned, and whether an alternative is offered."""
    levels = [DEGREE_LEVELS[key] for token in doc
              for key in [token.lemma_.lower(), token.lower_] if key in DEGREE_LEVELS]
    if not levels and not mentions_degree(doc):
        return None, False
    level = max(levels, key=LEVEL_ORDER.index) if levels else None
    return level, _equivalent_offered(doc)


def _equivalent_offered(doc):
    """True when the sentence offers something instead of the degree. The parse has to agree: the equivalence word
    must be grammatically connected to an alternative - heading its own phrase ("in lieu of a degree"), coordinated
    with one ("diploma or equivalent"), or modifying one ("or equivalent experience") - not merely sharing a line
    with it."""
    for token in doc:
        if token.lemma_.lower() not in EQUIVALENT_LEMMAS and token.lower_ not in EQUIVALENT_LEMMAS:
            continue
        related = [*token.subtree, token.head, token.head.head, *token.head.children]
        if any(t.lemma_.lower() in ALTERNATIVE_LEMMAS for t in related):
            return True
    return False


def facts(doc):
    """Everything this module reads from one piece of requirement text."""
    level, equivalent = degree_asked(doc)
    return Facts(years_asked(doc), level, equivalent)


def date_entities(doc):
    """The spans the statistical NER model tags as DATE and that are about years - the independent cross-check on
    `years_asked`. Not used to produce any number."""
    return [ent.text for ent in doc.ents if ent.label_ == "DATE"
            and any(t.lemma_.lower() in YEAR_LEMMAS or t.lower_ in YEAR_LEMMAS for t in ent)]


def facts_frame(texts, nlp, batch_size=64, n_process=1):
    """One row per text: years, degree_level, degree_equivalent, and the NER cross-check columns.

    `texts` should be the *distinct* requirement texts - parsing is the expensive step and description bullets
    repeat heavily across listings, so the notebook parses each distinct one once and joins the facts back."""
    texts = list(texts)
    rows = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        level, equivalent = degree_asked(doc)
        dates = date_entities(doc)
        rows.append((years_asked(doc), mentions_degree(doc), level, equivalent, len(dates) > 0, len(doc)))
    out = pd.DataFrame(rows, columns=["years", "degree_named", "degree_level", "degree_equivalent",
                                      "ner_says_years", "n_tokens"])
    out.insert(0, "text", texts)
    out["years"] = out.years.astype("Int64")
    out["degree_level"] = out.degree_level.astype("category")
    return out
