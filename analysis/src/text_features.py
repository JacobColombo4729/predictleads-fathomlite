"""Text features from listing descriptions (EDA_SDA_3), built batch by batch so the full scope fits in memory.

Descriptions come from EDA_SDA_1's scope_descriptions.parquet (id, description HTML). Each batch is cleaned (HTML to
lines, boilerplate removed) and turned into features; only the features and the cleaned text are written out, never
held for the whole scope at once. Rules read the cleaned text, which keeps case, numbers and punctuation ("3+ years",
"B.S.", "Security+"), and tell requirement lines from preferred ones using the headings kept in to_lines.

Every rule here is a keyword rule: EDA_SDA_3 validates each one on hand-checked samples before any of them hides
listings. Nothing here prints listing text.
"""
import hashlib
import html
import re
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)  # short plain text looks like a file name to bs4

# ---------- 2a HTML to lines ----------
BLOCK_TAGS = ["p", "li", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol"]


def to_lines(s):
    """Description HTML -> a list of clean text lines. Paragraphs, list items and headings each become a line, so
    "Requirements" and "Preferred" headings survive; long unbroken text is split into sentences."""
    if not isinstance(s, str) or not s.strip():
        return []
    if "<" in s and ">" in s:
        soup = BeautifulSoup(s, "html.parser")
        for tag in soup.find_all(BLOCK_TAGS):
            tag.insert_before("\n")
            tag.insert_after("\n")
        s = soup.get_text()
    s = html.unescape(s).replace("\r", "\n")
    units = []
    for line in s.split("\n"):
        line = " ".join(line.replace("•", " ").split()).strip(" -*·")
        if not line:
            continue
        if len(line.split()) > 120:  # one long block: split into sentences so boilerplate can be cut without the rest
            units += [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if x.strip()]
        else:
            units.append(line)
    return units


# ---------- 2b boilerplate ----------
BOILERPLATE = {
    "equal opportunity and legal": r"equal (?:employment )?opportunit|without regard to|protected (?:veteran|class)|affirmative action"
                                   r"|e-?verify|reasonable accommodation|\beeoc?\b|nondiscrimination|fair chance|background check"
                                   r"|drug[- ]free|at[- ]will|privacy (?:notice|policy)|recruitment (?:fraud|scam)|unsolicited",
    "benefits": r"\b401\s*\(?k\)?|\bdental\b|vision (?:insurance|coverage)|paid time off|\bpto\b|parental leave|tuition (?:reimbursement|assistance)"
                r"|employee assistance|wellness (?:program|stipend)|benefits? (?:include|package|offered)|health (?:insurance|benefits)"
                r"|life insurance|retirement (?:plan|savings)|paid holidays",
    "application logistics": r"click (?:apply|here)|apply (?:now|today|online)|submit (?:your )?(?:resume|application)"
                             r"|to apply,|applications? (?:will be )?accepted until",
}
BOILERPLATE_RE = {k: re.compile(p, re.IGNORECASE) for k, p in BOILERPLATE.items()}
HEADING = re.compile(r"^[A-Z][\w &/',’-]{1,60}:?$")  # a short line, starting with a capital letter
ABOUT_HEAD = re.compile(r"^(?:about (?!the (?:role|job|position|team|opportunity))[\w&.,' -]{1,40}|who we are|our (?:company|mission|story|culture|values)"
                        r"|company (?:overview|description))\s*:?$", re.IGNORECASE)
MIN_TEMPLATE_WORDS, MIN_TEMPLATE_COMPANIES = 8, 5
REQUIREMENT_CUE = re.compile(r"\b(?:years?|yrs|experience|degree|bachelor|master|associate|diploma|certif\w*|licen[sc]\w*|clearance"
                             r"|required|requirements?|preferred|qualifications?|proficien\w*|knowledge|ability|skills?|familiar\w*"
                             r"|responsib\w*|you will|must)\b|\d\s*\+", re.IGNORECASE)


def norm_line(x):
    return " ".join(re.sub(r"[^a-z ]", " ", x.lower()).split())


def line_key(x):
    """A stable 64-bit key for a normalized line (template lines are counted by key, not by string, to save memory;
    Python's built-in hash changes between runs, so it can't be used here)."""
    return int.from_bytes(hashlib.blake2b(norm_line(x).encode(), digest_size=8).digest(), "little", signed=True)


def template_keys(units_series, company_ids):
    """Keys of lines (8+ words) that appear word for word in listings from 5+ companies. Learn it on training months only."""
    keys, comps = [], []
    for units, comp in zip(units_series, company_ids):
        for x in set(units):
            if len(x.split()) >= MIN_TEMPLATE_WORDS:
                keys.append(line_key(x))
                comps.append(comp)
    pairs = pd.DataFrame({"k": np.array(keys, dtype=np.int64), "c": pd.Series(comps).astype("category").cat.codes})
    n = pairs.drop_duplicates().groupby("k").size()
    return set(n.index[n >= MIN_TEMPLATE_COMPANIES])


def classify(units, templates):
    """For each line: '' (kept) or the reason it's removed (known boilerplate, the About section, shared template text).
    Lines with a requirement cue are never cut as template text: generic requirements are shared across employers too."""
    out, in_about = [], False
    for x in units:
        if len(x.split()) <= 6 and HEADING.match(x):
            in_about = bool(ABOUT_HEAD.match(x))
        if in_about:
            out.append("about the company")
            continue
        reason = next((k for k, pat in BOILERPLATE_RE.items() if pat.search(x)), "")
        if (not reason and len(x.split()) >= MIN_TEMPLATE_WORDS and line_key(x) in templates
                and not REQUIREMENT_CUE.search(x)):
            reason = "shared template text"
        out.append(reason)
    return out


# ---------- sections: required vs. preferred ----------
REQ_HEAD = re.compile(r"\b(?:requirements?|qualifications?|what you(?:'ll| will)? (?:need|bring)|must[- ]haves?|you (?:have|bring)"
                      r"|who you are|skills|experience|minimum|basic|required)\b", re.IGNORECASE)
PREF_HEAD = re.compile(r"\b(?:preferred|nice[- ]to[- ]haves?|bonus|desired|pluses|a plus|additional)\b", re.IGNORECASE)
PREF_LINE = re.compile(r"\b(?:preferred|nice to have|is a plus|are a plus|a bonus|desired|ideally|helpful)\b", re.IGNORECASE)


def line_sections(lines):
    """For each kept line: 'preferred' if it sits under a preferred-style heading or says preferred/a plus itself,
    otherwise 'required'. Unheaded text counts as required (conservative: a rule may hide on it only after validation)."""
    out, current = [], "required"
    for x in lines:
        if len(x.split()) <= 6 and HEADING.match(x):
            current = "preferred" if PREF_HEAD.search(x) else ("required" if REQ_HEAD.search(x) else current)
        out.append("preferred" if current == "preferred" or PREF_LINE.search(x) else "required")
    return out


# ---------- 3. the blocker rules ----------
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12, "fifteen": 15}
YEARS_NEW = re.compile(r"\b(\d{1,2}|" + "|".join(NUM_WORDS) + r")(?:\s*\(\d{1,2}\))?\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*\d{1,2}\s*\+?\s*)?"
                       r"(?:years?|yrs?)\b(?:\s+of)?[^.;]{0,50}?\b(?:experience|exp\b|background|working)", re.IGNORECASE)
YEARS_LEAD = re.compile(r"\b(?:minimum|at least|min\.?)\s+(?:of\s+)?(\d{1,2}|" + "|".join(NUM_WORDS) + r")\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)
MAX_YEARS = 20  # "50 years of experience" is the company's history, not a requirement
DEGREE_WORDS = {"phd": r"\bph\.?\s?d\b|\bdoctor(?:ate|al)\b", "master": r"\bmaster'?s?\b|\bm\.s\.|\bmba\b",
                "bachelor": r"\bbachelor'?s?\b|\bb\.?s\.?\b|\bb\.?a\.?\b|\bbsc\b|\bundergraduate degree\b|\bfour[- ]year degree\b",
                "associate": r"\bassociate'?s? degree\b|\ba\.a\.s?\.?\b|\btwo[- ]year degree\b"}
DEGREE_ANY = re.compile(r"\bdegree\b|" + "|".join(DEGREE_WORDS.values()), re.IGNORECASE)
DEGREE_LEVEL = {k: re.compile(p, re.IGNORECASE) for k, p in DEGREE_WORDS.items()}
EQUIVALENT = re.compile(r"\bor (?:an? )?(?:equivalent|comparable|related|relevant)\b[^.;]{0,40}\b(?:experience|work|combination)\b"
                        r"|\bequivalent (?:combination|work experience|practical experience|experience)\b|\bin lieu of (?:a )?degree\b"
                        r"|\bdegree or (?:equivalent )?(?:\d+ )?(?:years?|experience)\b", re.IGNORECASE)


def _year(tok):
    tok = tok.lower()
    return NUM_WORDS.get(tok, int(tok) if tok.isdigit() else np.nan)


def years_in(lines):
    """Most years asked across these lines (lower end of each range; up to 20); NaN if none."""
    ys = [_year(m.group(1)) for x in lines for pat in (YEARS_NEW, YEARS_LEAD) for m in pat.finditer(x)]
    ys = [y for y in ys if y == y and 0 < y <= MAX_YEARS]
    return max(ys) if ys else np.nan


# ---------- 4. other text features ----------
def _any(pattern):
    return re.compile(pattern, re.IGNORECASE)


FLAGS = {
    # blockers for some applicants (hidden per applicant in the app once validated)
    "clearance_required": _any(r"\b(?:active|current|must (?:have|hold|possess))\b[^.;]{0,40}\b(?:clearance|ts/sci|top secret|secret)\b"
                               r"|\b(?:ts/sci|top secret|secret|public trust)\b[^.;]{0,20}\bclearance\b(?![^.;]{0,30}\b(?:ability|able|eligible|obtain))"),
    "clearance_obtainable": _any(r"\b(?:ability|able|eligible|willing) to (?:obtain|acquire|get)\b[^.;]{0,40}\bclearance\b"),
    "no_sponsorship": _any(r"\b(?:not|unable to|cannot|can't|won't|will not|do not|does not|no)\b[^.;]{0,25}\bsponsor"
                           r"|\bsponsorship (?:is )?(?:not|un)(?:available| ?available|provided|offered)"
                           r"|\bwithout (?:the )?(?:need for |current or future )?(?:visa )?sponsorship\b"),
    "us_citizen_required": _any(r"\bmust be (?:a )?u\.?s\.? citizen|\bu\.?s\.? citizenship (?:is )?required\b"),
    "shift_or_oncall": _any(r"\bon[- ]call\b|\bnight shift|\bovernight shift|\brotating shift|\bsecond shift|\bthird shift|\bweekends? (?:are )?required"
                            r"|\b24/7 (?:support|coverage|on-call)"),
    "contract_to_hire": _any(r"\bcontract[- ]to[- ](?:hire|perm)|\btemp[- ]to[- ](?:hire|perm)|\bc2h\b"),
    # what a newcomer is offered
    "entry_wording": _any(r"\bentry[- ]level\b|\bno (?:prior |previous )?experience (?:is )?(?:required|necessary|needed)\b"
                          r"|\bwill train\b|\bpaid training\b|\btraining (?:is )?provided\b|\bon[- ]the[- ]job training\b|\bapprenticeships?\b"),
    "changer_wording": _any(r"\bcareer (?:changers?|change|switchers?|transition)\b|\btransferable skills\b|\bnon[- ]traditional (?:backgrounds?|candidates?)\b"
                            r"|\bbootcamp (?:graduates?|grads?)\b|\bself[- ]taught\b"),
    # hiring difficulty and urgency, read in context ("urgent care" is not urgency)
    "sign_on_bonus": _any(r"\bsign[- ]on bonus|\bsigning bonus"),
    "hiring_immediately": _any(r"\bhiring immediately\b|\bimmediate (?:start|opening|hire|need)\b|\bstart immediately\b"),
    "relocation_offered": _any(r"\brelocation (?:assistance|package|support|bonus|is available|available)\b"),
    "urgency_context": _any(r"\burgent(?:ly)? (?:need|needed|hiring|requirement|opening|role)\b|\bimmediate need\b|\bhiring urgently\b"),
    "remote_mention": _any(r"\bremote\b(?! (?:patient|monitoring|sensing|desktop|access|support tools|management))"),
    # evergreen wording: collecting applications rather than filling one job (a ghost-listing signal)
    "evergreen_talent_pool": _any(r"\btalent (?:community|pool|network)\b|\bfuture (?:opportunities|openings|roles)\b"),
    "evergreen_no_opening": _any(r"\bgeneral application\b|\bnot (?:for )?a (?:specific|current) (?:position|opening|role)\b|\bno (?:current|specific) opening"),
    "evergreen_always_hiring": _any(r"\balways (?:accepting|hiring|looking)\b|\brolling basis\b|\bongoing (?:hiring|recruitment|basis)\b"),
    "evergreen_many_openings": _any(r"\bmultiple (?:openings|positions|roles)\b|\bvarious locations\b|\bseveral (?:openings|positions)\b"),
    "evergreen_pipeline": _any(r"\bpipeline (?:req|requisition|role|position)|\bevergreen (?:req|requisition|role|position|posting)"),
    "evergreen_staffing": _any(r"\bon behalf of (?:our|a) client\b|\bour client is (?:seeking|looking)\b"),
}
EVERGREEN = [k for k in FLAGS if k.startswith("evergreen_")]
TRAVEL = re.compile(r"(\d{1,3})\s*%[^.;]{0,30}\btravel|\btravel[^.;]{0,30}?(\d{1,3})\s*%", re.IGNORECASE)
CERTS = {
    "comptia": r"\bcomptia\b|\bsecurity\+|\bnetwork\+|\ba\+ (?:cert|certification)",
    "cisco": r"\bccna\b|\bccnp\b|\bccie\b",
    "security": r"\bcissp\b|\bcism\b|\bcisa\b|\bceh\b|\boscp\b|\bgiac\b|\bgsec\b",
    "cloud": r"\baws certified\b|\baws (?:solutions architect|certification)\b|\bazure (?:certified|certification)\b|\bgoogle cloud certified\b",
    "project": r"\bpmp\b|\bcapm\b|\bscrum master certif|\bcsm\b|\bprince2\b",
    "itil": r"\bitil\b",
}
CERTS_RE = {k: _any(p) for k, p in CERTS.items()}
CERT_REQUIRED_CUE = _any(r"\b(?:required|must (?:have|hold|possess|obtain)|mandatory)\b")


def features(lines):
    """Text features for one listing's cleaned lines; unknown (None / NaN) when there are no lines."""
    if not lines:
        return {}
    sections = line_sections(lines)
    req = [x for x, s in zip(lines, sections) if s == "required"]
    pref = [x for x, s in zip(lines, sections) if s == "preferred"]
    text = "\n".join(lines)
    deg_req = [x for x in req if DEGREE_ANY.search(x)]
    out = {
        "years_asked": years_in(lines),
        "years_required": years_in(req),
        "years_preferred": years_in(pref),
        "degree_mentioned": bool(DEGREE_ANY.search(text)),
        "degree_required": bool(deg_req),
        "degree_preferred": any(DEGREE_ANY.search(x) for x in pref),
        "degree_equivalent": bool(EQUIVALENT.search(text)),
        "degree_level": next((k for k in ("phd", "master", "bachelor", "associate") if any(DEGREE_LEVEL[k].search(x) for x in deg_req)), None),
        "travel_pct": max([int(m.group(1) or m.group(2)) for m in TRAVEL.finditer(text)
                           if 0 < int(m.group(1) or m.group(2)) <= 100] or [np.nan]),
        "cert_mentioned": any(p.search(text) for p in CERTS_RE.values()),
        "cert_required": any(p.search(x) and CERT_REQUIRED_CUE.search(x) for x in req for p in CERTS_RE.values()),
        "words_clean": sum(len(x.split()) for x in lines),
    }
    for k, p in CERTS_RE.items():
        out[f"cert_{k}"] = bool(p.search(text))
    for k, p in FLAGS.items():
        out[k] = bool(p.search(text))
    out["evergreen_any"] = any(out[k] for k in EVERGREEN)
    return out


# Column types, fixed up front so every batch writes the same schema (a batch where a column is all blank would
# otherwise get a different type). Blank = unknown: the listing has no description to read.
NUMERIC = ["years_asked", "years_required", "years_preferred", "travel_pct"]
COUNTS = ["words_clean", "words_raw"]
FLAG_COLS = [c for c in features(["Requirements:", "3+ years of experience"]) if c not in NUMERIC + COUNTS + ["degree_level"]]
FEATURE_COLS = NUMERIC + ["degree_level"] + COUNTS + FLAG_COLS
SCHEMA = pa.schema([("id", pa.string())] + [(c, pa.float32()) for c in NUMERIC] + [("degree_level", pa.string())]
                   + [(c, pa.int32()) for c in COUNTS] + [(c, pa.bool_()) for c in FLAG_COLS])


def typed(f):
    """A batch of features with the fixed column types (nullable, so unknown stays blank)."""
    f = f.reindex(columns=FEATURE_COLS)
    for c in NUMERIC:
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("float32")
    for c in COUNTS:
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("Int32")
    f["degree_level"] = f["degree_level"].astype("string")
    for c in FLAG_COLS:
        f[c] = f[c].astype("boolean")
    return f


# ---------- the batch pipeline ----------
def iter_descriptions(desc_path, batch_size=20_000, ids=None):
    """(id, description) batches from scope_descriptions.parquet, one batch in memory at a time."""
    pf = pq.ParquetFile(desc_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["id", "description"]):
        d = batch.to_pandas()
        yield d[d.id.isin(ids)] if ids is not None else d


def build(desc_path, meta, templates, features_path, clean_path, batch_size=20_000, progress=None):
    """Clean every description and compute its features, writing both to Parquet batch by batch.
    meta: DataFrame indexed by id (the scope table's ids). Returns the words removed per boilerplate reason."""
    removed = {}
    fw = cw = None
    try:
        for n, d in enumerate(iter_descriptions(desc_path, batch_size, ids=set(meta.index))):
            units = d.description.map(to_lines)
            reasons = [classify(u, templates) for u in units]
            kept = [[x for x, r in zip(u, rs) if not r] for u, rs in zip(units, reasons)]
            for u, rs in zip(units, reasons):
                for x, r in zip(u, rs):
                    if r:
                        removed[r] = removed.get(r, 0) + len(x.split())
            f = pd.DataFrame([features(k) for k in kept])
            f["words_raw"] = [sum(len(x.split()) for x in u) for u in units]
            f = typed(f)
            f.insert(0, "id", d.id.to_numpy())
            ft = pa.Table.from_pandas(f, schema=SCHEMA, preserve_index=False)
            ct = pa.table({"id": d.id.to_numpy(), "desc_clean": ["\n".join(k) for k in kept]})
            if fw is None:
                fw = pq.ParquetWriter(features_path, SCHEMA)
                cw = pq.ParquetWriter(clean_path, ct.schema)
            fw.write_table(ft)
            cw.write_table(ct)
            if progress:
                progress(n, len(d))
    finally:
        if fw is not None:
            fw.close()
            cw.close()
    return removed
