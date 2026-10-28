"""A resume reader that runs on this computer: no OpenAI, nothing leaves the machine (Jacob's app: run_jacob.ps1 / .sh,
PREDICTLEADS_RESUME_EXTRACTION=local). Jesse's app keeps the OpenAI reader (resume_profile_extraction.OpenAIResumeProvider).

It returns the same fields the OpenAI reader does, so Magic fill, its grounding check (every value must appear in the
resume) and the Profile page's editable blocks work unchanged:
  - sections from the resume's own headings (Experience, Education, Projects, Skills, ...);
  - one entry per job, project or school: a date range ("Jan 2020 - Present") anchors an entry, the short lines around
    it give the title and the organization, the lines after it the description. The dates, copied as written, open
    the description, since the form has no date field (ML_Fit's live scoring reads them back from there);
  - skills: the items of the Skills section as written, plus the tools O*NET names that the entries mention, found with
    the same vocabulary and matcher ML_Fit uses on the listings (analysis/src/onet.py; O*NET 31.0, CC BY 4.0);
  - name, headline and location from the lines above the first heading.
Rules, not a model: ordinary one-column resumes read well; two-column or heavily designed PDFs can come out in a
scrambled order. Whatever it reads goes into the form for the user to check before saving.
"""
from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]  # fetch/ and analysis/ are shared at the top
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "analysis") not in sys.path:
    sys.path.append(str(REPO_ROOT / "analysis"))  # last, so nothing in analysis/ shadows the app's own modules

VERSION = "local-resume-reader-v1"

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
          r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_POINT = rf"(?:{_MONTH}\s*,?\s*(?:19|20)\d{{2}}|\d{{1,2}}\s*/\s*(?:19|20)?\d{{2}}|(?:19|20)\d{{2}})"
_END = rf"(?:{_POINT}|present|current|now|today|ongoing)"
DATE_RANGE = re.compile(rf"(?<![\w/]){_POINT}\s*(?:-|–|—|to|until|through)\s*{_END}(?![\w/])", re.IGNORECASE)
SINGLE_DATE = re.compile(rf"(?<![\w/])(?:{_MONTH}\s*,?\s*(?:19|20)\d{{2}}|(?:19|20)\d{{2}})(?![\w/])", re.IGNORECASE)
BULLET = re.compile(r"^\s*(?:[-*•●▪◦·‣∙■➢>]|[-]|o\s)\s*")
_STATES = ("Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|"
           "Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|"
           "Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|"
           "Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
           "West Virginia|Wisconsin|Wyoming|District of Columbia")
_STATE_CODES = ("AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|"
                "OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC")
# "City, ST" or "City, State" (US): a real state is required, so "Acme Corp, Austin" isn't read as a place
LOCATION = re.compile(rf"\b[A-Z][a-zA-Z.'-]+(?: [A-Z][a-zA-Z.'-]+){{0,2}},\s*(?:{_STATE_CODES}|{_STATES})\b")
REMOTE = re.compile(r"^\(?(?:remote|hybrid|on[- ]site)\)?$", re.IGNORECASE)
SEPARATORS = re.compile(r"\s*(?:\||•|·|\s–\s|\s—\s|\s-\s|,\s|\sat\s|\s@\s|\s{3,})\s*")  # a wide gap = a column

HEADINGS = {
    "job": ["experience", "work experience", "professional experience", "employment", "employment history", "work history",
            "career history", "relevant experience", "professional background", "career experience", "experience summary",
            "volunteer experience", "volunteer", "volunteering", "internships", "internship experience",
            "leadership experience", "teaching experience", "industry experience"],
    "education": ["education", "education and training", "academic background", "academics", "training", "certifications",
                  "certificates", "certification", "licenses and certifications", "certifications and licenses",
                  "certifications and training", "education and certifications", "professional development",
                  "courses", "relevant coursework", "coursework"],
    "project": ["projects", "personal projects", "selected projects", "academic projects", "portfolio", "key projects",
                "side projects", "project experience", "technical projects", "capstone projects", "portfolio projects"],
    "skills": ["skills", "technical skills", "core skills", "core competencies", "competencies", "tools", "technologies",
               "tools and technologies", "skills and tools", "areas of expertise", "expertise", "technical proficiencies",
               "key skills", "skills summary", "languages and tools", "technical expertise", "software", "tech stack",
               "skills and certifications", "skills and abilities", "proficiencies", "computer skills"],
    "summary": ["summary", "professional summary", "profile", "about", "about me", "objective", "career objective",
                "overview", "professional profile", "career summary", "summary of qualifications", "qualifications"],
    "other": ["awards", "honors", "honors and awards", "awards and honors", "interests", "hobbies", "references",
              "publications", "languages", "activities", "affiliations", "additional information", "memberships",
              "contact", "contact information"],
}
SECTION_OF = {h: kind for kind, heads in HEADINGS.items() for h in heads}
TITLE_WORDS = re.compile(
    r"\b(?:engineer|developer|analyst|scientist|manager|director|specialist|assistant|associate|coordinator|consultant|"
    r"administrator|technician|designer|architect|lead|intern|teacher|instructor|tutor|nurse|representative|agent|"
    r"officer|clerk|supervisor|accountant|advisor|adviser|owner|founder|president|executive|programmer|tester|"
    r"strategist|researcher|editor|writer|planner|operator|cashier|server|bartender|driver|mechanic|electrician|"
    r"paralegal|therapist|pharmacist|recruiter|buyer|sales|support|head|chief|vp|partner|volunteer|fellow|student|"
    r"barista|chef|cook|attendant|aide|educator|counselor|librarian|producer|marketer|merchandiser|auditor|"
    r"bookkeeper|scheduler|dispatcher|inspector|soldier|sergeant|specialist|professor|lecturer|trainer|apprentice)s?\b",
    re.IGNORECASE)
DEGREE_WORDS = re.compile(
    r"\b(?:bachelor|master|doctor|associate|b\.?\s?[as]\.?|b\.?sc|m\.?\s?[as]\.?|m\.?sc|mba|ph\.?\s?d|diploma|certificate|"
    r"certification|certified|bootcamp|program|degree|ged|high school|a\.?a\.?s?|course|nanodegree|training)\b",
    re.IGNORECASE)
ORG_WORDS = re.compile(
    r"\b(?:inc|llc|ltd|corp|corporation|company|co|group|university|college|institute|school|academy|hospital|bank|"
    r"agency|labs?|technologies|solutions|systems|services|partners|consulting|foundation|center|centre|department|"
    r"district|county|city of|state of|army|navy|air force|marines|tripleten|coursera|udemy|udacity|edx|google|"
    r"amazon|microsoft)\b", re.IGNORECASE)
STUDENT = re.compile(r"\bstudent\b", re.IGNORECASE)
MAX_HEADER_LINES = 2


def _heading_key(line: str) -> str:
    return " ".join(re.sub(r"[^a-z ]+", " ", line.casefold().replace("&", " and ")).split())


def _section_of(line: str) -> str | None:
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped.split()) > 5 or BULLET.match(line):
        return None
    return SECTION_OF.get(_heading_key(stripped))


def _clean(line: str) -> str:
    return " ".join(BULLET.sub("", line).split())


def _headerish(line: str) -> bool:
    """A short line that can be a title, an organization or a date line (not a bullet, not a sentence)."""
    text = line.strip()
    return bool(text) and not BULLET.match(line) and len(text.split()) <= 14 and not text.endswith(".")


def _sections(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """The lines above the first heading, and (kind, lines) for each section in order."""
    header: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        kind = _section_of(line)
        if kind is not None:
            sections.append((kind, []))
        elif sections:
            sections[-1][1].append(line)
        else:
            header.append(line)
    if not sections:  # no recognisable headings: read everything below the name block as experience
        sections = [("job", header[3:])]
        header = header[:3]
    return header, sections


def _parts(text: str) -> list[str]:
    """The pieces of a title/organization line, with dates, locations and 'Remote' removed."""
    text = DATE_RANGE.sub(" | ", text)
    text = LOCATION.sub(" | ", text)
    pieces = [p.strip(" ()[]|,;:-–—") for p in SEPARATORS.split(text)]
    return [p for p in pieces if p and not REMOTE.match(p) and not SINGLE_DATE.fullmatch(p) and re.search(r"[A-Za-z]", p)]


def _title_and_org(header_lines: list[str], kind: str) -> tuple[str, str]:
    parts = [p for line in header_lines for p in _parts(line)]
    if not parts:
        return "", ""
    title_re = DEGREE_WORDS if kind == "education" else TITLE_WORDS
    title = next((p for p in parts if title_re.search(p) and not ORG_WORDS.search(p)), None)
    title = title or next((p for p in parts if title_re.search(p)), None) or parts[0]
    rest = [p for p in parts if p != title]
    org = next((p for p in rest if ORG_WORDS.search(p)), None) or (rest[0] if rest else "")
    return title, org


def _entries(kind: str, lines: list[str]) -> list[dict[str, str]]:
    """Split one section into entries: each date range (or, without any, each short line before a bullet) starts one."""
    n = len(lines)
    dated = [bool(DATE_RANGE.search(line)) for line in lines]
    if kind == "education":  # a degree often has only its year ("B.S. Mathematics, State University, 2015")
        dated = [d or (_headerish(line) and bool(SINGLE_DATE.search(line))) for d, line in zip(dated, lines)]
    if any(dated):
        anchors = [i for i in range(n) if dated[i]]
    else:
        anchors = [i for i in range(n) if _headerish(lines[i]) and i + 1 < n and not _headerish(lines[i + 1])]
        if not anchors:
            anchors = [i for i in range(n) if _headerish(lines[i])]
    starts = []
    floor = 0
    for a in anchors:
        s = a
        while s - 1 >= floor and a - (s - 1) <= MAX_HEADER_LINES and _headerish(lines[s - 1]) and not dated[s - 1]:
            s -= 1
        starts.append(s)
        floor = a + 1
    out = []
    for k, (a, s) in enumerate(zip(anchors, starts)):
        end = starts[k + 1] if k + 1 < len(anchors) else n
        header = lines[s:a + 1]
        body_from = a + 1
        if len([p for line in header for p in _parts(line)]) < 2:  # e.g. "Company   Jan 2020 - Present" then "Title"
            while body_from < end and body_from - a <= 1 and _headerish(lines[body_from]) and not dated[body_from]:
                header = header + [lines[body_from]]
                body_from += 1
        title, org = _title_and_org(header, kind)
        if not title:
            continue
        date = next((m.group(0) for line in header for m in [DATE_RANGE.search(line) or SINGLE_DATE.search(line)] if m), "")
        body: list[str] = []
        for line in lines[body_from:end]:
            text = _clean(line)
            if not text:
                continue
            if body and not BULLET.match(line) and body[-1][-1] not in ".!?:" and (text[0].islower() or not _headerish(line)):
                body[-1] += " " + text  # a wrapped line continues the bullet above it
            else:
                body.append(text)
        sentences = [b if b[-1] in ".!?" else b + "." for b in body]
        description = " ".join(([f"{date}." if date else ""]) + sentences).strip()
        entry_kind = "education" if kind == "education" or STUDENT.search(title) else kind
        out.append({"kind": entry_kind, "title": title, "organization": org or None, "description": description,
                    "source_quote": title})
    return out


def _skill_items(lines: list[str]) -> list[tuple[str, str]]:
    """(skill, the line it is on) for each item of a Skills section; "Languages:" style labels are dropped. The whole
    line is the quote, so one-letter skills such as R still pass the app's grounding check."""
    items = []
    for line in lines:
        text = _clean(line)
        for segment in re.split(r"\s*[|•·]\s*|\s{3,}", BULLET.sub("", line).strip()):  # a wide gap separates items too
            if ":" in segment:
                segment = segment.split(":", 1)[1]
            for piece in re.split(r"\s*[,;/]\s*", segment):
                piece = piece.strip(" .()[]")
                if piece and len(piece) <= 40 and 1 <= len(piece.split()) <= 5 and re.search(r"[A-Za-z]", piece):
                    items.append((piece, text))
    return items


@lru_cache(maxsize=1)
def _onet_matcher():
    """ML_Fit's skill vocabulary and matcher over O*NET's tool names, or None when O*NET isn't downloaded
    (python -m fetch.get_onet); then skills come from the Skills section only."""
    try:
        from fetch.paths import external
        from src import onet

        folder = external("onet")
        if not (folder / "Software Skills.txt").exists():
            return None
        import pandas as pd

        no_skill_elements = pd.DataFrame({"element": [], "element_id": [], "code": []})  # tools only
        vocab = onet.vocabulary(onet.software(folder), [no_skill_elements])
        return onet.Matcher(vocab[vocab.kind.eq("tool")])
    except Exception:  # noqa: BLE001  the reader still works without the vocabulary
        return None


def _header_fields(header: list[str]) -> dict[str, str | None]:
    # each piece of the top lines on its own: "Name      City, ST" or "Title | City, ST" are two pieces
    lines = [p.strip() for line in header for p in re.split(r"\s{3,}|[|•·]", line) if p.strip()]
    name =next((l for l in lines[:3] if 2 <= len(l.split()) <= 4 and re.fullmatch(r"[A-Za-z][A-Za-z.' -]+", l)), None)
    location = next((m.group(0) for l in lines for m in [LOCATION.search(l)] if m), None)
    headline = None
    for l in lines:
        if l == name or (location and l.strip() == location) or len(l.split()) > 14 or DATE_RANGE.search(l):
            continue
        rest = [p for p in re.split(r"\s*[|•·]\s*", l) if p and p != location and re.search(r"[A-Za-z]{3}", p)]
        if rest:
            headline = rest[0].strip()
            break
    return {"name": name, "headline": headline, "location": location}


def read_resume(text: str, matcher: Any = None) -> dict[str, Any]:
    """The resume as the OpenAI reader would return it (name, headline, location, experience, skills; every value with
    the resume text it came from). `matcher` defaults to ML_Fit's O*NET tool matcher when O*NET is downloaded."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    header, sections = _sections(lines)
    entries = [e for kind, body in sections if kind in ("job", "project", "education") for e in _entries(kind, body)]
    order = {"job": 0, "project": 1, "education": 2}
    entries = sorted(entries, key=lambda e: order[e["kind"]])  # stable: each kind keeps the resume's order
    fields = _header_fields(header)
    skills: list[dict[str, str]] = []
    seen: set[str] = set()

    def used_in(name: str) -> str:
        key = name.casefold()
        hit = next((e for e in entries if e["kind"] != "education" and key in (e["title"] + " " + e["description"]).casefold()), None)
        return f"used in the {hit['title']} role" if hit and hit["kind"] == "job" else (f"{hit['title']} project" if hit else "")

    for kind, body in sections:
        if kind == "skills":
            for item, line in _skill_items(body):
                if item.casefold() not in seen:
                    seen.add(item.casefold())
                    skills.append({"name": item, "evidence": used_in(item), "source_quote": line})

    def listed(name: str) -> bool:  # O*NET's "Microsoft Excel" when the resume already lists "Excel"
        words = set(re.findall(r"[a-z0-9+#]+", name.casefold()))
        return name.casefold() in seen or any(s in words for s in seen)

    matcher = _onet_matcher() if matcher is None else matcher
    if matcher is not None:
        for e in entries:
            if e["kind"] == "education":
                continue
            for line in [e["title"]] + re.split(r"(?<=[.!?])\s+", e["description"]):
                for name, _, _ in sorted(matcher.find(line)):
                    if not listed(name):
                        seen.add(name.casefold())
                        where = f"used in the {e['title']} role" if e["kind"] == "job" else f"{e['title']} project"
                        skills.append({"name": name, "evidence": where, "source_quote": line[:300]})
    return {"name": fields["name"], "name_quote": fields["name"],
            "headline": fields["headline"], "headline_quote": fields["headline"],
            "location": fields["location"], "location_quote": fields["location"],
            "experience": entries, "skills": skills}


def pdf_to_text(data: bytes) -> str:
    """The PDF's text with its lines as laid out on the page (pypdf's layout mode), for the rules above. The app's
    shared reader (resume_profile_extraction.pdf_to_text) uses pypdf's default mode, which gives some PDFs one word per
    line and then joins them into one line: fine for a model, no structure left for rules. Same checks and limits."""
    import io

    import resume_profile_extraction as extraction
    from grounded_match_explanations import ResponseValidationError

    if len(data) > extraction.MAX_PDF_BYTES:
        raise ResponseValidationError("That PDF is too large for a resume (limit 2 MB).")
    if not data.startswith(b"%PDF"):
        raise ResponseValidationError("That file is not a PDF.")
    from pypdf import PdfReader

    try:
        pdf = PdfReader(io.BytesIO(data))
        if pdf.is_encrypted:
            raise ResponseValidationError("That PDF is password-protected.")
        if len(pdf.pages) > extraction.MAX_PDF_PAGES:
            raise ResponseValidationError(f"That PDF has more than {extraction.MAX_PDF_PAGES} pages; a resume should be shorter.")
        text = "\n".join((page.extract_text(extraction_mode="layout") or "") for page in pdf.pages)
    except ResponseValidationError:
        raise
    except Exception:
        raise ResponseValidationError("That PDF could not be read.") from None
    # keep each line and its column gaps, but not the page's padding (a wide gap becomes four spaces)
    lines = [re.sub(r" {4,}", "    ", line.rstrip()) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line.strip())
    if len(text.strip()) < extraction.MIN_RESUME_CHARS:
        raise ResponseValidationError("No readable text was found in that PDF (it may be a scan). Try a text-based PDF or a .txt file.")
    return text


class LocalResumeProvider:
    """resume_profile_extraction.ResumeProvider that reads the resume on this computer."""

    model = VERSION
    reader = "local"  # what the Profile page tells the user (nothing is sent anywhere)

    def generate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return read_resume(str(payload.get("resume_text") or ""))

    @staticmethod
    def pdf_to_text(data: bytes) -> str:
        return pdf_to_text(data)
