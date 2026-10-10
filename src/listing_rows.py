"""One cleaned row per job listing: EDA_SDA_1's section 6.1, shared by the notebooks and the full-file stream.

lean_part(flattened_part) -> one row per listing with the columns EDA_SDA_1 keeps, the text flags (unknown, not False,
when there's no description), description length, the repost fingerprint and the posting source. No text is kept.
EDA_SDA_3 validates the keyword rules; change them here so every notebook and the stream use the same version.
"""
import hashlib
import html
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

COLS = {  # flattened column -> short name
    "id": "id",
    "relationships.company.data.id": "company_id",
    "attributes.title": "title",
    "attributes.first_seen_at": "first_seen",
    "attributes.last_seen_at": "last_seen",
    "attributes.posted_at": "posted",
    "attributes.status": "status",
    "attributes.seniority": "seniority",
    "attributes.language": "language",
    "attributes.onet_data.code": "onet_code",
    "attributes.onet_data.family": "occupation_group",
    "attributes.onet_data.occupation_name": "occupation",
    "attributes.salary_data.salary_low_usd": "salary_low_usd",
    "attributes.salary_data.salary_high_usd": "salary_high_usd",
    "attributes.salary_data.salary_time_unit": "salary_unit",
    "attributes.contract_types": "contract_types",
    "attributes.location_data": "location_data",
    "attributes.tags": "tags",
    "attributes.categories": "categories",
}

# ---------- text patterns ----------
TAG_RE = re.compile(r"<[^>]+>")
URGENT_RE = r"\burgent|\bimmediate(?:ly)?\b|\basap\b"

# Career-change accessibility: what a listing asks of someone new to the field (keyword rules; EDA_SDA_3 validates them)
YEARS_RE = re.compile(r"\b(\d{1,2})\s*(?:\+|(?:-|–|to)\s*\d{1,2}\s*\+?)?\s*(?:years?|yrs?)\b[^.;]{0,40}?\bexperience", re.IGNORECASE)
MAX_YEARS = 20  # "50 years of experience" is the company's history, not a requirement
ENTRY_RE = (r"\bentry[- ]level\b|\bno (?:prior |previous )?experience (?:is )?(?:required|necessary|needed)\b"
            r"|\bwill train\b|\bpaid training\b|\btraining (?:is )?provided\b|\bon[- ]the[- ]job training\b|\bapprenticeships?\b")
CHANGER_RE = r"\bcareer (?:changers?|change|switchers?|transition)\b|\btransferable skills\b|\bnon[- ]traditional (?:backgrounds?|candidates?)\b"
DEGREE_RE = r"\b(?:bachelor|master|associate)(?:'s|’s|s)?\b(?:\s+\S+){0,6}?\s+(?:is\s+)?required\b|\bdegree\s+(?:is\s+)?required\b"
EQUIV_RE = r"\bor equivalent (?:\w+ )?experience\b|\bequivalent combination of education and experience\b"

# ---------- posting source ----------
# Hiring platforms (applicant tracking systems) and job boards, matched against the url's domain
ATS = {"greenhouse.io": "Greenhouse", "lever.co": "Lever", "myworkdayjobs.com": "Workday", "workday.com": "Workday",
       "smartrecruiters.com": "SmartRecruiters", "icims.com": "iCIMS", "ashbyhq.com": "Ashby", "bamboohr.com": "BambooHR",
       "jobvite.com": "Jobvite", "taleo.net": "Taleo", "successfactors": "SuccessFactors", "recruitee.com": "Recruitee",
       "workable.com": "Workable", "breezy.hr": "Breezy", "applytojob.com": "JazzHR", "paylocity.com": "Paylocity",
       "ultipro.com": "UKG", "ukg.com": "UKG", "adp.com": "ADP", "dayforcehcm.com": "Dayforce", "oraclecloud.com": "Oracle",
       "teamtailor.com": "Teamtailor", "personio": "Personio", "rippling.com": "Rippling", "jobs.lever": "Lever"}
BOARDS = {"indeed.com": "Indeed", "linkedin.com": "LinkedIn", "ziprecruiter.com": "ZipRecruiter",
          "glassdoor.com": "Glassdoor", "monster.com": "Monster", "dice.com": "Dice", "builtin.com": "Built In",
          "wellfound.com": "Wellfound"}


def years_required(text):
    """Most years of experience the listing asks for (lower end of each range); NaN if it states none."""
    years = [int(y) for y in YEARS_RE.findall(text) if int(y) <= MAX_YEARS]
    return max(years) if years else np.nan


def source_of(url):
    """(source_type, platform) from the url's domain; company sites aren't named."""
    if not isinstance(url, str):
        return ("(missing)", None)
    host = urlparse(url).netloc.lower()
    for table, kind in ((ATS, "hiring platform"), (BOARDS, "job board")):
        for key, name in table.items():
            if key in host:
                return (kind, name)
    return ("company site / other", None)


def clean_text(s):
    """Strip HTML tags, decode entities like &amp;, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    return " ".join(html.unescape(TAG_RE.sub(" ", s)).split())


def text_flag(text, has_desc, pattern):
    """True / False from the description; unknown (NA) when there's no description to read."""
    return text.str.contains(pattern, case=False).astype("boolean").where(has_desc, pd.NA)


def first_country(locs):
    """Country of the first location; postings can list several."""
    if isinstance(locs, (list, np.ndarray)) and len(locs) and isinstance(locs[0], dict):
        return locs[0].get("country")
    return None


def lean_part(d):
    """One flattened part -> lean rows (EDA_SDA_1 6.1e)."""
    out = d.reindex(columns=list(COLS)).rename(columns=COLS)
    raw = d.reindex(columns=["attributes.description"])["attributes.description"]
    text = raw.map(clean_text)
    out["has_desc"] = text.str.len() > 0
    out["has_html"] = raw.fillna("").str.contains(TAG_RE)
    out["desc_words"] = text.str.split().str.len()
    in_title = out["title"].fillna("").str.contains(r"\bremote\b", case=False)
    out["mentions_remote"] = (in_title | text.str.contains(r"\bremote\b", case=False)).astype("boolean").where(out["has_desc"] | in_title, pd.NA)
    out["urgency"] = text_flag(text, out["has_desc"], URGENT_RE)
    out["years_required"] = text.map(years_required)
    out["entry_wording"] = text_flag(text, out["has_desc"], ENTRY_RE)
    out["changer_wording"] = text_flag(text, out["has_desc"], CHANGER_RE)
    out["degree_required"] = text_flag(text, out["has_desc"], DEGREE_RE)
    out["equivalent_experience"] = text_flag(text, out["has_desc"], EQUIV_RE)
    out["desc_hash"] = text.str.lower().str[:1000].map(lambda t: hashlib.md5(t.encode()).hexdigest() if t else None)
    out["n_locations"] = out["location_data"].map(lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 0)
    out["country"] = out.pop("location_data").map(first_country)
    src = d.reindex(columns=["attributes.url"])["attributes.url"].map(source_of)
    out["source_type"] = src.str[0]
    out["platform"] = src.str[1]
    return out
