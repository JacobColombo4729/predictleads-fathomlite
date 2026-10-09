"""Find career changers into computer and mathematical roles in the public structured resume dataset
(Kaggle: suriyaganesh/resume-dataset-structured, MIT license, version 2 of 2024-11-14).

Each person's resume is read as one timeline: jobs and education in date order, each labelled tech/math, other field,
unclear, or skip. Their original background is everything before their first tech or math entry. They're a career
changer if that background includes another field, however they got into tech (changed jobs, new grad, or retrained).

Every rule lives in RULES, so each review round changes one place. Bump RULES_VERSION whenever a rule changes,
and record the change in the rules log in notebooks/Profiles.ipynb. Nothing here prints rows.
"""
import hashlib
import re

import numpy as np
import pandas as pd

RULES_VERSION = 4

RULES = {
    # "Present" / "Current" end dates mean the month the dataset was last updated
    "as_of": pd.Period("2024-11", "M"),
    # Target roles: computer and mathematical (O*NET 15-xxxx), grouped into buckets for role discovery.
    # The first bucket that matches a title wins, so order matters.
    "cm_buckets": {
        "data and analytics": r"\bdata (?:analyst|scientist|engineer|architect)\b|\bbusiness intelligence\b|\bbi (?:analyst|developer)\b"
                              r"|\bstatistician\b|\bmachine learning\b|\bml engineer\b|\breporting analyst\b",
        "database": r"\bdatabase (?:administrator|developer|analyst|engineer)\b|\bdba\b|\bsql developer\b",
        "quality assurance": r"\b(?:qa|software quality assurance) (?:analyst|engineer|tester|specialist)\b|\bsoftware tester\b|\btest (?:engineer|analyst)\b",
        "security": r"\b(?:cyber ?security|information security|it security) (?:analyst|engineer|specialist)\b",
        "networks and systems": r"\bnetwork (?:administrator|engineer|analyst|technician|specialist)\b|\bsystems? (?:administrator|engineer|analyst)\b"
                                r"|\bsysadmin\b|\bcloud engineer\b|\bdevops\b",
        "it support": r"\bhelp ?desk\b|\bservice desk\b|\bdesktop support\b|\bit support\b|\btechnical support (?:specialist|analyst|engineer|technician)\b"
                      r"|\bcomputer (?:support|technician)\b|\bit technician\b|\bit specialist\b",
        "software development": r"\bsoftware (?:developer|engineer|programmer)\b|\bdeveloper\b|\bprogrammer\b|\bfull[- ]?stack\b|\bfront[- ]?end\b"
                                r"|\bback[- ]?end\b|\bweb developer\b|\bapplication (?:developer|engineer)\b",
    },
    # A first computer and mathematical job at these levels isn't a newcomer's job: the person was probably already in tech
    "senior_target": r"\b(?:senior|sr\.?|lead|principal|staff|manager|director|head|chief|vp|vice president|architect)\b",
    # Pre-switch jobs that don't count as a background to switch from
    "not_a_background": r"\bintern(?:ship)?\b|\bstudent\b|\bvolunteer\b|\b(?:teaching|research|graduate) assistant\b|\bfellow(?:ship)?\b",
    # Job titles outside the target buckets that still sound technical ("IT Manager"); they count as tech in the timeline
    # while tech_titles_count_as_tech is True
    "tech_source": r"\bit\b|\binformation technology\b|\bcomputer\b|\bsoftware\b|\bweb\b|\btechnical\b|\btechnology\b|\bprogramm",
    "tech_titles_count_as_tech": True,
    # Months of jobs in another field needed for a job background (a shorter job background with no other-field study is rejected)
    "min_background_months": 12,
    # Programs in these fields are tech or math. General engineering (mechanical, civil, ...) doesn't count as tech for now.
    "tech_study": r"comput|\binformation (?:technology|systems|science)\b|\binformatics\b|\bsoftware\b|\bit\b|\bcis\b|\bcs\b"
                  r"|mathemat|\bmaths?\b|statistic|\bdata (?:science|analytics)\b|\banalytics\b|\bactuarial\b|cyber|\bnetwork"
                  r"|\belectric(?:al)?\b|\belectronics?\b|\bb\.?\s?tech\b|\bm\.?\s?tech\b|\bbca\b|\bmca\b|\bprogramm"
                  r"|\bcoding\b|\bbootcamp\b|\bweb (?:development|developer)\b|\bfull[- ]?stack\b|\bsoftware development\b",
    "high_school": r"\bhigh school\b|\bged\b|\bsecondary school\b|\bssc\b|\bhsc\b",
    # Words that name a degree level, not a field. A program with nothing left after removing them is "field unclear"
    "degree_words": r"\b(?:bachelors?|masters?|associates?|doctor(?:ate)?|ph\.?\s?d|degree|diploma|certificate|certification|of|in|and|the"
                    r"|science|sciences|arts?|applied|general|studies|honou?rs|hons|program|course(?:work)?|graduate|undergraduate"
                    r"|b\.?\s?s\.?c?|b\.?\s?a|m\.?\s?s\.?c?|m\.?\s?a|a\.?\s?a\.?s?|major|minor|concentration)\b",
    # Degree level, used to place programs with no start date: below late_study_level they go at the start of the
    # timeline (degrees usually come before a career); at or above it, just before the switch (when people retrain).
    # Checked in this order; anything unmatched counts as a bachelor's (2).
    "study_levels": [
        (4, r"\bdoctor|\bph\.?\s?d\b"),
        (3, r"\bmaster|\bmba\b|\bm\.?\s?s\.?c?\b|\bm\.?\s?a\b|\bm\.?\s?tech\b|\bmca\b|\bpost ?grad|\bcertificat|\bbootcamp\b|\bnanodegree\b|\bdiploma\b"),
        (1, r"\bassociate|\ba\.?\s?a\.?s?\b"),
        (2, r"\bbachelor|\bb\.?\s?s\.?c?\b|\bb\.?\s?a\b|\bb\.?\s?tech\b|\bbca\b|\bb\.?\s?e\b|\bundergrad"),
    ],
    "late_study_level": 3,
    # Only people who retrained (a tech or math program between their original field and the tech job) are candidates.
    # True also keeps direct switches (changed jobs, or a new grad straight into tech, with no tech or math training).
    "keep_direct_switches": False,
}

REASONS = {  # review reason codes; "good" means a real career change into a computer and mathematical role
    "good": "A real career change into a computer and mathematical role",
    "not_a_switch": "Same kind of work under a new title",
    "already_in_tech": "The background was already tech work",
    "student_or_intern": "The background is school, an internship or volunteering",
    "wrong_target": "The new role isn't really computer and mathematical",
    "too_senior": "The new role is too senior for a newcomer",
    "studied_tech": "Their original background was tech or math study, so it isn't a change of background",
    "field_unclear": "Can't tell what they studied",
    "garbled": "OCR or extraction errors make it unreadable",
    "dates_wrong": "Dates are missing, impossible or out of order",
    "other": "Anything else (explain in the note)",
}

_MONTH = re.compile(r"^\s*(\d{1,2})\s*[/-]\s*(\d{4})\s*$")
_YEAR = re.compile(r"^\s*(\d{4})\s*$")
_NOW = re.compile(r"^\s*(?:present|current|now|today|to date|ongoing)\s*$", re.IGNORECASE)


def parse_month(value, as_of):
    """MM/YYYY or YYYY to a monthly Period; 'Present' to as_of; anything else to NaT."""
    if not isinstance(value, str):
        return pd.NaT
    if _NOW.match(value):
        return as_of
    m = _MONTH.match(value)
    if m and 1 <= int(m.group(1)) <= 12:
        return pd.Period(year=int(m.group(2)), month=int(m.group(1)), freq="M")
    m = _YEAR.match(value)
    if m:
        return pd.Period(year=int(m.group(1)), month=1, freq="M")
    return pd.NaT


def tag_jobs(experience, rules=RULES):
    """Add parsed dates and title flags to experience rows (person_id, title, start_date, end_date)."""
    jobs = experience.copy()
    jobs["title"] = jobs["title"].fillna("").astype(str).str.strip()
    jobs["start"] = jobs["start_date"].map(lambda v: parse_month(v, rules["as_of"]))
    jobs["end"] = jobs["end_date"].map(lambda v: parse_month(v, rules["as_of"]))
    low = jobs["title"].str.lower()
    jobs["cm_bucket"] = None
    for bucket, pattern in reversed(list(rules["cm_buckets"].items())):  # reversed, so the first bucket wins
        jobs.loc[low.str.contains(pattern, regex=True), "cm_bucket"] = bucket
    jobs["is_cm"] = jobs["cm_bucket"].notna()
    jobs["senior"] = low.str.contains(rules["senior_target"], regex=True)
    jobs["not_background"] = low.str.contains(rules["not_a_background"], regex=True)
    jobs["tech_title"] = low.str.contains(rules["tech_source"], regex=True)
    return jobs


def _months(start, end):
    if pd.isna(start) or pd.isna(end):
        return 0
    return max((end - start).n, 0)


def _study_label(program, rules):
    if re.search(rules["tech_study"], program):
        return "tech"
    if re.search(rules["high_school"], program):
        return "skip"
    field = re.sub(r"[^a-z]+", " ", re.sub(rules["degree_words"], " ", program)).strip()
    return "other" if len(field) >= 3 else "unclear"


def _study_level(program, rules):
    for level, pattern in rules["study_levels"]:
        if re.search(pattern, program):
            return level
    return 2


def person_timeline(jobs_p, edu_p, switch, rules=RULES):
    """One person's jobs and education in one ordered table, each labelled tech / other / unclear / skip.
    `key` orders the timeline: a dated entry sits at its start month (a program just before a job starting the same
    month); an undated program sits at the start (below late_study_level) or just before the switch."""
    rows = []
    for j in jobs_p.dropna(subset=["start"]).itertuples():
        if j.is_cm or (rules["tech_titles_count_as_tech"] and j.tech_title):
            label = "tech"
        elif j.not_background:
            label = "skip"
        else:
            label = "other"
        rows.append(("job", j.title, j.start, j.end, label, True, float(j.start.ordinal), j.is_cm and not j.not_background))
    if edu_p is not None:
        for e in edu_p.itertuples():
            prog = str(e.program).lower().strip() if isinstance(e.program, str) else ""
            if not prog:
                continue
            if pd.notna(e.start):
                key = e.start.ordinal - 0.1
            elif _study_level(prog, rules) >= rules["late_study_level"]:
                key = switch.ordinal - 0.5 + 0.01 * _study_level(prog, rules)
            else:
                key = -1e9 + _study_level(prog, rules)
            rows.append(("study", prog, e.start, pd.NaT, _study_label(prog, rules), pd.notna(e.start), key, False))
    tl = pd.DataFrame(rows, columns=["kind", "text", "start", "end", "label", "dated", "key", "target_job"])
    return tl.sort_values("key", kind="stable").reset_index(drop=True)


def _one_person(g, edu, rules):
    out = {"status": None, "origin_type": None, "switch_month": pd.NaT, "target_title": None, "target_bucket": None,
           "source_title": None, "last_source_title": None, "source_study": None, "retrained": False,
           "retrain_study": None, "study_undated": False, "background_months": 0, "gap_months": np.nan,
           "n_jobs": int(g.start.notna().sum()), "n_jobs_before": 0}
    g = g.dropna(subset=["start"]).sort_values("start")
    if g.empty:
        return {**out, "status": "no dated jobs"}
    cm = g[g.is_cm & ~g.not_background]
    if cm.empty:
        return {**out, "status": "never in a computer and mathematical role"}
    first = cm.iloc[0]
    out.update(switch_month=first.start, target_title=first.title, target_bucket=first.cm_bucket)
    if first.senior:
        return {**out, "status": "first computer and mathematical role is senior"}

    tl = person_timeline(g, edu, first.start, rules)
    before = tl[tl.key < first.start.ordinal]
    out["n_jobs_before"] = int((before.kind == "job").sum())
    out["study_undated"] = bool(((before.kind == "study") & ~before.dated).any())
    tech_keys = before.key[before.label == "tech"]
    first_tech = tech_keys.min() if len(tech_keys) else float(first.start.ordinal)
    origin = before[(before.key < first_tech) & (before.label != "skip")]
    retrain = before[(before.key >= first_tech) & (before.kind == "study") & (before.label == "tech")]
    out["retrained"] = not retrain.empty
    out["retrain_study"] = retrain.text.iloc[-1] if out["retrained"] else None

    other = origin[origin.label == "other"]
    if other.empty:
        out.update(retrained=False, retrain_study=None)  # only career changers can have retrained
        if (origin.label == "unclear").any():
            status = "original field of study unclear"
        elif len(tech_keys):
            status = "original background in tech or math"
        elif (before.label == "skip").any():
            status = "only school, internships or high school before the switch"
        else:
            status = "no background before the switch"
        return {**out, "status": status}

    # Months in other-field jobs, counted up to the first tech entry
    cutoff = pd.Period(ordinal=int(np.ceil(first_tech)), freq="M")
    jobs_o, studies_o = other[other.kind == "job"], other[other.kind == "study"]
    months = [_months(s, min(e, cutoff) if pd.notna(e) else cutoff) for s, e in zip(jobs_o.start, jobs_o.end)]
    out["background_months"] = int(sum(months))
    if len(jobs_o):
        out["source_title"] = jobs_o.text.iloc[int(np.argmax(months))]
        out["last_source_title"] = jobs_o.text.iloc[-1]
        last_end = jobs_o.end.iloc[-1]
        out["gap_months"] = (first.start - last_end).n if pd.notna(last_end) else np.nan
    if len(studies_o):
        out["source_study"] = studies_o.text.iloc[-1]

    job_background = out["background_months"] >= rules["min_background_months"]
    if not job_background and studies_o.empty:
        return {**out, "status": f"background under {rules['min_background_months']} months"}
    out["origin_type"] = "job" if job_background else "study"
    if out["retrained"]:
        status = ("candidate: retrained, after working in another field" if job_background
                  else "candidate: retrained, new grad from another field")
    elif rules["keep_direct_switches"]:
        status = "candidate: changed jobs" if job_background else "candidate: new grad from another field"
    else:
        status = ("changed jobs without tech or math training" if job_background
                  else "new grad from another field without tech or math training")
    return {**out, "status": status}


def prepare_education(education, rules=RULES):
    """Parse education start dates; returns {person_id: rows}."""
    e = education.copy()
    e["start"] = e["start_date"].map(lambda v: parse_month(v, rules["as_of"]))
    return dict(tuple(e.groupby("person_id", sort=False)))


def find_switches(jobs, education=None, rules=RULES):
    """One row per person: a 'candidate: ...' status if their original background was another field and they moved
    into a computer and mathematical role (changed jobs, new grad, or retrained), else the first rule that rejected them."""
    edu_by = prepare_education(education, rules) if education is not None else {}
    rows = {pid: _one_person(g, edu_by.get(pid), rules) for pid, g in jobs.groupby("person_id", sort=False)}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "person_id"
    out["retrain_field"] = out.retrain_study.map(training_field)
    out["switch_period"] = out.switch_month.map(switch_period)
    out["rules_version"] = RULES_VERSION
    return out


# Role buckets for listings: O*NET-SOC codes of computer and mathematical occupations mapped to the same buckets as
# RULES["cm_buckets"] (the resume side), so listings and career switches line up role by role. Matched on the first 7
# characters ("15-1252.00" -> "15-1252"). Both the 2019 SOC codes and the older 2010 ones are listed, since the
# version in the listings isn't confirmed. Codes not listed come back as None; EDA_SDA_2 reports them.
ONET_ROLE_BUCKETS = {
    # 2019 SOC
    "15-1211": "networks and systems",       # computer systems analysts
    "15-1212": "security",                   # information security analysts
    "15-1221": "data and analytics",         # computer and information research scientists
    "15-1231": "networks and systems",       # computer network support specialists
    "15-1232": "it support",                 # computer user support specialists
    "15-1241": "networks and systems",       # computer network architects
    "15-1242": "database",                   # database administrators
    "15-1243": "database",                   # database architects
    "15-1244": "networks and systems",       # network and computer systems administrators
    "15-1251": "software development",       # computer programmers
    "15-1252": "software development",       # software developers
    "15-1253": "quality assurance",          # software quality assurance analysts and testers
    "15-1254": "software development",       # web developers
    "15-1255": "software development",       # web and digital interface designers
    "15-1299": "other computer roles",       # computer occupations, all other
    "15-2011": "credentialed math",          # actuaries
    "15-2021": "credentialed math",          # mathematicians
    "15-2031": "data and analytics",         # operations research analysts
    "15-2041": "data and analytics",         # statisticians
    "15-2051": "data and analytics",         # data scientists
    "15-2099": "data and analytics",         # mathematical science occupations, all other
    # 2010 SOC
    "15-1111": "data and analytics",         # computer and information research scientists
    "15-1121": "networks and systems",       # computer systems analysts
    "15-1122": "security",                   # information security analysts
    "15-1131": "software development",       # computer programmers
    "15-1132": "software development",       # software developers, applications
    "15-1133": "software development",       # software developers, systems software
    "15-1134": "software development",       # web developers
    "15-1141": "database",                   # database administrators
    "15-1142": "networks and systems",       # network and computer systems administrators
    "15-1143": "networks and systems",       # computer network architects
    "15-1151": "it support",                 # computer user support specialists
    "15-1152": "networks and systems",       # computer network support specialists
    "15-1199": "other computer roles",       # computer occupations, all other
}


def role_bucket(onet_code):
    """The role bucket of a listing's O*NET-SOC code; None if it isn't a mapped computer and mathematical code."""
    if not isinstance(onet_code, str):
        return None
    return ONET_ROLE_BUCKETS.get(onet_code.strip()[:7])


# Broad areas for describing where career changers come from (analysis only: changing these never changes who is a
# candidate, so no RULES_VERSION bump). Applied to the main background job title, or the field studied. First match wins.
BACKGROUND_AREAS = {
    "customer service and support": r"customer (?:service|support|care|success)|call cent|client (?:service|support)|help line",
    "education": r"teach|tutor|instructor|professor|educat|school|lecturer|curriculum",
    "healthcare": r"nurs|medical|health|clinic|pharma|patient|dental|therap|caregiver|physician|hospital",
    "food and hospitality": r"server|barista|cook|chef|restaurant|hospitality|hotel|bartend|food|waiter|waitress|kitchen",
    "sales and retail": r"sales|retail|cashier|store|merchandis|account executive|shop",
    "finance and accounting": r"account|financ|bank|audit|tax|bookkeep|econom|insurance|actuar|loan|credit",
    "administration and office": r"admin|office|clerk|reception|secretar|assistant|data entry|coordinator",
    "business, marketing and management": r"business|marketing|manag|operations|mba|project|consult|human resources|\bhr\b|recruit",
    "science and research": r"research|scientist|lab|physics|chemi|biolog|geolog|environment",
    "engineering (not software)": r"engineer|mechanical|civil|industrial|architect|drafting",
    "arts, media and communications": r"design|\bart|music|writ|journalis|media|communicat|editor|photograph|film|graphic",
    "humanities and social sciences": r"english|history|philosoph|psycholog|sociolog|politic|literature|language|anthropolog|social",
    "military and public service": r"army|navy|military|air force|marine|police|government|public|officer|security guard",
    "logistics and trades": r"warehouse|driver|logistic|supply|mechanic|electrician|construction|technician|maintenance|install",
}


def background_area(text):
    """The broad area of a job title or field of study; 'other' if none matches."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    low = text.lower()
    return next((area for area, pattern in BACKGROUND_AREAS.items() if re.search(pattern, low)), "other")


# What retrained people trained into (analysis only; never changes who is a candidate). First match wins, so order matters.
TRAINING_FIELDS = {
    "data science and analytics": r"data (?:science|analytics)|\banalytics\b|machine learning|artificial intelligence|\bai\b|business intelligence",
    "mathematics and statistics": r"mathemat|\bmaths?\b|statistic|actuar",
    "cybersecurity": r"cyber|security",
    "networking": r"network|\bccna\b|\bcisco\b",
    "computer applications (BCA / MCA)": r"\bbca\b|\bmca\b|computer application",
    "computer science": r"computer science|computing|\bcs\b",
    "information technology and systems": r"information (?:technology|systems|science)|\bit\b|\bcis\b|\bmis\b|informatics",
    "software and web development": r"software|\bweb\b|coding|bootcamp|full[- ]?stack|programm|developer",
    "computer engineering and electronics": r"comput|electr",
    "tech degree, field not named": r"\bb\.?\s?tech|\bm\.?\s?tech|technolog",
}


def training_field(text):
    """The group of a tech or math program; None if there's no program."""
    if not isinstance(text, str) or not text.strip():
        return None
    low = text.lower()
    return next((f for f, pattern in TRAINING_FIELDS.items() if re.search(pattern, low)), "other tech or math")


SWITCH_PERIODS = [(1999, "before 2000"), (2004, "2000-2004"), (2009, "2005-2009"), (2014, "2010-2014"), (2019, "2015-2019")]


def switch_period(month):
    """Grouped range of the switch date; 'not dated' when there's no date."""
    if pd.isna(month):
        return "not dated"
    return next((label for last, label in SWITCH_PERIODS if month.year <= last), "2020 or later")


def fingerprints(experience, education=None):
    """One fingerprint per person: every job (title and raw dates) plus every program (name and raw start), sorted,
    so the same resume under different person_ids gets the same fingerprint. Exact copies only; OCR variants differ."""
    def norm(s):
        return s.fillna("").astype(str).str.lower().str.strip()
    parts = [pd.DataFrame({"person_id": experience.person_id,
                           "item": "j|" + norm(experience.title) + "|" + norm(experience.start_date) + "|" + norm(experience.end_date)})]
    if education is not None:
        parts.append(pd.DataFrame({"person_id": education.person_id,
                                   "item": "e|" + norm(education.program) + "|" + norm(education.start_date)}))
    items = pd.concat(parts)
    items = items[items.person_id.isin(experience.person_id)]
    joined = items.groupby("person_id").item.agg(lambda v: "\n".join(sorted(v)))
    return joined.map(lambda s: hashlib.md5(s.encode()).hexdigest())


def review_summary(reviews):
    """Counts and precision from review files: aggregates only, safe to share. Notes are never included."""
    r = reviews.assign(good=reviews.verdict.eq("good"), bad=reviews.verdict.eq("bad"))
    by = r.groupby(["round", "rules_version", "pool"]).agg(
        reviewed=("verdict", lambda v: v.isin(["good", "bad", "unsure"]).sum()),
        good=("good", "sum"), bad=("bad", "sum"), unsure=("verdict", lambda v: v.eq("unsure").sum()))
    by["precision"] = (by.good / (by.good + by.bad)).round(2)
    reasons = (r[r.verdict.eq("bad")].groupby(["round", "pool"]).reason.value_counts()
               .rename("bad candidates").reset_index())
    return by, reasons
