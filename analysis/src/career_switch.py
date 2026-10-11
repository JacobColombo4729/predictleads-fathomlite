"""Find career changers into computer and mathematical roles in the public structured resume dataset
(Kaggle: suriyaganesh/resume-dataset-structured, MIT license, version 2 of 2024-11-14).

Nobody who reached a computer and mathematical job is thrown away: each person gets labels (origin,
retrained, first_level, roles_reached), and `status` names the main one. Career changers are a view: the "candidate: ..." statuses,
in three pools: retrained after working in another field, retrained as a new grad from another field (another-field
origin, then a tech or math program, then the tech job), and changed roles within tech.

Each person's resume is read as one timeline: jobs and education in date order, each labelled tech/math, other field,
unclear, or skip. Their original background is everything before their first tech or math entry. They're a career
changer if that background includes another field, however they got into tech (changed jobs, new grad, or retrained).

Every rule lives in RULES, so each review round changes one place. Nothing here prints rows.
"""
import hashlib
import re

import numpy as np
import pandas as pd


RULES = {
    # "Present" / "Current" end dates mean the month the dataset was last updated
    "as_of": pd.Period("2024-11", "M"),
    # Target roles: computer and mathematical (O*NET 15-xxxx), grouped into buckets for role discovery.
    # The first bucket that matches a title wins, so order matters.
    "cm_buckets": {
        # Added 2026-10-10: IT project managers are 13% of the scope's listings (O*NET 15-1299.09). A plain
        # "project manager" with no tech word isn't matched, since it could be any field.
        "it project management": r"\b(?:it|i\.t\.|technical|technology|software|systems?|infrastructure|erp|digital|data|cloud"
                                 r"|application|web) (?:project|program|delivery) manager\b|\btechnical program manager\b"
                                 r"|\b(?:it|technical) pm\b|\bscrum master\b|\bagile (?:coach|project manager|delivery lead)\b",
        "data science": r"\bdata (?:analyst|scientist|engineer|architect|specialist|management|modell?er|warehouse)\b"
                              r"|\bbusiness intelligence\b|\bbi (?:analyst|developer|engineer)\b|\bpower ?bi\b|\btableau\b"
                              r"|\betl (?:developer|engineer)\b|\banalytics\b|\bquantitative analyst\b|\bmis analyst\b"
                              r"|\bbig data\b|\bhadoop\b|\bdata visuali[sz]ation\b|\bdecision scien"
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
    # Level of the job at the switch, recorded in first_level as a label and model feature; no level is left out
    # (prior experience can qualify someone for a senior or manager role).
    # Project, program and delivery managers aren't people managers, so they don't count as "manager or above"
    "manager_target": r"\b(?:(?<!project )(?<!program )(?<!delivery )manager|director|head|chief|vp|vice president|president|cto|cio|ceo)\b",
    "senior_target": r"\b(?:senior|sr\.?|lead|principal|staff|architect)\b",
    # Pre-switch jobs that don't count as a background to switch from
    "not_a_background": r"\bintern(?:ship)?\b|\bstudent\b|\bvolunteer\b|\b(?:teaching|research|graduate) assistant\b|\bfellow(?:ship)?\b",
    # Job titles outside the target buckets that still sound technical ("IT Manager"); they count as tech in the timeline
    # while tech_titles_count_as_tech is True
    "tech_source": r"\bit\b|\binformation technology\b|\bcomputer\b|\bsoftware\b|\bweb\b|\btechnical\b|\btechnology\b|\bprogramm"
                   r"|\bwebmaster\b|\bsecurity (?:analyst|engineer|specialist|administrator|architect|consultant)\b|\bnetwork"
                   r"|\bpc (?:technician|support)\b|\bdatabase\b",
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
                    r"|b\.?\s?s\.?c?|b\.?\s?a|m\.?\s?s\.?c?|m\.?\s?a|a\.?\s?a\.?s?|major|minor|concentration"
                    r"|technology|engineering)\b",  # alone they name no field ("Bachelor of Technology in Technology")
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
    # Changed roles within tech: someone already in tech who moves into a different role bucket after at least this
    # many months in the first one (for example IT support -> data science). Their own pool; False turns it off.
    "within_tech_switches": True,
    "min_months_in_first_bucket": 12,
}

_MONTH = re.compile(r"^\s*(\d{1,2})\s*[/-]\s*(\d{4})\s*/?\s*$")             # 06/2015
_MONTH_YY = re.compile(r"^\s*(\d{1,2})\s*[/-]\s*(\d{2})\s*/?\s*$")         # 06/15 (month / 2-digit year)
_YEAR_MONTH = re.compile(r"^\s*(\d{4})\s*[/-]\s*(\d{1,2})\s*/?\s*$")       # 2015/06, 2015/06/
_FULL_DATE = re.compile(r"^\s*(\d{1,2})\s*/\s*\d{1,2}\s*/\s*(\d{2}|\d{4})\s*$")  # 06/30/2015, 06/30/15 (month first)
_YEAR = re.compile(r"^\s*(\d{4})\s*/?\s*$")                                   # 2015, 2015/
_NAMED = re.compile(r"^\s*([A-Za-z]{3,9})\.?\s*,?\s*(\d{4})\s*$")             # June 2015, Sept. 2015
_NOW = re.compile(r"^\s*(?:present|current|now|today|to date|till date|ongoing)\s*$", re.IGNORECASE)
_MONTHS = {name: i for i, names in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
     ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
     ("dec", "december")], start=1) for name in names}


def _year(two_or_four, as_of):
    """A 4-digit year as is; a 2-digit year in this century up to as_of's year, otherwise the 1900s."""
    y = int(two_or_four)
    if len(two_or_four) == 4:
        return y
    return 2000 + y if y <= as_of.year % 100 else 1900 + y


def parse_month(value, as_of):
    """A start or end date to a monthly Period; 'Present' and similar to as_of; anything unreadable to NaT.
    Reads MM/YYYY, MM/YY, YYYY/MM, MM/DD/YYYY, YYYY, and month names ("June 2015"). An impossible month after a 4-digit
    year keeps the year; elsewhere it stays unreadable (a first number above 12 can't be a month)."""
    if not isinstance(value, str):
        return pd.NaT
    if _NOW.match(value):
        return as_of
    for pattern, year_group, month_group in ((_MONTH, 2, 1), (_MONTH_YY, 2, 1), (_YEAR_MONTH, 1, 2), (_FULL_DATE, 2, 1)):
        m = pattern.match(value)
        if m:
            month = int(m.group(month_group))
            if not 1 <= month <= 12:
                if pattern is _YEAR_MONTH:  # "2015/00", "2015/13": keep the year, as for a bare year
                    return pd.Period(year=int(m.group(1)), month=1, freq="M")
                return pd.NaT
            return pd.Period(year=_year(m.group(year_group), as_of), month=month, freq="M")
    m = _YEAR.match(value)
    if m:
        return pd.Period(year=int(m.group(1)), month=1, freq="M")
    m = _NAMED.match(value)
    if m and m.group(1).lower() in _MONTHS:
        return pd.Period(year=int(m.group(2)), month=_MONTHS[m.group(1).lower()], freq="M")
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
    jobs["manager_plus"] = low.str.contains(rules["manager_target"], regex=True)
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
    """Labels for one person. Everyone who reached a computer and mathematical job gets every label; `status` is the
    main one, so tables can count people by it. Career changers have a 'candidate: retrained ...' status."""
    out = {"status": None, "origin": None, "origin_type": None, "first_level": None, "from_level": None, "roles_reached": None, "from_bucket": None,
           "reached_data": False, "switch_month": pd.NaT, "target_title": None, "target_bucket": None,
           "source_title": None, "last_source_title": None, "source_study": None, "retrained": False,
           "retrain_study": None, "study_undated": False, "study_level_before": np.nan, "background_months": 0, "gap_months": np.nan,
           "n_jobs": int(g.start.notna().sum()), "n_jobs_before": 0}
    g = g.dropna(subset=["start"]).sort_values("start")
    if g.empty:
        return {**out, "status": "no dated jobs"}
    cm = g[g.is_cm & ~g.not_background]
    if cm.empty:
        return {**out, "status": "never in a computer and mathematical role"}
    first = cm.iloc[0]
    reached = sorted(cm.cm_bucket.unique())
    out.update(switch_month=first.start, target_title=first.title, target_bucket=first.cm_bucket,
               roles_reached=", ".join(reached), reached_data="data science" in reached,
               first_level="manager or above" if first.manager_plus else "senior" if first.senior else "other")

    tl = person_timeline(g, edu, first.start, rules)
    before = tl[tl.key < first.start.ordinal]
    out["n_jobs_before"] = int((before.kind == "job").sum())
    out["study_undated"] = bool(((before.kind == "study") & ~before.dated).any())
    studies = before[before.kind == "study"]
    out["study_level_before"] = max((0 if lab == "skip" else _study_level(txt, rules) for txt, lab in zip(studies.text, studies.label)),
                                    default=np.nan)  # 0 high school, 1 associate, 2 bachelor's, 3 master's or certificate, 4 doctorate
    tech_keys = before.key[before.label == "tech"]
    first_tech = tech_keys.min() if len(tech_keys) else float(first.start.ordinal)
    origin = before[(before.key < first_tech) & (before.label != "skip")]
    retrain = before[(before.key >= first_tech) & (before.kind == "study") & (before.label == "tech")]
    other = origin[origin.label == "other"]

    if other.empty:
        if (origin.label == "unclear").any():
            out["origin"], status = "unclear field of study", "original field of study unclear"
        elif len(tech_keys):
            out["origin"], status = "tech or math", "original background in tech or math"
        elif (before.label == "skip").any():
            out["origin"], status = "school or internships only", "only school, internships or high school before the switch"
        else:
            out["origin"], status = "none", "no background before the switch"
        return _within_tech({**out, "status": status}, cm, rules)  # retrained stays False: only another field can retrain

    out["retrained"] = not retrain.empty
    out["retrain_study"] = retrain.text.iloc[-1] if out["retrained"] else None
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
        out["origin"] = "short other-field job"
        return _within_tech({**out, "status": f"background under {rules['min_background_months']} months"}, cm, rules)
    out["origin_type"] = "job" if job_background else "study"
    out["origin"] = "other-field job" if job_background else "other-field study"
    if out["retrained"]:
        status = ("candidate: retrained, after working in another field" if job_background
                  else "candidate: retrained, new grad from another field")
    elif rules["keep_direct_switches"]:
        status = "candidate: changed jobs" if job_background else "candidate: new grad from another field"
    else:
        status = ("changed jobs without tech or math training" if job_background
                  else "new grad from another field without tech or math training")
    return {**out, "status": status}


def _within_tech(out, cm, rules):
    """If someone already in tech later moved into a different role bucket, after enough months in the first one,
    relabel them as 'candidate: changed roles within tech', with the new bucket as their switch."""
    if not rules["within_tech_switches"]:
        return out
    first_bucket = cm.cm_bucket.iloc[0]
    moved = cm[cm.cm_bucket != first_bucket]
    if moved.empty:
        return out
    new = moved.iloc[0]
    earlier = cm[(cm.start < new.start) & (cm.cm_bucket == first_bucket)]
    ends = earlier.end.where(earlier.end.notna() & (earlier.end <= new.start), new.start)
    months = int(sum(_months(s, e) for s, e in zip(earlier.start, ends)))
    if months < rules["min_months_in_first_bucket"]:
        return out
    return {**out, "status": "candidate: changed roles within tech", "from_bucket": first_bucket,
            "switch_month": new.start, "target_title": new.title, "target_bucket": new.cm_bucket,
            "first_level": "manager or above" if new.manager_plus else "senior" if new.senior else "other",
            # highest level held in the first role, so moves can be split into sideways and up (descriptive only)
            "from_level": "manager or above" if earlier.manager_plus.any() else "senior" if earlier.senior.any() else "other",
            "background_months": months, "source_title": earlier.title.iloc[int(np.argmax([_months(s, e) for s, e in zip(earlier.start, ends)]))],
            "last_source_title": earlier.title.iloc[-1]}


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
    return out


# Role buckets for listings: O*NET-SOC codes of computer and mathematical occupations mapped to the same buckets as
# RULES["cm_buckets"] (the resume side), so listings and career switches line up role by role. Matched on the first 7
# characters ("15-1252.00" -> "15-1252"). Both the 2019 SOC codes and the older 2010 ones are listed, since the
# version in the listings isn't confirmed. Codes not listed come back as None; EDA_SDA_2 reports them.
ONET_ROLE_BUCKETS = {
    # 2019 SOC
    "15-1211": "networks and systems",       # computer systems analysts
    "15-1212": "security",                   # information security analysts
    "15-1221": "data science",         # computer and information research scientists
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
    "15-2031": "data science",         # operations research analysts
    "15-2041": "data science",         # statisticians
    "15-2051": "data science",         # data scientists
    "15-2099": "data science",         # mathematical science occupations, all other
    # 2010 SOC
    "15-1111": "data science",         # computer and information research scientists
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


# Detailed O*NET codes inside "computer occupations, all other" (15-1299), matched on the full code first, so they
# don't all land in "other computer roles".
ONET_DETAILED_BUCKETS = {
    "15-1299.01": "networks and systems",    # web administrators
    "15-1299.04": "security",                # penetration testers
    "15-1299.05": "security",                # information security engineers
    "15-1299.06": "security",                # digital forensics analysts
    "15-1299.07": "software development",    # blockchain engineers
    "15-1299.08": "networks and systems",    # computer systems engineers/architects
    "15-1299.09": "it project management",   # information technology project managers
}


# The scope is the roles TripleTen trains for, plus adjacent tech roles (decided 2026-10-10): a neutral, outside definition,
# not any team member's background. Two TripleTen roles sit outside the computer and mathematical group; each is its own
# bucket, listings side only (the career-switch profiles, frozen, have no paths into them).
EXTRA_ONET_BUCKETS = {
    "41-9031": "technical sales",              # sales engineers (Forward Deployed Engineering, AI Automation)
    "13-1111": "business and data analysis",   # management analysts, often titled business analyst (Data Analytics)
    "13-1161": "business and data analysis",   # market research analysts and marketing specialists (Data Analytics)
}
EXTRA_ORDER = ["technical sales", "business and data analysis"]

# O*NET 13-1161 mixes market research analysts with marketing specialists (decided 2026-10-11). Titles that name analysis
# or research stay in business and data analysis; the rest are labelled marketing specialists and hidden in the app
# (kept in the data), like document management: marketing work, not what the Data Analytics program trains for.
MARKETING_BUCKET = "marketing specialists"
MARKET_ANALYST_TITLE = (r"analy|research|insight|\bdata\b|intelligence|statistic|measurement|quantitative|reporting"
                        r"|\bbi\b|metrics|scientist")


def marketing_specialist(onet_codes, titles):
    """True for 13-1161 listings whose title names no analysis or research (marketing specialists, not analysts)."""
    code = pd.Series(onet_codes).fillna("").astype(str).str.strip().str[:7]
    title = pd.Series(titles, index=code.index).fillna("").astype(str).str.lower()
    return code.eq("13-1161") & ~title.str.contains(MARKET_ANALYST_TITLE, regex=True)

# TripleTen program -> the O*NET codes its graduates are trained for (tripleten.com, checked 2026-10-10). Detailed codes
# (10 characters) win over 7-character codes. Scope listings with no program are "adjacent tech roles" (IT support,
# networks and systems, database, IT project management, other computer roles): reachable, but no TripleTen program.
TRIPLETEN_PROGRAMS = {
    "AI Software Engineering": ["15-1251", "15-1252", "15-1254", "15-1131", "15-1132", "15-1133", "15-1134", "15-1299.07"],
    "UX/UI Design": ["15-1255"],
    "AI & Machine Learning": ["15-1221", "15-1111", "15-2051.00"],
    "Data Analytics": ["15-2051.01", "15-2051.02", "15-2031", "15-2041", "15-2099", "13-1111", "13-1161"],
    "Quality Assurance": ["15-1253"],
    "Cybersecurity": ["15-1212", "15-1122", "15-1299.04", "15-1299.05", "15-1299.06"],
    "Forward Deployed Engineering / AI Automation": ["41-9031"],
}
PROGRAM_ORDER = list(TRIPLETEN_PROGRAMS) + ["adjacent tech role"]
_PROGRAM_OF = {code: program for program, codes in TRIPLETEN_PROGRAMS.items() for code in codes}


def tripleten_program(onet_code):
    """The TripleTen program that trains for a listing's O*NET code; None if no program does (an adjacent role)."""
    if not isinstance(onet_code, str):
        return None
    code = onet_code.strip()
    return _PROGRAM_OF.get(code[:10]) or _PROGRAM_OF.get(code[:7])


def role_bucket(onet_code):
    """The role bucket of a listing's O*NET-SOC code: a computer and mathematical bucket, or one of the TripleTen roles
    outside that group (EXTRA_ONET_BUCKETS); None if the code isn't mapped."""
    if not isinstance(onet_code, str):
        return None
    code = onet_code.strip()
    return ONET_DETAILED_BUCKETS.get(code[:10]) or ONET_ROLE_BUCKETS.get(code[:7]) or EXTRA_ONET_BUCKETS.get(code[:7])


# Broad areas for describing where career changers come from (analysis only: changing these never changes who is a
# candidate). Applied to the main background job title, or the field studied. First match wins.
BACKGROUND_AREAS = {
    "self-employed and small business": r"\bowner\b|self[- ]employed|freelanc|entrepreneur|\bfounder\b",
    "legal": r"\blaw\b|\blegal\b|attorney|lawyer|paralegal|\bllb\b|\bjuris",
    "customer service and support": r"customer (?:service|support|care|success)|call cent|client (?:service|support)|help line|\bcsr\b|customer\b",
    "education": r"teach|tutor|instructor|professor|educat|school|lecturer|curriculum|trainer|training|coach|librar",
    "healthcare": r"nurs|medical|health|clinic|pharma|patient|dental|therap|caregiver|physician|hospital|\bemt\b|paramedic|\bcna\b|\baide\b|kinesiolog|nutrition|dietit|fitness|optic",
    "food and hospitality": r"server|barista|cook|chef|restaurant|hospitality|hotel|bartend|food|waiter|waitress|kitchen|dishwash|\bhost(?:ess)?\b|crew member|catering",
    "sales and retail": r"sales|retail|cashier|store|merchandis|account executive|shop|representative|\brep\b|real estate|realtor",
    "finance and accounting": r"account|financ|bank|audit|tax|bookkeep|econom|insurance|actuar|loan|credit|commerce|\bb\.?\s?com\b|\bca\b|chartered",
    "administration and office": r"admin|\boffice\b|clerk|reception|secretar|assistant|data entry|coordinator|dispatch|schedul|records|front desk|typist|\bmail",
    "agriculture and environment": r"agricultur|farm|forestry|horticult|veterinar|fisher",
    "business, marketing and management": r"business|marketing|manag|operations|mba|project|consult|human resources|\bhr\b|recruit|analyst|supervisor|director|executive|president|\bceo\b|planner|purchas|procurement|buyer|strateg|\bevent",
    "science and research": r"research|scientist|lab|physics|chemi|biolog|geolog|environment",
    "engineering (not software)": r"engineer|mechanical|civil|industrial|architect|drafting",
    "humanities and social sciences": r"english|history|philosoph|psycholog|sociolog|politic|literature|language|anthropolog|social|humanities|liberal arts|geograph|general studies|liberal studies|interdisciplinary|counsel|church|ministr|pastor|volunteer|non-?profit",
    "arts, media and communications": r"design|\bart|music|writ|journalis|media|communicat|editor|photograph|film|graphic|video|producer|animat|\bactor|theat|danc|reporter|broadcast|content",
    "military and public service": r"army|navy|military|air force|marine|police|government|public|officer|security guard|firefight|soldier|sergeant|corps|veteran|criminal justice|correction",
    "logistics and trades": r"warehouse|driver|logistic|supply|mechanic|electrician|construction|technician|maintenance|install|inventory|stock|shipping|receiving|deliver|courier|forklift|carpenter|plumb|weld|janitor|custodian|cleaner|landscap|painter|hvac|labor|contractor",
    "manufacturing and production": r"manufactur|production|assembl|operator|machin|factory|plant|quality control",
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


# One order for training fields and tech roles, from infrastructure to code to data, so related fields and roles sit
# near each other and the people in path tables form a band down the diagonal.
ROLE_ORDER = ["it support", "networks and systems", "security", "database", "software development", "quality assurance",
              "data science", "it project management"]
FIELD_ORDER = ["networking", "cybersecurity", "information technology and systems", "computer engineering and electronics",
               "computer applications (BCA / MCA)", "computer science", "software and web development",
               "data science and analytics", "mathematics and statistics", "tech degree, field not named"]


def path_table(df, row, col, row_order, row_label, col_label, total_label):
    """Where people went: % of everyone in df (one decimal, 0 where nobody did) by row × col, both in the given
    order; every role is a column, a `people` column gives each row's size and a total row each column's share."""
    counts = pd.crosstab(df[row], df[col])
    rows = [r for r in row_order if r in counts.index] + [r for r in counts.index if r not in row_order]
    cols = ROLE_ORDER + [c for c in counts.columns if c not in ROLE_ORDER]
    counts = counts.reindex(index=rows, columns=cols, fill_value=0)
    n = int(counts.values.sum())
    table = (100 * counts / n).round(1)
    table.loc[total_label] = (100 * counts.sum() / n).round(1)
    table["people"] = counts.sum(axis=1).tolist() + [n]
    return table.rename_axis(index=row_label, columns=col_label)


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
