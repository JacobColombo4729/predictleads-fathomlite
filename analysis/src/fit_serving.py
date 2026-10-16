"""A new person's résumé to ML_Fit's trained-layer inputs (the combined model, step 2), with Profiles' own rules.

The app's profile entries ({kind: job | project | education, title, organization, description}; the local résumé
reader writes each entry's dates, as written, at the start of its description, "Jan 2020 - Present. ...") become the
two tables Profiles reads (jobs: title, start_date, end_date; education: program, start_date), and then go through
career_switch unchanged: tag_jobs, person_timeline and _one_person, the rules that built the modeling table.

A Profiles person is described at their switch, the first tech job. An app user hasn't switched yet, so:
  - career change (no tech job on the résumé): the switch is now. A placeholder tech job starting this month stands
    in for it, and _one_person reads everything before it, exactly as for a Profiles career changer;
  - within tech (a tech job on the résumé): origin, training and the study columns come from _one_person at their
    first tech job, as for Profiles' within-tech movers; the move starts from their current tech role (the latest
    one), with its level and the years spent in it.
The pre-switch text is Profiles 5c's: job titles and study programs before the switch, internships and high school
left out, joined with " | ". Projects aren't jobs or programs in Profiles, so they're left out too.

    person_features(entries, as_of=None) -> one-row DataFrame: the Profiles feature columns, `text`, and the notes
                                            below (`case`, `undated_entries`); RoleScorer.features() takes it as is.
Nothing here prints entries; the notes are counts.
"""
import re

import numpy as np
import pandas as pd

from src import career_switch as cs
from src.fit_model import CATEGORICAL, FLAGS, NUMERIC

# Dates as the local résumé reader writes them at the start of a description (app/code/local_resume_reader.py)
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
          r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_POINT = rf"(?:{_MONTH}\s*,?\s*(?:19|20)\d{{2}}|\d{{1,2}}\s*/\s*(?:19|20)?\d{{2}}|(?:19|20)\d{{2}})"
_END = rf"(?:{_POINT}|present|current|now|today|ongoing)"
_RANGE = re.compile(rf"^\s*({_POINT})\s*(?:-|–|—|to|until|through)\s*({_END})(?![\w/])", re.IGNORECASE)
_SINGLE = re.compile(rf"^\s*({_POINT})(?![\w/])", re.IGNORECASE)
PLACEHOLDER_SWITCH = "data analyst"  # a title every rule reads as a tech job, never manager or senior
USER = "user"


def _norm_date(text):
    """A date as written to a form parse_month reads ("Sept. 2015" -> "Sept 2015", "06 / 2015" -> "06/2015")."""
    return re.sub(r"\s*/\s*", "/", re.sub(r"\.", "", text.strip()))


def entry_dates(description):
    """(start, end) strings from the dates at the start of a description; (start, None) for a single date;
    (None, None) if it doesn't start with a date."""
    text = str(description or "")
    m = _RANGE.match(text)
    if m:
        return _norm_date(m.group(1)), _norm_date(m.group(2))
    m = _SINGLE.match(text)
    return (_norm_date(m.group(1)), None) if m else (None, None)


def profile_tables(entries):
    """The app's entries as Profiles' two tables: (jobs: person_id, title, start_date, end_date;
    education: person_id, program, start_date). Projects are left out, as Profiles has none."""
    jobs, education = [], []
    for e in entries:
        kind, title = e.get("kind", "job"), str(e.get("title") or "").strip()
        if not title or kind == "project":
            continue
        start, end = entry_dates(e.get("description"))
        if kind == "education":
            education.append({"person_id": USER, "program": title, "start_date": start})
        else:
            jobs.append({"person_id": USER, "title": title, "start_date": start, "end_date": end})
    return (pd.DataFrame(jobs, columns=["person_id", "title", "start_date", "end_date"]),
            pd.DataFrame(education, columns=["person_id", "program", "start_date"]))


def _level(rows):
    return "manager or above" if rows.manager_plus.any() else "senior" if rows.senior.any() else "other"


def person_features(entries, as_of=None):
    """One row: Profiles' feature columns for this person, their pre-switch `text`, and two notes (`case`: career
    change or within tech; `undated_entries`: jobs and programs with no readable date, which the timeline can't place)."""
    as_of = pd.Period(pd.Timestamp.today(), "M") if as_of is None else pd.Period(as_of, "M")
    rules = {**cs.RULES, "as_of": as_of}  # "Present" means this month
    jobs_raw, edu_raw = profile_tables(entries)
    jobs = cs.tag_jobs(jobs_raw, rules)
    edu = cs.prepare_education(edu_raw, rules).get(USER) if len(edu_raw) else None
    undated = int(jobs.start.isna().sum()) + (int(edu.start.isna().sum()) if edu is not None else 0)
    tech = jobs[jobs.is_cm & ~jobs.not_background & jobs.start.notna()]

    if tech.empty:  # career change: the switch is now
        switch = cs.tag_jobs(pd.DataFrame([{"person_id": USER, "title": PLACEHOLDER_SWITCH,
                                            "start_date": str(as_of), "end_date": None}]), rules)
        switch["start"] = as_of
        timeline_jobs = pd.concat([jobs, switch], ignore_index=True) if len(jobs) else switch
        out = cs._one_person(timeline_jobs, edu, rules)
        case = "career change"
    else:  # within tech: Profiles' rules at the first tech job, the move from the current tech role
        timeline_jobs = jobs
        out = cs._one_person(jobs, edu, rules)
        current = tech.sort_values("start").cm_bucket.iloc[-1]
        held = tech[tech.cm_bucket.eq(current)]
        months = sum(cs._months(s, min(e, as_of) if pd.notna(e) else as_of) for s, e in zip(held.start, held.end))
        out = {**out, "from_bucket": current, "from_level": _level(held), "background_months": int(months),
               "origin_type": None}
        case = "within tech"

    if out.get("from_bucket"):
        area = "tech: " + out["from_bucket"]
    else:
        area = cs.background_area(out.get("source_title") if out.get("origin_type") == "job" else out.get("source_study"))
    tl = cs.person_timeline(timeline_jobs, edu, as_of, rules)
    before = tl[(tl.key < as_of.ordinal) & tl.label.ne("skip")]
    row = {"origin": out.get("origin"), "from_level": out.get("from_level"), "background_area": area,
           "retrain_field": cs.training_field(out.get("retrain_study")), "from_bucket": out.get("from_bucket"),
           "retrained": bool(out.get("retrained")), "study_undated": bool(out.get("study_undated")),
           "years_before": round(out.get("background_months", 0) / 12, 1), "n_jobs_before": int(out.get("n_jobs_before", 0)),
           "study_level_before": out.get("study_level_before", np.nan),
           "text": " | ".join(before.text.astype(str)), "case": case, "undated_entries": undated}
    return pd.DataFrame([row], index=pd.Index([USER], name="person_id"))[CATEGORICAL + FLAGS + NUMERIC +
                                                                         ["text", "case", "undated_entries"]]
