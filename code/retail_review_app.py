"""Local Streamlit screen for human review of retail/inventory recommendations."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

STORE_SPEC = importlib.util.spec_from_file_location(
    "retail_review_store", Path(__file__).with_name("retail_review_store.py")
)
if STORE_SPEC is None or STORE_SPEC.loader is None:
    raise ImportError("Could not load retail_review_store.py")
review_store = importlib.util.module_from_spec(STORE_SPEC)
sys.modules[STORE_SPEC.name] = review_store
STORE_SPEC.loader.exec_module(review_store)

PROFILE_ID = review_store.PROFILE_ID
load_progress = review_store.load_progress
load_retail_jobs = review_store.load_retail_jobs
load_saved_judgments = review_store.load_saved_judgments
rating_from_status = review_store.rating_from_status
review_counts = review_store.review_counts
save_progress = review_store.save_progress
write_saved_judgment = review_store.write_saved_judgment

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "data" / "matching" / "description_coverage_20261007T141953Z"
BLINDED_PATH = EXPERIMENT_DIR / "blinded_review_view.csv"
JUDGMENTS_PATH = EXPERIMENT_DIR / "retail_review_judgments.csv"
PROGRESS_PATH = EXPERIMENT_DIR / "retail_review_progress.json"
SHORT_DESCRIPTION_CHARS = 250
RATING_OPTIONS = ["0: Unsuitable", "1: Plausible", "2: Strong fit", "Insufficient evidence"]
RATING_VALUES = {
    "0: Unsuitable": "0",
    "1: Plausible": "1",
    "2: Strong fit": "2",
}


st.set_page_config(page_title="Retail / Inventory Job Review", layout="wide")
st.title("Retail / Inventory Recommendation Review")
st.caption("Human review of the saved description-coverage experiment. This screen does not run matching or change the source recommendations.")
st.warning("These are historical job listings, not verified current openings. Similarity scores are intentionally hidden during review.")

if not BLINDED_PATH.is_file():
    st.error(f"Review input is missing: {BLINDED_PATH}")
    st.info("Obtain the approved experiment folder and place `blinded_review_view.csv` at the path shown above.")
    st.stop()

try:
    jobs = load_retail_jobs(BLINDED_PATH)
    judgments = load_saved_judgments(JUDGMENTS_PATH)
except (OSError, ValueError) as error:
    st.error(f"Could not load review data: {error}")
    st.stop()

if not jobs:
    st.error(f"No rows found for {PROFILE_ID} in {BLINDED_PATH}")
    st.stop()

job_ids = [row["job_id"] for row in jobs]
if "review_index" not in st.session_state:
    st.session_state.review_index = load_progress(PROGRESS_PATH, job_ids)
st.session_state.review_index = min(max(int(st.session_state.review_index), 0), len(jobs) - 1)
index = st.session_state.review_index
job = jobs[index]
job_id = job["job_id"]
key = (PROFILE_ID, job_id)
saved = judgments.get(key)
counts = review_counts(job_ids, judgments)

st.caption(f"Profile: `{PROFILE_ID}` · Job {index + 1} of {len(jobs)} · Stable job ID: `{job_id}`")
progress_columns = st.columns(3)
progress_columns[0].metric("Rated", counts["rated"])
progress_columns[1].metric("Insufficient evidence", counts["insufficient_evidence"])
progress_columns[2].metric("Unreviewed", counts["unreviewed"])

candidate_raw = job.get("candidate_evidence") or "{}"
try:
    candidate = json.loads(candidate_raw)
except json.JSONDecodeError:
    candidate = {"raw": candidate_raw}
with st.expander("Candidate background and preferences", expanded=True):
    if isinstance(candidate, dict):
        left, right = st.columns(2)
        left.subheader("Experience")
        for item in candidate.get("experience", []):
            left.markdown(f"- {item}")
        left.subheader("Evidenced skills")
        for item in candidate.get("evidenced_skills", []):
            left.markdown(f"- {item}")
        right.subheader("Desired roles")
        for item in candidate.get("desired_roles", []):
            right.markdown(f"- {item}")
        right.write(f"**Location preference:** {candidate.get('location') or 'Not specified'}")
        preferences = candidate.get("preferences") or {}
        right.write("**Other preferences:**")
        st.json(preferences, expanded=False)
    else:
        st.text(str(candidate))

left, right = st.columns([2, 1])
with left:
    st.subheader("Job details")
    st.text(f"Title: {job.get('source_title') or job.get('searchable_title') or 'Untitled job'}")
    st.text(f"Company: {job.get('company_name') or 'Not provided'}")
    st.text(f"Location: {job.get('location_raw') or 'Unknown / not provided'}")
    st.text(f"Status: {job.get('status_derived') or 'unknown'} (historical source status: {job.get('status_raw') or 'unknown'})")
    source_url = job.get("source_url") or ""
    parsed_url = urlparse(source_url)
    if parsed_url.scheme in {"http", "https"} and parsed_url.netloc:
        st.link_button("Open source listing", source_url)
    elif source_url:
        st.text(f"Source URL: {source_url}")
    available = str(job.get("description_available", "")).strip().casefold() == "true"
    try:
        description_length = int(job.get("description_length") or 0)
    except ValueError:
        description_length = 0
    st.write(f"**Description:** {'available' if available else 'missing'} · {description_length:,} prepared characters")
    if not available or not (job.get("description_full_prepared") or "").strip():
        st.error("Description missing: relevance evidence is limited to the title and available metadata.")
    elif description_length <= SHORT_DESCRIPTION_CHARS:
        st.warning(f"Very short description ({description_length} characters); assess cautiously.")
    st.subheader("Prepared job description")
    # Text area renders the value as plain text, never as source HTML/Markdown.
    st.text_area("Full prepared description (plain text)", value=job.get("description_full_prepared") or "",
                 height=420, disabled=True, label_visibility="collapsed", key=f"description_{job_id}")

with right:
    st.subheader("Human relevance review")
    st.caption("Rate the evidence for this candidate profile and job, not whether the source seniority label guarantees eligibility.")
    saved_rating = rating_from_status(saved)
    default_option = next((label for label, value in RATING_VALUES.items() if value == saved_rating), None)
    if saved and saved.get("review_status") == "insufficient_evidence":
        default_option = "Insufficient evidence"
    rating_key = f"rating_{job_id}"
    if rating_key not in st.session_state:
        st.session_state[rating_key] = default_option
    selected_rating = st.radio(
        "Choose one; leave unselected to keep this job unreviewed",
        options=RATING_OPTIONS,
        index=RATING_OPTIONS.index(default_option) if default_option in RATING_OPTIONS else None,
        key=rating_key,
    )
    reason_key = f"reason_{job_id}"
    notes_key = f"notes_{job_id}"
    reviewer_key = f"reviewer_{job_id}"
    if reason_key not in st.session_state:
        st.session_state[reason_key] = (saved or {}).get("insufficient_evidence_reason", "")
    if notes_key not in st.session_state:
        st.session_state[notes_key] = (saved or {}).get("notes", "")
    if reviewer_key not in st.session_state:
        st.session_state[reviewer_key] = (saved or {}).get("reviewer", "")

    reason = st.text_area("Reason when evidence is insufficient", key=reason_key, height=90)
    notes = st.text_area("Optional review notes", key=notes_key, height=120)
    reviewer = st.text_input("Reviewer (optional)", key=reviewer_key)
    st.caption("0 = unsuitable · 1 = plausible · 2 = strong fit · insufficient evidence leaves the score blank")
    if st.button("Save this review", type="primary", use_container_width=True):
        if selected_rating == "Insufficient evidence" and not reason.strip():
            st.error("Please record a brief reason for insufficient evidence.")
        else:
            if selected_rating == "Insufficient evidence":
                status = "insufficient_evidence"
                score = None
                saved_reason = reason.strip()
            elif selected_rating in RATING_VALUES:
                status = "rated"
                score = RATING_VALUES[selected_rating]
                saved_reason = ""
            else:
                status = "unreviewed"
                score = None
                saved_reason = ""
            write_saved_judgment(
                JUDGMENTS_PATH, PROFILE_ID, job_id,
                relevance_score=score,
                review_status=status,
                insufficient_evidence_reason=saved_reason,
                reviewer=reviewer,
                notes=notes,
            )
            save_progress(PROGRESS_PATH, index, job_id)
            st.success(f"Saved {status.replace('_', ' ')} review.")
            st.rerun()

nav_left, nav_right = st.columns(2)
with nav_left:
    if st.button("Previous", disabled=index == 0, use_container_width=True):
        st.session_state.review_index = max(index - 1, 0)
        save_progress(PROGRESS_PATH, st.session_state.review_index, job_ids[st.session_state.review_index])
        st.rerun()
with nav_right:
    if st.button("Next", disabled=index >= len(jobs) - 1, use_container_width=True):
        st.session_state.review_index = min(index + 1, len(jobs) - 1)
        save_progress(PROGRESS_PATH, st.session_state.review_index, job_ids[st.session_state.review_index])
        st.rerun()

st.caption(f"Judgments are saved separately to `{JUDGMENTS_PATH}`. The source blinded CSV is read-only; progress resumes at the last visited stable job ID.")
