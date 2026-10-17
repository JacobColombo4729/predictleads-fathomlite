"""EDA_SDA_1's feature dictionary: every column of the scope table (analysis_table.parquet), its role, the
prioritization factor it serves, what it means, and its coverage.

    feature_dictionary(final, path) -> (dictionary DataFrame, undocumented columns)
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FEATURES = {
    # column: (role, prioritization factor, description)
    "id": ("identifier", "", "PredictLeads listing id; joins to scope_descriptions.parquet (full descriptions)"),
    "company_id": ("identifier", "", "PredictLeads company id; links to the companies table"),
    "title": ("text", "employer intent", "Listing title as posted"),
    "row": ("identifier", "", "Row position in the stream's rows table (data/flat/all_fields_rows_full/), for joins back to it"),
    "title_norm": ("helper", "", "Lowercased title with whitespace collapsed; used to match reposts"),
    "job_title": ("text", "transferable fit", "PredictLeads' normalized title; where missing, the posted title lowercased and trimmed"),
    "job_title_fallback": ("helper", "", "job_title is the posted title (PredictLeads gave no normalized title)"),
    "has_title": ("helper", "", "The listing has a title"),
    "repost_fp": ("helper", "", "64-bit fingerprint of company, normalized title and description fingerprint; matches reposts (2c)"),
    "contract_bits": ("helper", "", "contract_types as bit flags (labels unified), for counts in memory"),
    "category_bits": ("helper", "", "categories as bit flags, for counts in memory"),
    "n_tags": ("category", "transferable fit", "Number of tags"),
    "n_categories": ("category", "transferable fit", "Number of categories"),
    "has_recruiter": ("metadata", "", "A recruiter is named on the listing (a flag only: names and contacts are personal data and aren't kept)"),
    "last_processed": ("time", "", "When PredictLeads last processed the listing; likely moves with last_seen, so audited for leakage before any use"),
    "salary_currency": ("factor", "pay", "Currency the pay was posted in"),
    "salary_low": ("factor", "pay", "Low end of the pay range as posted, in salary_currency"),
    "salary_high": ("factor", "pay", "High end of the pay range as posted, in salary_currency"),
    "was_translated": ("metadata", "", "PredictLeads translated the title"),
    "first_seen": ("time", "freshness", "When the crawler first found the listing (UTC)"),
    "last_seen": ("time", "still open / apply-by", "When the crawler last saw the listing (UTC)"),
    "posted": ("time", "freshness", "Employer's posting date, where given (about 41%); midnight Central European time"),
    "month": ("time", "", "Month the listing was first seen; the monthly time horizon"),
    "role_bucket": ("category", "role bucket", "Role bucket from the O*NET code (src/career_switch.py): the eight computer and mathematical buckets, other computer roles, the two added TripleTen buckets (technical sales, business and data analysis), or an unmapped 15- code"),
    "company_earlier": ("factor", "competition", "The company's listings in any field first seen before this one (2e; same-day listings count together; reposts included)"),
    "full_week": ("helper", "", "First seen in a full week (not the partial first or last week); weekly rates use these only"),
    "recruiter_key": ("factor", "recruiters", "Anonymous key per recruiter (fingerprint of name and company with a per-run salt that's never saved); names and contacts aren't kept"),
    "recruiter_title": ("category", "recruiters", "The recruiter's job title, e.g. technical recruiter"),
    "salary_text": ("text", "pay", "Pay as written in the listing"),
    "pay_from_text": ("helper", "pay", "salary_hourly_usd was parsed from salary_text, since salary_data had no numbers (4d)"),
    "location_text": ("text", "", "Location as written in the listing (e.g. a city or 'Remote - US'); for the app's location filter"),
    "url": ("text", "source", "Link to the listing (the app links to it)"),
    "record_type": ("metadata", "", "PredictLeads record type; the same for every listing (kept for completeness)"),
    "company_record_type": ("metadata", "", "Type of the linked company record; the same for every listing (kept for completeness)"),
    "tripleten_program": ("category", "role bucket", "The TripleTen program that trains for the listing's O*NET code (src/career_switch.py), or 'adjacent tech role' where none does (4c)"),
    "batch_first_seen": ("helper", "freshness", "First seen in a one-day batch holding 60%+ of the company's 5+ listings (2e): the company was added to the crawl, not a hiring surge"),
    "crawl_dropout": ("helper", "still open / apply-by", "Disappeared on a day holding 60%+ of the company's 5+ disappearances (2e): the crawler likely lost the site, so days_open may end early"),
    "partial_month": ("scope", "", "First seen on 2026-03-30 or 31, the data's first two days: kept, but left out of per-month counts"),
    "days_open": ("target", "still open / apply-by", "Days from first seen to last seen"),
    "live": ("target", "still open / apply-by", "Last seen within 2 days of September 30, the data's end date: still open then, so we don't know when it closes (censored)"),
    "status": ("metadata", "", "PredictLeads status; only 'closed' or missing, so not used as a live flag"),
    "seniority": ("category", "career-change accessibility", "Seniority level, e.g. junior, mid_senior, manager"),
    "occupation_group": ("category", "occupation group", "O*NET occupation group (PredictLeads field onet_data.family); computer and mathematical, or missing where a 15- code placed the listing"),
    "occupation": ("category", "transferable fit", "O*NET occupation name; finer than the occupation group"),
    "onet_code": ("category", "occupation group", "O*NET-SOC code; maps to a role bucket in the computer and mathematical scope (EDA_SDA_2 3.6)"),
    "categories": ("category", "transferable fit", "PredictLeads category labels (27 values)"),
    "tags": ("category", "transferable fit", "PredictLeads tags; sparse (about half empty)"),
    "contract_types": ("category", "transferable fit", "Contract types, e.g. full time, part time, remote; labels unified in 2b"),
    "language": ("scope", "", "Listing language; the scope table is English only"),
    "country": ("scope", "", "Country of the first listed location; the final table is United States only"),
    "n_locations": ("category", "", "Number of locations the listing names"),
    "source_type": ("factor", "source", "Hiring platform, job board, or company site / other (from the url domain)"),
    "platform": ("factor", "source", "Named hiring platform or job board; empty for company sites"),
    "salary_low_usd": ("factor", "pay", "Low end of the salary range, USD, in its original time unit"),
    "salary_high_usd": ("factor", "pay", "High end of the salary range, USD, in its original time unit"),
    "salary_unit": ("factor", "pay", "Salary time unit: hour, day, week, month or year"),
    "salary_hourly_usd": ("factor", "pay", "Midpoint pay as an hourly rate (salaried pay ÷ 2,080 hours, full time); blank if not disclosed or implausible"),
    "pay_suspect": ("helper", "pay", "Pay was fixed in 2d: reversed range swapped, or a clearly wrong unit re-read"),
    "has_desc": ("text", "employer intent", "Has a description"),
    "has_html": ("helper", "", "Description contained HTML before cleaning"),
    "desc_words": ("text", "employer intent", "Words in the cleaned description"),
    "desc_hash": ("helper", "", "Fingerprint of the cleaned description; used to match reposts"),
    "mentions_remote": ("text", "employer intent", "Title or description mentions remote work; blank if there's no description and the title doesn't say remote"),
    "urgency": ("text", "employer intent", "Keyword flag for urgency wording (urgent, immediately, ASAP); blank without a description. Saved for EDA_SDA_3, which tests urgency with NLP; not analyzed here"),
    "years_required": ("text", "career-change accessibility", "Most years of experience the description asks for (lower end of each range, up to 20); blank if none stated"),
    "entry_wording": ("text", "career-change accessibility", "Says entry level, no experience required, will train, paid training or apprenticeship; blank without a description"),
    "changer_wording": ("text", "career-change accessibility", "Mentions career changers, transferable skills or non-traditional backgrounds; blank without a description"),
    "degree_required": ("text", "career-change accessibility", "Says a bachelor's, master's or associate degree is required; blank without a description"),
    "equivalent_experience": ("text", "career-change accessibility", "Accepts equivalent experience in place of a degree; blank without a description"),
    "entry_seniority": ("factor", "career-change accessibility", "Seniority label is entry level (junior)"),
    "low_experience_ask": ("factor", "career-change accessibility", "Asks for 2 years of experience or fewer"),
    "career_open": ("factor", "career-change accessibility", "Any sign of being open to career changers: entry-level seniority, 2 years or fewer, entry-level or training wording, or career-changer wording. Blank when nothing says yes and the text is missing. First pass; vetted in EDA_SDA_4"),
    "is_repost": ("factor", "reposts", "This job was listed before"),
    "times_listed": ("factor", "reposts", "How many times this job has been listed so far (1 = first listing)"),
    "days_since_first_listed": ("factor", "reposts", "Days since the job's original listing"),
    "days_since_repost_last_listed": ("factor", "reposts", "Days since the job's previous listing; 0 for the first listing"),
}
ROLE_ORDER = ["identifier", "time", "target", "factor", "category", "text", "scope", "metadata", "helper"]


def feature_dictionary(final, path):
    """Every column saved at `path`: role, factor, description and coverage (% of listings with a value; from `final`
    where loaded, else read from the file one column at a time). Also returns the saved columns with no description."""
    def coverage(col):
        if col in final.columns:
            return round(100 * final[col].notna().mean(), 1)
        c = pq.read_table(path, columns=[col]).column(0)
        return round(100 * (1 - c.null_count / max(len(c), 1)), 1)

    saved = pq.read_schema(path).names
    out = pd.DataFrame(
        [(col, *FEATURES.get(col, ("?", "", "no description yet")), coverage(col)) for col in saved],
        columns=["column", "role", "prioritization factor", "description", "coverage %"],
    ).set_index("column")
    out["low coverage"] = np.where(out["coverage %"] < 20, "under 20%", "")
    out = out.sort_values("role", key=lambda s: s.map({r: i for i, r in enumerate(ROLE_ORDER)}).fillna(99), kind="stable")
    return out, [c for c in saved if c not in FEATURES]
