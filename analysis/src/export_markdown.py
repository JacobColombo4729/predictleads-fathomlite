"""Export a notebook to markdown, without its code.

Usage, from the project folder:
    python -m src.export_markdown notebooks/EDA_SDA_1.ipynb [more notebooks ...]
        -> docs/<name>.md: the notebook's text only. Safe to commit.
    python -m src.export_markdown --report notebooks/EDA_SDA_1.ipynb
        -> reports/<name>.md: a 5-minute version: the summary, each step's findings, and the chosen tables (small
           ones side by side) and charts, as set in LAYOUTS below.
    python -m src.export_markdown --outputs notebooks/EDA_SDA_1.ipynb [...]
        -> reports/<name>_full.md: all the text plus every table and chart, in notebook order.
    Both need the notebook run from the top and saved first. reports/ is git-ignored: outputs are never committed
    (CLAUDE.md), and sharing them follows the advisor's rules on presenting results.

Cells tagged `data-rows` (row previews) and error messages are always left out. Regenerate after editing the
notebook; don't edit the .md by hand.
"""
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE = re.compile(r"<style.*?</style>", re.DOTALL)


def _markdown(cell):
    """A markdown cell's text, minus notes and headings that only make sense next to the code."""
    text = "".join(cell["source"]).strip()
    lines = text.splitlines()
    if not text or text.startswith("*No output"):
        return None  # notes about code cells mean nothing without the code
    while len(lines) > 1 and lines[-1].startswith("#"):
        lines = lines[:-1]  # a trailing sub-heading whose section is only code ("### 7.0 Helpers")
    if len(lines) == 1 and lines[0].startswith("#") and not lines[0].startswith("# "):
        return None  # a bare sub-heading whose section is only code
    return "\n".join(lines).strip()


def _outputs(cell, fig_dir, fig_count):
    """A code cell's tables (HTML), printed titles and charts (saved as PNG); never rows or errors."""
    if "data-rows" in cell.get("metadata", {}).get("tags", []):
        return [], fig_count
    parts = []
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream" and out.get("name") == "stdout":
            text = "".join(out["text"]).strip()
            if text:
                parts.append("\n\n".join(f"**{line.strip()}**" if line.strip() else "" for line in text.splitlines()).strip())
        elif kind in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "image/png" in data:
                fig_count += 1
                fig_dir.mkdir(parents=True, exist_ok=True)
                name = f"fig_{fig_count:02d}.png"
                (fig_dir / name).write_bytes(base64.b64decode("".join(data["image/png"])))
                parts.append(f"![Figure {fig_count}]({fig_dir.name}/{name})")
            elif "text/html" in data:
                parts.append(STYLE.sub("", "".join(data["text/html"])).strip())
            elif "text/plain" in data and not "".join(data["text/plain"]).startswith("<Figure"):
                parts.append("```\n" + "".join(data["text/plain"]).strip() + "\n```")
    return parts, fig_count


# 5-minute report layouts, per notebook. Sections are matched on the first word of their heading ("6.2", "Q1b.", "10a");
# tables on the start of their printed title; charts on the start of their code cell's first line. Tables listed in
# one group are shown side by side once every member has appeared.
LAYOUTS = {
    "EDA_SDA_1": {
        "skip_sections": {"Work", "Notebook", "1.", "2.", "4a", "4b", "7."},  # "4." stays: 4c–4e need their parent heading
        "full_text_sections": {"Job", "8."},  # the purpose, and the conclusion with its summary (as in EDA_SDA_2)
        "max_bullets": 2,  # per "What the data shows" block; the full version keeps them all
        "bullets_by_section": {"4c": 1, "4e": 1, "6.": 3},
        "table_groups": [
            ["3a. Where computer and mathematical stands"],
            ["4c. How English US listings enter the scope", "4c. From the full file to the scope", "4c. The scope by role bucket"],
            ["4d. Scope-only fields"],
            ["4e. The scope table", "4e. Share of scope listings with each key field"],
            ["5d. Source type", "5c. Median days open"],
            ["5f. Recruiters in the scope", "5f. Window by recruiter"],
            ["6a. History and weekly arrivals"],
        ],
        "charts": ["# 3c chart", "# 5a chart", "# 5b chart", "# 5c chart", "# 5h chart", "# 6a chart"],
    },
    "EDA_SDA_1_Lean": {  # EDA_SDA_1's layout, with the Lean notebook's headings and cell labels
        "skip_sections": {"About", "1.", "2.", "PredictLeads", "4a", "4b", "7.", "Where"},  # "4." stays: 4c-4e need their parent heading
        "full_text_sections": {"EDA/SDA", "8."},  # the title and purpose, and the conclusion with its summary
        "full_text_headings": {"### Why we need the Gehan-Wilcoxon test"},  # this one cell, in full (not its whole section)
        "max_bullets": 2,
        "bullets_by_section": {"4c": 1, "4e": 1, "6.": 3},
        "table_groups": [
            ["3a. Where computer and mathematical stands"],
            ["4c. How English US listings enter the scope", "4c. From the full file to the scope", "4c. The scope by role bucket"],
            ["4d. Scope-only fields"],
            ["4e. The in-scope table", "4e. Share of in-scope listings with each key field"],
            ["5d. Source type", "5c. Median days open"],
            ["5c. Gehan-Wilcoxon test on every pair"],
            ["5f. Recruiters in the scope", "5f. Window by recruiter"],
            ["6a. History and weekly arrivals"],
        ],
        "charts": ["# 3c chart", "# 5a-3 Chart", "# 5b-2 Chart", "# 5c-3 Chart", "# 5h-2 Chart", "# 6a-2 Chart"],
    },
    "EDA_SDA_2_Lean": {  # written from its question: the rules, the evidence for each, the buckets, the 2026 market
        "skip_sections": {"About", "1."},
        "full_text_sections": {"EDA/SDA", "8."},  # the title and purpose, and the conclusion with its answers
        "max_bullets": 4,
        "table_groups": [
            ["2 The rules at a glance"],
            ["3a Closing by listing age"],
            ["3b Signals among"],
            ["4.1 Level of each listing"],
            ["4.3 Median days open (Kaplan-Meier): staffing agencies vs. employers", "4.3 Median days open (Kaplan-Meier) by recruiter"],
            ["4.4 Senior title"],
            ["5.2 Role buckets in the scope"],
            ["5.3 Role buckets: own analysis"],
            ["6.1 The 2026 market by role bucket", "6.1 The 2026 market by TripleTen program"],
            ["7.1 Listings each rule touches"],
        ],
        "charts": ["# 3a"],
    },
    "EDA_SDA_2": {
        "skip_sections": {"2.", "2b."},
        "hide_headings": {"4a–4e"},  # its tables (the funnel and the summary) still show
        "full_text_sections": {"5."},  # the conclusion opens with the summary and answers the clarifying questions
        "max_bullets": 1,
        "bullets_by_section": {"3.1": 2, "3.9": 2, "3.11": 2, "3.13": 3},
        "table_groups": [
            ["1d. The"],  # prefixes match the titles before and after the niche → scope rename
            ["3.1e The whole scope vs. the career-changer view"],
            ["3.3c Median window (Kaplan–Meier) by company volume"],
            ["3.6b Role buckets in the"],
            ["3.7c Within mid_senior"],
            ["3.11a What each rule hides"],
            ["3.12a The 2026 market by role bucket"],
            ["3.13e Verdicts by family"],
            ["4b Final table: listings after each step", "4c Final"],
        ],
        "charts": ["# 3.1 Chart", "# 3.2 Chart", "# 3.3e Chart", "# 3.6 Chart", "# 3.9 Chart", "# 3.11 Chart",
                   "# 3.12 Chart", "# 3.13 Chart"],
    },
    "EDA_SDA_3": {
        "skip_sections": {"Plan:", "1.", "7."},
        "full_text_sections": {"8."},  # the conclusion with its summary; the opener duplicates PROJECT.md's notebook table
        "max_bullets": 3,
        "bullets_by_section": {"3.": 4, "4.": 4, "6.": 7},
        "table_groups": [
            ["2b Words removed as boilerplate", "2b Share of each description kept"],
            ["3a EDA_SDA_1's years rule vs. the new one", "3a The 5+ years rule"],
            ["3a Years required by role bucket"],
            ["3b Degree rules", "3b Highest degree level"],
            ["3b Degree required (no equivalent) by role bucket"],
            ["4a Share of listings with a description where each feature fires"],
            ["4b Text features vs. closing within 7 days"],
            ["4c Evergreen wording by ghost status"],
            ["5b Validation results"],
            ["6c Coverage"],
            ["6d Median share", "6e Job Zone"],
            ["6f Transferable skills both rate"],
        ],
        "charts": ["# 2 Chart", "# 3 Chart", "# 6c Chart"],
    },
    "EDA_SDA_3_Lean": {  # EDA_SDA_3's layout, with the Lean notebook's table titles
        "skip_sections": {"Plan:", "1.", "7."},
        "full_text_sections": {"8."},
        "max_bullets": 3,
        "bullets_by_section": {"3.": 4, "4.": 4, "6.": 7},
        "table_groups": [
            ["2b Words removed as boilerplate", "2b Share of each description kept"],
            ["3a EDA_SDA_1_Lean's years rule vs. the new one", "3a The 5+ years rule"],
            ["3a Years required by role bucket"],
            ["3b Degree rules", "3b Highest degree level"],
            ["3b Degree required (no equivalent) by role bucket"],
            ["4a Share of listings with a description where each feature fires"],
            ["4b Text features vs. closing within 7 days"],
            ["4c Evergreen wording by ghost status"],
            ["5b Validation results"],
            ["6c Coverage"],
            ["6d Median share", "6e Job Zone"],
            ["6f Transferable skills both rate"],
        ],
        "charts": ["# 2 Chart", "# 3 Chart", "# 6c Chart"],
    },
    "EDA_SDA_4": {
        "skip_sections": {"1.", "Roadmap"},
        "full_text_sections": {"12."},  # the conclusion with its answers, limits and handoff; the opener duplicates PROJECT.md's notebook table
        "max_bullets": 3,
        "bullets_by_section": {"10.": 3, "8d.": 2},
        "table_groups": [
            ["3b Point-biserial"],
            ["4b Flags whose gap reverses"],
            ["4c Close rate by role bucket"],
            ["6a 7-day close rate"],
            ["8b Survival by age"],
            ["8d Joint Wald"],  # the per-covariate proportional-hazards dump (72 rows) stays in _full.md and the notebook
            ["9a Stability over months"],
            ["9b Within-company gap"],
            ["10a Feature decisions"],
            ["11a The modeling-ready table"],
        ],
        "charts": ["# 2b", "# 3a"],
    },
    "EDA_SDA_4_Lean": {  # EDA_SDA_4's layout, with the Lean notebook's table titles
        "skip_sections": {"1.", "Roadmap"},
        "full_text_sections": {"12."},  # the conclusion with its answers, limits and handoff; the opener duplicates PROJECT.md's notebook table
        "max_bullets": 3,
        "bullets_by_section": {"10.": 3, "8d.": 2},
        "table_groups": [
            ["3b Point-biserial"],
            ["4b Flags whose gap reverses"],
            ["4c Close rate by role bucket"],
            ["6a 7-day close rate"],
            ["8b Survival by age"],
            ["8d Joint Wald"],  # the per-covariate proportional-hazards dump (72 rows) stays in _full.md and the notebook
            ["9a Stability over months"],
            ["9b Within-company gap"],
            ["10a Decisions (kept / dropped)"],
            ["11a The modeling-ready table"],
        ],
        "charts": ["# 2b", "# 3a"],
    },
    "TimeSeries": {
        "skip_sections": {"1.", "Roadmap"},
        "full_text_sections": {"Job", "0.", "Clarifying", "What", "Answers", "Limitations", "Handoff"},  # the purpose, and the conclusion
        "max_bullets": 4,
        "bullets_by_section": {"4.": 5, "5.": 4},
        "table_groups": [
            ["2a Days and weeks", "2d Share of the week by weekday"],
            ["2e Mix by month first seen"],
            ["3b Strength of the STL components", "3d Stationarity tests"],
            ["4a Features and split"],
            ["4e Validation scores", "4f Rolling-origin cross-validation"],
            ["4h Test scores"],
            ["5a Closing speed by month first seen"],
            ["5b Share of new listings you see still open, checking every N days: all originals",
             "5b Share of new listings you see still open, checking every N days: what the app shows"],
            ["5c Longest gap between checks", "5c Check cadence by role bucket"],
            ["6b Listing momentum by role bucket"],
        ],
        "charts": ["# 2b", "# 2d", "# 2e", "# 3b", "# 4e", "# 4h", "# 5a", "# 5b", "# 6a"],
    },
    "TimeSeries_Lean": {  # TimeSeries' layout, with the Lean notebook's title and Purpose heading
        "skip_sections": {"1.", "Roadmap", "About"},
        "full_text_sections": {"TimeSeries", "Purpose", "Clarifying", "What", "Answers", "Limitations", "Handoff"},  # the purpose, and the conclusion
        "max_bullets": 4,
        "bullets_by_section": {"4.": 5, "5.": 4},
        "table_groups": [
            ["2a Days and weeks", "2d Share of the week by weekday"],
            ["2e Mix by month first seen"],
            ["3b Strength of the STL components", "3d Stationarity tests"],
            ["4a Features and split"],
            ["4e Validation scores", "4f Rolling-origin cross-validation"],
            ["4h Test scores"],
            ["5a Closing speed by month first seen"],
            ["5b Share of new listings you see still open, checking every N days: all originals",
             "5b Share of new listings you see still open, checking every N days: what the app shows"],
            ["5c Longest gap between checks", "5c Check cadence by role bucket"],
            ["6b Listing momentum by role bucket"],
        ],
        "charts": ["# 2b", "# 2d", "# 2e", "# 3b", "# 4e", "# 4h", "# 5a", "# 5b", "# 6a"],
    },
    "Profiles": {
        "skip_sections": {"1.", "3.", "4b", "5."},
        "full_text_sections": {"Job", "6."},
        "max_bullets": 2,
        "bullets_by_section": {"4a.3": 3},
        "table_groups": [
            ["Duplicate resumes", "From every resume to viable candidates"],
            ["Data science: everyone vs. the career-changer view", "Level of the job at the switch"],
            ["Summary statistics, all viable candidates"],
            ["Retrained career changers by path"],
            ["Prior career → first tech job"],
            ["Retraining compass"],
            ["Changed roles within tech"],
            ["Moves within tech: sideways or up, by", "Moves within tech: sideways or up, overall"],
        ],
        "charts": ["# 4a.1a", "# 4a.1b", "# 4a.2a", "# 4a.2b", "# 4a.2c"],
    },
    "Profiles_Lean": {  # Profiles' layout, with the Lean notebook's title
        "skip_sections": {"1.", "3.", "4b", "5."},
        "full_text_sections": {"Profiles", "6."},  # the title and purpose, and the conclusion
        "max_bullets": 2,
        "bullets_by_section": {"4a.3": 3},
        "table_groups": [
            ["Duplicate resumes", "From every resume to viable candidates"],
            ["Data science: everyone vs. the career-changer view", "Level of the job at the switch"],
            ["Summary statistics, all viable candidates"],
            ["Retrained career changers by path"],
            ["Prior career → first tech job"],
            ["Retraining compass"],
            ["Changed roles within tech"],
            ["Moves within tech: sideways or up, by", "Moves within tech: sideways or up, overall"],
        ],
        "charts": ["# 4a.1a", "# 4a.1b", "# 4a.2a", "# 4a.2b", "# 4a.2c"],
    },
    "ML_Fit": {
        "skip_sections": {"1.", "Roadmap", "Leakage", "Data"},
        "full_text_sections": {"Where"},  # where the notebook stands; the opener duplicates PROJECT.md's notebook table
        "max_bullets": 3,
        "bullets_by_section": {"6.": 4, "5.": 4, "7.": 3, "8.": 3},
        "table_groups": [
            ["Career changers by pool"],
            ["5a People with a background profile", "5a Each role's most important skills"],
            ["5b Trained layer candidates"],
            ["Development: hit rate over 5 folds", "Development: top-3 hit rate per pool"],
            ["Test: hit rate with 95% intervals", "Test: top-3 hit rate per pool"],
            ["Current-role pull"],
            ["7a Average size of each group", "7a For each test person"],
            ["7a The 12 inputs"],
            ["8a Paths by kind"],
            ["8b Each role's most asked skills"],
            ["8b Of each role's 10 most asked skills"],
            ["8c Pay change moving into each role"],
        ],
        "charts": [],
    },
    "ML_Fit_Lean": {  # ML_Fit's layout, with the Lean notebook's "About this Lean draft" cell skipped
        "skip_sections": {"About", "1.", "Roadmap", "Leakage", "Data"},
        "full_text_sections": {"Where"},  # where the notebook stands; the opener duplicates PROJECT.md's notebook table
        "max_bullets": 3,
        "bullets_by_section": {"6.": 4, "5.": 4, "7.": 3, "8.": 3},
        "table_groups": [
            ["Career changers by pool"],
            ["5a People with a background profile", "5a Each role's most important skills"],
            ["5b Trained layer candidates"],
            ["Development: hit rate over 5 folds", "Development: top-3 hit rate per pool"],
            ["Test: hit rate with 95% intervals", "Test: top-3 hit rate per pool"],
            ["Current-role pull"],
            ["7a Average size of each group", "7a For each test person"],
            ["7a The 12 inputs"],
            ["8a Paths by kind"],
            ["8b Each role's most asked skills"],
            ["8b Of each role's 10 most asked skills"],
            ["8c Pay change moving into each role"],
        ],
        "charts": [],
    },
    "ML_Rank": {
        # 9, 10 and 12 are "_To do._" stubs (scope frozen, ANALYSIS.md): skipped rather than shown empty.
        "skip_sections": {"1.", "9.", "10.", "12."},
        "full_text_sections": {"13."},  # the hand-off to the app; the opener duplicates PROJECT.md's notebook table
        "max_bullets": 2,
        "bullets_by_section": {"7c": 4, "7": 3, "8": 3, "8d": 4, "8e": 4, "11a": 3},
        "table_groups": [
            ["6a Imbalance handling: AUC and F1"],
            ["7c Cox on the training months", "7c Cox on the training months"],  # the same cell's two displays
            ["7d AFT models"],
            ["8e Test month, once: H4, H5, H6"],
            ["11a Does the result hold at 7, 14 and 30 days"],
        ],
        "charts": [],
    },
    "ML_Rank_Lean": {  # ML_Rank's layout, with the Lean notebook's "About this Lean draft" cell skipped
        "skip_sections": {"About", "1.", "9.", "10.", "12."},
        "full_text_sections": {"13."},
        "max_bullets": 2,
        "bullets_by_section": {"7c": 4, "7": 3, "8": 3, "8d": 4, "8e": 4, "11a": 3},
        "table_groups": [
            ["6a Imbalance handling: AUC and F1"],
            ["7c Cox on the training months", "7c Cox on the training months"],
            ["7d AFT models"],
            ["8e Test month, once: H4, H5, H6"],
            ["11a Does the result hold at 7, 14 and 30 days"],
        ],
        "charts": [],
    },
    "Evaluate": {
        "skip_sections": {"1."},
        "full_text_sections": {"Job", "5."},  # the opener's ranker legend (short, not duplicated elsewhere) and the conclusion
        "max_bullets": 6,  # one "What the data shows" block carries the whole comparison
        "table_groups": [
            ["Career changers by pool"],
            ["Seconds per person"],
            ["4b Right role in the top k", "4b Top 3, by pool"],
            ["4c Difference in top-3 hit rate"],
            ["4d Top 3 from each ranker's top 10"],
        ],
        "charts": ["# 4e"],
    },
    "Evaluate_Lean": {  # Evaluate's layout, with the Lean notebook's "About this Lean draft" cell skipped
        "skip_sections": {"About", "1."},
        "full_text_sections": {"Evaluate", "5."},  # the title and purpose (ranker legend), and the conclusion
        "max_bullets": 6,
        "table_groups": [
            ["Career changers by pool"],
            ["Seconds per person"],
            ["4b Right role in the top k", "4b Top 3, by pool"],
            ["4c Difference in top-3 hit rate"],
            ["4d Top 3 from each ranker's top 10"],
        ],
        "charts": ["# 4e"],
    },
}


def _section(text):
    """The first word of a cell's first heading ("6.1", "Q1b.", "Executive"), or None."""
    for line in text.splitlines():
        if line.startswith("#"):
            words = line.lstrip("#").split()
            return words[0] if words else None
    return None


def _skipped(section, skip):
    """A skipped section also skips its sub-sections ("4." covers "4.2")."""
    return section in skip or any(p.endswith(".") and section and section.startswith(p) for p in skip)


def _first_bullets(text, n):
    """A findings block cut to its heading line and first n top-level bullets (with their sub-bullets)."""
    out, count = [], 0
    for line in text.splitlines():
        if line.startswith("- "):
            count += 1
        if count > n:
            break
        out.append(line)
    return "\n".join(out).strip()


def _side_by_side(tables):
    """One table with its title, or several in one row (HTML, so it renders in Markdown previews)."""
    tables = [(title, "\n".join(l for l in html.splitlines() if l.strip())) for title, html in tables]  # a blank line ends an HTML block
    if len(tables) == 1:
        title, html = tables[0]
        return f"**{title}**\n\n{html}"
    cells = "".join(f'<td valign="top"><b>{title}</b><br>{html}</td>' for title, html in tables)
    return f"<table><tr>{cells}</tr></table>"


def _anchor(heading):
    """A heading's link target, the way Markdown viewers build it: lowercase, punctuation dropped, spaces hyphenated."""
    text = heading.lstrip("#").strip()
    kept = "".join(c for c in text.lower() if c.isalnum() or c in " -_")
    return "-".join(kept.split())


def _toc(parts, min_headings=4):
    """A contents block linking to every ## and ### heading in the assembled report ("" when there are too few)."""
    seen, rows = {}, []
    for part in parts:
        for line in part.splitlines():
            line = line.strip()
            if not (line.startswith("## ") or line.startswith("### ")):
                continue
            anchor = _anchor(line)
            seen[anchor] = seen.get(anchor, 0) + 1
            if seen[anchor] > 1:
                anchor = f"{anchor}-{seen[anchor] - 1}"  # repeated headings get -1, -2 ...
            depth = "  " if line.startswith("### ") else ""
            rows.append(f"{depth}- [{line.lstrip('#').strip()}](#{anchor})")
    if len(rows) < min_headings:
        return ""
    return "## Contents\n\n" + "\n".join(rows)


def _with_toc(parts):
    """The report's parts with a contents block after the title (or first, when there is no title)."""
    toc = _toc(parts)
    if not toc:
        return parts
    after = next((i for i, p in enumerate(parts) if p.lstrip().startswith("# ")), -1)
    return parts[:after + 1] + [toc] + parts[after + 1:]


def report(nb_path):
    """The 5-minute version: summary, findings and chosen tables and charts. Returns (path, figures, tables, prose words)."""
    nb_path = Path(nb_path).resolve()
    layout = LAYOUTS[nb_path.stem]
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    fig_dir = out_dir / f"{nb_path.stem}_figures"
    groups = layout["table_groups"]
    found = {i: [] for i in range(len(groups))}
    parts, figs, n_tables, words, section = [], 0, 0, 0, None

    def flush():
        nonlocal n_tables
        for i in sorted(found):
            if found[i]:
                parts.append(_side_by_side(found[i]))
                n_tables += len(found[i])
                found[i] = []

    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            text = "".join(cell["source"]).strip()
            new = _section(text)
            if new:
                section = new  # groups wait for all their tables, even across headings (unfinished ones go at the end)
            if _skipped(section, layout["skip_sections"]):
                continue
            if section in layout["full_text_sections"] or any(text.startswith(h) for h in layout.get("full_text_headings", ())):
                kept = _markdown(cell)  # whole sections, or single cells that open with one of these headings
            elif text.startswith("**What"):
                kept = _first_bullets(text, layout.get("bullets_by_section", {}).get(section, layout["max_bullets"]))
            elif section in layout.get("hide_headings", set()):
                kept = None
            else:
                kept = next((l for l in text.splitlines() if l.startswith("#")), None)
            if kept:
                parts.append(kept)
                words += len(kept.split())
            continue
        if _skipped(section, layout["skip_sections"]) or "data-rows" in cell.get("metadata", {}).get("tags", []):
            continue
        label = "".join(cell["source"]).splitlines()[0] if cell["source"] else ""
        cell_title = label.lstrip("#").strip()  # a table with no printed title falls back to the cell's own heading comment
        title = None
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                lines = [l.strip() for l in "".join(out["text"]).splitlines() if l.strip()]
                title = lines[-1] if lines else title
            elif "image/png" in data and any(label.startswith(c) for c in layout["charts"]):
                figs += 1
                fig_dir.mkdir(parents=True, exist_ok=True)
                name = f"fig_{figs:02d}.png"
                (fig_dir / name).write_bytes(base64.b64decode("".join(data["image/png"])))
                parts.append(f"![Figure {figs}]({fig_dir.name}/{name})")
            elif "text/html" in data and (title or cell_title):
                use = title or cell_title
                for i, group in enumerate(groups):
                    if any(use.startswith(g) for g in group):
                        found[i].append((use, STYLE.sub("", "".join(data["text/html"])).strip()))
                        if len(found[i]) == len(group):
                            parts.append(_side_by_side(found[i]))
                            n_tables += len(found[i])
                            found[i] = []
                        break
                title = None
    flush()
    rel = nb_path.relative_to(ROOT).as_posix()
    header = (f"<!-- Generated from {rel} by src/export_markdown.py (5-minute report; no code). "
              "Regenerate after editing the notebook; don't edit by hand. Full version: "
              f"{nb_path.stem}_full.md -->")
    out = out_dir / f"{nb_path.stem}.md"
    out.write_text(header + "\n\n" + "\n\n".join(_with_toc(parts)) + "\n", encoding="utf-8")
    return out, figs, n_tables, words


def export(nb_path, outputs=False):
    nb_path = Path(nb_path).resolve()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out_dir = ROOT / ("reports" if outputs else "docs")
    out_dir.mkdir(exist_ok=True)
    fig_dir = out_dir / f"{nb_path.stem}_full_figures"
    parts, figs = [], 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            text = _markdown(cell)
            if text:
                parts.append(text)
        elif outputs and cell["cell_type"] == "code":
            new, figs = _outputs(cell, fig_dir, figs)
            parts.extend(new)
    rel = nb_path.relative_to(ROOT).as_posix()
    what = "text, tables and charts" if outputs else "text only"
    header = (f"<!-- Generated from {rel} by src/export_markdown.py ({what}; no code). "
              "Regenerate after editing the notebook; don't edit by hand. -->")
    out = out_dir / (f"{nb_path.stem}_full.md" if outputs else f"{nb_path.stem}.md")
    out.write_text(header + "\n\n" + "\n\n".join(_with_toc(parts)) + "\n", encoding="utf-8")
    return out, figs


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    for arg in (a for a in args if not a.startswith("--")):
        if "--report" in flags:
            out, figs, tables, words = report(arg)
            print(f"wrote {out.relative_to(ROOT)}: {words:,} words of text, {tables} tables, {figs} charts")
        else:
            out, figs = export(arg, outputs="--outputs" in flags)
            print(f"wrote {out.relative_to(ROOT)}" + (f" and {figs} figures" if "--outputs" in flags else ""))
