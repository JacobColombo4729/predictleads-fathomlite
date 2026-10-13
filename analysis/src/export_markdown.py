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
        "skip_sections": {"Work", "Notebook", "1.", "2.", "3.", "4.", "5.", "6.0", "6.2", "6.3", "6.4", "6.5", "6.6",
                          "6.7", "7.0", "7.2", "8.", "8a", "10."},
        "full_text_sections": {"Job", "13."},  # the purpose, and the conclusion with its summary (as in EDA_SDA_2)
        "max_bullets": 2,  # per "What the data shows" block; the full version keeps them all
        "bullets_by_section": {"6.1": 1, "11.": 1, "12.": 1},
        "hide_headings": {"11."},  # its tables still show, in the row under 12
        "table_groups": [
            ["Text cleaning results", "Years of experience required", "Posting source"],
            ["Q3a. Where computer and mathematical stands"],
            ["8c. How English US listings enter the scope", "8c. From the full file to the scope", "8c. The scope by role bucket"],
            ["8d. Scope-only fields"],
            ["Q1c. Source type", "Q1d. Do high-volume"],
            ["Q1f. Text features vs. application window", "Q1g. Career-change flags vs. application window"],
            ["Q1h. Recruiters in the scope", "Q1h. Window by recruiter"],
            ["Q2a. How much history", "Q2d. Weekly arrivals"],
            ["11d. Share of scope listings with each key field", "The scope table at a glance", "Scope table: role buckets"],
        ],
        "charts": ["# 6.1i", "# Q3a heat map", "# Q3a chart", "# Q3a charts", "# Q1a chart", "# Q1b chart", "# Q1c chart", "# Q1d charts",
                   "# Q1e charts", "# Q1f chart", "# Q1g chart", "# Q2 charts: weekday", "# 9.3a Chart"],
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
        "full_text_sections": {"Job", "8."},  # the purpose, and the conclusion with its summary
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
            if section in layout["full_text_sections"]:
                kept = _markdown(cell)
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
            elif "text/html" in data and title:
                for i, group in enumerate(groups):
                    if any(title.startswith(g) for g in group):
                        found[i].append((title, STYLE.sub("", "".join(data["text/html"])).strip()))
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
    out.write_text(header + "\n\n" + "\n\n".join(parts) + "\n", encoding="utf-8")
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
    out.write_text(header + "\n\n" + "\n\n".join(parts) + "\n", encoding="utf-8")
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
