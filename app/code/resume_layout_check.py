"""Why did the local resume reader find nothing? Prints the resume's layout only, never its text.

    python app/code/resume_layout_check.py "C:\\path\\to\\resume.pdf"

Counts lines, words per line, bullets and date ranges, lists which standard section headings it recognised (generic
words such as "Experience"), and how many entries and skills the reader got from each section. No names, employers,
titles or other resume text are printed, so the output is safe to share.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_resume_reader as reader  # noqa: E402
import resume_profile_extraction as extraction  # noqa: E402


def layout(path: Path) -> None:
    data = path.read_bytes()
    raw_lines = None
    if data.startswith(b"%PDF"):
        from io import BytesIO

        from pypdf import PdfReader

        pages = PdfReader(BytesIO(data)).pages
        raw = "\n".join((p.extract_text() or "") for p in pages)
        raw_lines = [l for l in raw.splitlines() if l.strip()]
        print(f"PDF pages: {len(pages)}; text lines as the PDF gives them: {len(raw_lines)}; "
              f"median words per line: {statistics.median(len(l.split()) for l in raw_lines) if raw_lines else 0}")
        text = reader.pdf_to_text(data)  # what Magic fill reads in Jacob's app (the page's lines kept)
    else:
        text = data.decode("utf-8", errors="replace")
    text = extraction.redact_contact_details(text)
    lines = [l for l in text.split("\n") if l.strip()]
    print(f"lines the reader sees: {len(lines)}; median words per line: {statistics.median(len(l.split()) for l in lines) if lines else 0}")
    print(f"bullet lines: {sum(bool(reader.BULLET.match(l)) for l in lines)}; lines with a date range: "
          f"{sum(bool(reader.DATE_RANGE.search(l)) for l in lines)}; date ranges in the whole text: {len(reader.DATE_RANGE.findall(text))}; "
          f"single years/months: {len(reader.SINGLE_DATE.findall(text))}")
    header, sections = reader._sections(lines)
    found = [reader._heading_key(l.strip().rstrip(':')) for l in lines if reader._section_of(l)]
    print(f"standard headings recognised: {found or 'none'}")
    print(f"lines above the first heading: {len(header)}")
    for kind, body in sections:
        n_entries = len(reader._entries(kind, body)) if kind in ("job", "project", "education") else "-"
        n_skills = len(reader._skill_items(body)) if kind == "skills" else "-"
        print(f"  section '{kind}': {len(body)} lines, entries read: {n_entries}, skills read: {n_skills}")
    near = sorted({reader._heading_key(l) for l in lines if 1 <= len(l.split()) <= 4 and l.strip().isupper()} - set(found))
    print(f"other short ALL-CAPS lines (possible headings it did not recognise): {len(near)}")
    out = reader.read_resume(text)
    kinds = [e["kind"] for e in out["experience"]]
    print(f"read: name {'yes' if out['name'] else 'no'}, headline {'yes' if out['headline'] else 'no'}, "
          f"location {'yes' if out['location'] else 'no'}; entries: {kinds.count('job')} jobs, {kinds.count('project')} projects, "
          f"{kinds.count('education')} education; with an organization: {sum(bool(e['organization']) for e in out['experience'])}; "
          f"skills: {len(out['skills'])}")
    kept = extraction.extract_profile(text, reader.LocalResumeProvider())
    print(f"after the app's grounding check: {len(kept['profile']['experience'])} entries, {len(kept['profile']['skills'])} skills, "
          f"{kept['dropped']} dropped")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    layout(Path(sys.argv[1]))
