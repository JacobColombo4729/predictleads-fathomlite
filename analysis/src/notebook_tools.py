"""Checks that keep a notebook's text in step with its results (used by Thread, and by hand).

Usage, from analysis/ (.venv activated):
    python -m src.notebook_tools stale notebooks/EDA_SDA_2.ipynb
        -> numbers quoted in the markdown that no longer appear in any output (years, section numbers and small
           counts skipped), with the heading they sit under. Prints only the quoted numbers, which are already in
           the committed text.
    python -m src.notebook_tools outputs notebooks/EDA_SDA_2.ipynb 52 64 [max chars per cell]
        -> markdown cells and the compact text of code outputs for a cell range: aggregate tables only. Images and
           cells tagged data-rows (row previews) are skipped.
    python -m src.notebook_tools status notebooks/EDA_SDA_2.ipynb
        -> how many code cells ran, carry outputs or errors, and whether its reports are newer than the notebook.
The notebook must be run and saved first. Nothing here reads data files.
"""
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(\s?%|\s?pp)?(?![\w])")


def load(path):
    return json.load(open(path, encoding="utf-8"))


def out_text(o):
    """The text of one output: streams and HTML tables flattened (cells ' | ', rows ' // '); images skipped."""
    if o.get("output_type") == "stream":
        return "".join(o.get("text", ""))
    d = o.get("data", {})
    if "text/html" in d:
        h = re.sub(r"<style.*?</style>", "", "".join(d["text/html"]), flags=re.S)
        h = re.sub(r"</t[dh]>", " | ", h)
        h = re.sub(r"</tr>", " // ", h)
        return html.unescape(re.sub(r"<[^>]+>", "", h))
    if "text/plain" in d and "image/png" not in d:
        return "".join(d["text/plain"])
    return ""


def is_row_preview(cell):
    return "data-rows" in cell.get("metadata", {}).get("tags", [])


def _norm(n):
    return n.replace(",", "")


def _variants(value):
    """Ways a number can be printed: 13887, 13,887, 13887.0, and longer decimals."""
    out = {value}
    try:
        f = float(value)
    except ValueError:
        return out
    if f.is_integer():
        i = int(f)
        out |= {str(i), f"{i:,}", f"{f:.1f}", f"{f:.2f}"}
    else:
        for d in range(len(value.split(".")[1]), 4):
            out.add(f"{f:.{d}f}")
    return out


def _worth_checking(num, unit):
    v = float(_norm(num))
    if unit:
        return True
    if 1990 <= v <= 2035 and "." not in num:
        return False  # a year
    return v >= 100 or "." in num or "," in num


def stale(path):
    """(heading, quoted number) pairs from the markdown that appear in no output."""
    nb = load(path)
    outputs = _norm("\n".join(out_text(o) for c in nb["cells"] if c["cell_type"] == "code" and not is_row_preview(c)
                              for o in c.get("outputs", [])))
    heading, found = "(top)", []
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        text = "".join(c["source"])
        for line in text.splitlines():
            if line.startswith("#"):
                heading = line.lstrip("#").strip()[:70]
        for num, unit in NUMBER.findall(text):
            if _worth_checking(num, unit) and not any(_norm(v) in outputs for v in _variants(_norm(num))):
                pair = (heading, num + (unit or "").strip())
                if pair not in found:
                    found.append(pair)
    return found


def outputs(path, first, last, cap=2500):
    """Markdown and compact output text for cells first..last (inclusive)."""
    nb = load(path)
    lines = []
    for i in range(first, min(last, len(nb["cells"]) - 1) + 1):
        c = nb["cells"][i]
        if c["cell_type"] == "markdown":
            lines.append(f"\n##### md {i}\n" + "".join(c["source"]).strip())
        elif is_row_preview(c):
            lines.append(f"\n===== code {i}: (data-rows, skipped)")
        else:
            t = re.sub(r"( \|)+", " |", re.sub(r"\s+", " ", " ".join(out_text(o) for o in c.get("outputs", [])))).strip()
            if t:
                first_line = "".join(c["source"]).strip().splitlines()[0][:90] if c["source"] else ""
                lines.append(f"\n===== code {i}: {first_line}\n{t[:cap]}")
    return "\n".join(lines)


def status(path):
    """Run state of a notebook and whether its reports and text export are newer than it."""
    path = Path(path)
    nb = load(path)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errors = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error")
    mtime = path.stat().st_mtime
    root = path.parents[1]
    lines = [f"{path.stem}: {sum(1 for c in code if c.get('execution_count'))}/{len(code)} code cells ran, "
             f"{sum(1 for c in code if c.get('outputs'))} with outputs, {errors} errors; saved {datetime.fromtimestamp(mtime):%H:%M:%S}"]
    for rel in (f"reports/{path.stem}.md", f"reports/{path.stem}_full.md", f"docs/{path.stem}.md"):
        p = root / rel
        state = "missing" if not p.exists() else ("up to date" if p.stat().st_mtime >= mtime else "OLDER than the notebook")
        lines.append(f"   {rel:30} {state}")
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2 or argv[0] not in ("stale", "outputs", "status"):
        print(__doc__)
        return 1
    cmd, path = argv[0], argv[1]
    if cmd == "stale":
        found = stale(path)
        for heading, num in found:
            print(f"{heading} | {num}")
        print(f"\n{len(found)} quoted numbers not found in any output")
    elif cmd == "outputs":
        print(outputs(path, int(argv[2]), int(argv[3]), int(argv[4]) if len(argv) > 4 else 2500))
    else:
        print(status(path))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
