"""Shared helpers for the EDA_SDA notebooks: paths, survival curves, summary tables and charts.

Everything here works on aggregates. Nothing prints listing rows.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# ---------- paths ----------

def project_root(start=None):
    """The nearest folder at or above `start` that holds requirements.txt."""
    start = Path(start or Path.cwd()).resolve()
    return next(d for d in [start, *start.parents] if (d / "requirements.txt").exists())

ROOT = project_root()
DATA_DIR = Path(os.environ.get("PL_DATA_DIR") or ROOT / "data")  # gitignored; PL_DATA_DIR overrides (tests on synthetic data)
CLEAN_DIR = DATA_DIR / "clean"
PART1_TABLE = CLEAN_DIR / "analysis_table.pkl"  # EDA_SDA_1's final table
PART2_TABLE = CLEAN_DIR / "scope_table.parquet"  # EDA_SDA_2's final table (the scope), for EDA_SDA_3

# ---------- chart style ----------

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical order: never reorder
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#c9c8c2", "#e4e3df", "#fcfcfb"

def use_chart_style():
    """The chart style shared by every notebook."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "font.size": 10.5, "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 110,
    })

def bar_h(ax, s, title, xlabel, fmt="{:,.0f}", color=BLUE):
    """Horizontal bars, largest at the top, each value labelled at the bar end."""
    s = s.dropna().sort_values()
    ax.barh([str(i)[:42] for i in s.index], s.values, color=color, height=0.65)
    for y, v in enumerate(s.values):
        ax.text(v, y, " " + fmt.format(v), va="center", color=INK_2, fontsize=8.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18)

def heatmap(ax, table, title, fmt="{:,.0f}", cmap="Blues", vmin=None, vmax=None):
    """Annotated heatmap of a small aggregate table (rows × columns)."""
    vals = table.values.astype(float)
    ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c)[:28] for c in table.columns], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(table.shape[0]), [str(i)[:34] for i in table.index], fontsize=8.5)
    finite = vals[np.isfinite(vals)]
    lo = vmin if vmin is not None else (finite.min() if finite.size else 0)
    hi = vmax if vmax is not None else (finite.max() if finite.size else 1)
    diverging = lo < 0 < hi
    for (r, c), v in np.ndenumerate(vals):
        if np.isfinite(v):
            depth = abs(v) / max(abs(lo), abs(hi)) if diverging else (v - lo) / ((hi - lo) or 1)
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8, color="white" if depth > 0.6 else INK)
    ax.set_title(title)
    ax.grid(False)

# ---------- shaded comparison tables ----------
# Wide tables comparing groups across measures: each column shaded on its own scale (darker = higher), numbers kept, and
# formats chosen from the column name: shares as whole percentages, days and years with one decimal, pay in dollars,
# counts compact. In the Markdown reports the shading drops out and the formatted numbers remain.

def compact(n):
    """13,017,464 -> 13.0M; 412,345 -> 412K; under 10,000 with commas."""
    if pd.isna(n):
        return ""
    n = float(n)
    if abs(n) >= 1e6:
        return f"{n / 1e6:.1f}M"
    if abs(n) >= 1e4:
        return f"{n / 1e3:.0f}K"
    return f"{n:,.0f}"


def cell_format(name, values):
    """A formatter for one column (or row), from its name and values: shares as percentages (small shares with
    decimals), pay in dollars, days and years with one decimal, counts compact (13.0M, 412K)."""
    name = str(name).lower()
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    blank = lambda f: (lambda x: "" if pd.isna(x) else f(x))
    in_01 = len(v) > 0 and v.between(0, 1).all()
    whole = len(v) > 0 and (v == v.round()).all()
    counts = any(k in name for k in ("listings", "companies", "postings", "per week", "count", "people", "sub-groups"))
    if "rank" in name:
        return blank(lambda x: f"{x:.0f}")
    if "spread" in name:
        return blank(lambda x: f"{x:.1f}×")
    named_share = any(k in name for k in ("share", "rate", "closed within", "open after", "still open", "or fewer",
                                          "or more", "states years"))
    looks_share = not whole and not any(k in name for k in ("median", "mean", "lag", "width", "usd", "pay", "words",
                                                            "skew", "iqr", "fence", "gap"))
    if in_01 and not counts and (named_share or looks_share):
        return blank(lambda x: f"{x:.2%}" if 0 < abs(x) < 0.01 else f"{x:.1%}" if abs(x) < 0.1 else f"{x:.0%}")
    if not in_01 and any(k in name for k in ("usd", "pay ($", "hourly pay")):
        return blank(lambda x: f"${x:,.2f}")
    if "words" in name or whole:
        return compact
    if len(v) and v.abs().max() < 1:
        return blank(lambda x: f"{x:.3f}")
    if len(v) and v.abs().max() < 10_000:
        return blank(lambda x: f"{x:,.1f}")
    return compact


def shaded(df, highlight=None, cmap="Blues"):
    """A Styler: numeric columns shaded per column, formatted by cell_format, the highlight row in bold."""
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    sty = df.style.format({c: cell_format(c, df[c]) for c in num})
    for c in num:
        if df[c].notna().sum() > 1 and df[c].nunique() > 1:
            sty = sty.background_gradient(cmap=cmap, subset=[c], low=0.05, high=0.35)
    if highlight is not None and highlight in df.index:
        sty = sty.apply(lambda row: ["font-weight: bold; border-top: 2px solid #eb6834; border-bottom: 2px solid #eb6834"
                                     if row.name == highlight else "" for _ in row], axis=1)
    return sty


def show_shaded(title, df, highlight=None):
    """Print a heading, then a shaded comparison table. Aggregates only, never rows."""
    print(f"\n{title}")
    display(shaded(df, highlight))


def column_heatmap(ax, df, title, highlight=None, cmap="Blues"):
    """A heat map of a comparison table for the reports: each column on its own scale, values written in, one row
    outlined. Columns are measures, rows are groups."""
    num = df[[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]]
    span = (num.max() - num.min()).replace(0, np.nan)
    depth = ((num - num.min()) / span).fillna(0.5)
    ax.imshow(depth.to_numpy(dtype=float), cmap=cmap, vmin=-0.15, vmax=1.6, aspect="auto")
    import textwrap
    ax.set_xticks(range(num.shape[1]), ["\n".join(textwrap.wrap(str(c), 16)) for c in num.columns], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(num.shape[0]), [str(i)[:34] for i in num.index], fontsize=8)
    for j, c in enumerate(num.columns):
        f = cell_format(c, num[c])
        for i, v in enumerate(num[c]):
            if pd.notna(v):
                ax.text(j, i, f(v), ha="center", va="center", fontsize=7, color="white" if depth.iloc[i, j] > 0.75 else INK)
    if highlight is not None and highlight in num.index:
        r = list(num.index).index(highlight)
        ax.add_patch(plt.Rectangle((-0.5, r - 0.5), num.shape[1], 1, fill=False, edgecolor=ORANGE, lw=2.2))
    ax.set_title(title)
    ax.grid(False)


def finish():
    plt.tight_layout()
    plt.show()

# ---------- summary tables ----------

def show(title, df, by="columns"):
    """Print a heading, then display a summary table. Aggregates only, never rows.
    Numbers are formatted from the column name (or the row name, by="rows", when rows are measures): shares as
    percentages, days and years with one decimal, pay in dollars, counts compact (13.0M, 412K). describe() tables keep
    their count row as a count. Wide group comparisons use show_shaded instead."""
    df = df.to_frame() if isinstance(df, pd.Series) else df.copy()
    describe = {"count", "mean", "50%"} <= set(map(str, df.index))
    numeric = lambda s: pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if by == "rows":
        out = df.astype(object)
        special = [c for c in df.columns if any(k in str(c) for k in ("rank", "spread"))]
        for r in df.index:
            f = cell_format(r, pd.to_numeric(df.loc[r].drop(special), errors="coerce"))
            for c in df.columns:
                v = df.at[r, c]
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    out.at[r, c] = cell_format(c, df[c])(v) if c in special else f(v)
        df = out
    else:
        for c in df.columns:
            if numeric(df[c]):
                f = cell_format(c, df[c].drop(index="count", errors="ignore") if describe else df[c])
                df[c] = [compact(v) if describe and str(i) == "count" else f(v) for i, v in df[c].items()]
    print(f"\n{title}")
    display(df)

def pct(x):
    return f"{100 * x:.1f}%"

def as_list(v):
    """Nested list fields come back as lists or numpy arrays; normalize to a list."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    return [] if v is None or (isinstance(v, float) and np.isnan(v)) else [v]

# ---------- survival (Kaplan–Meier) ----------

def km_curve(days, finished):
    """Share of listings still open after t days, indexed by t.
    Listings still open on September 30 (the data's end date) count as censored, so they don't bias the window short."""
    d = pd.DataFrame({"t": days, "e": finished}).dropna()
    by_t = d.groupby("t")["e"].agg(["size", "sum"])
    at_risk = len(d) - by_t["size"].cumsum().shift(fill_value=0)
    return (1 - by_t["sum"] / at_risk).cumprod()

def surv_at(curve, x):
    return curve[curve.index <= x].iloc[-1] if (curve.index <= x).any() else 1.0

def km_survival(days, finished, at):
    curve = km_curve(days, finished)
    return [surv_at(curve, x) for x in at]

def km_median(days, finished):
    """Days until half of listings have closed; NaN if more than half are still open."""
    curve = km_curve(days, finished)
    return curve[curve <= 0.5].index.min()
