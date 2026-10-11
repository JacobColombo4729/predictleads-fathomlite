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
PART2_TABLE = CLEAN_DIR / "scope_table.pkl"  # EDA_SDA_2's final table (the scope), for EDA_SDA_3

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

def finish():
    plt.tight_layout()
    plt.show()

# ---------- summary tables ----------

def show(title, df):
    """Print a heading, then display a summary table. Aggregates only, never rows."""
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
    Listings still open at the snapshot count as censored, so they don't bias the window short."""
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
