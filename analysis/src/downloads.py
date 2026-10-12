"""Shared by the download scripts (get_predictleads, get_resumes): a disk-space check before anything is written."""
import shutil
import sys
from pathlib import Path

MARGIN = 2_000_000_000  # keep 2 GB free after a download, for the OS and the notebooks' own outputs


def gb(n):
    return f"{n / 1e9:,.2f} GB"


def confirm_space(need, folder, what, yes=False):
    """Say how much `what` needs and how much space is free where it's saved, then ask before going on.
    Stops if it doesn't fit (with MARGIN to spare). `yes=True` skips the question (for scripts and --yes)."""
    folder = Path(folder)
    probe = next(p for p in (folder, *folder.parents) if p.exists())
    free = shutil.disk_usage(probe).free
    if need + MARGIN > free:
        raise SystemExit(f"Not enough disk space: {what} needs {gb(need)} and {gb(MARGIN)} should stay free, "
                         f"but only {gb(free)} is free. Free up {gb(need + MARGIN - free)} and run it again.")
    print(f"{what}: needs {gb(need)}; free on this drive: {gb(free)}; left after: {gb(free - need)}")
    if yes:
        return
    if not sys.stdin or not sys.stdin.isatty():
        try:  # notebooks have no terminal but can still ask
            from IPython import get_ipython
            interactive = get_ipython() is not None
        except ImportError:
            interactive = False
        if not interactive:
            raise SystemExit("No one to ask: rerun with --yes to confirm the download.")
    try:
        answer = input("Proceed? [y/N] ")
    except EOFError:  # no one typing (input redirected): don't guess
        raise SystemExit("No answer: rerun in a terminal, or with --yes to confirm the download.") from None
    if answer.strip().lower() not in {"y", "yes"}:
        raise SystemExit("Cancelled; nothing was downloaded.")
