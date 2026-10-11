"""Shim: archived experiments import `job_matching_engine`, which lives in the repo's live `code/` folder.

Importing this module loads the real `code/job_matching_engine.py` under the same module name, so the frozen
experiment scripts and their tests keep working without a copy of the engine.
"""

import importlib.util
import sys
from pathlib import Path

_REAL = Path(__file__).resolve().parents[3] / "code" / "job_matching_engine.py"
_spec = importlib.util.spec_from_file_location(__name__, _REAL)
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
# When this file is loaded by path (importlib), the caller keeps this shim object, so expose the real contents here too.
globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})
