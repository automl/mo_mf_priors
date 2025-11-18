from __future__ import annotations

from pathlib import Path

DEFAULT_ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DEFAULT_PRIORS_DIR = DEFAULT_ROOT_DIR / "src" / "priors"
DEFAULT_RESULTS_DIR = DEFAULT_ROOT_DIR.parent / "momf_priors_results"
DEFAULT_SCRIPTS_DIR = DEFAULT_ROOT_DIR / "scripts" / "meta"
DEFAULT_DATA_DIR = DEFAULT_ROOT_DIR.parent.absolute() / "data"