from __future__ import annotations
from pathlib import Path
import pandas as pd

# Make Streamlit optional: provide a no-op cache decorator when Streamlit
# isn't available (e.g., when using the Dash app without pyarrow/streamlit).
try:  # pragma: no cover - light compatibility shim
    import streamlit as _st  # type: ignore

    _cache_data = _st.cache_data
except Exception:  # Streamlit not installed or import error
    def _cache_data(show_spinner: bool = False):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def default_csv_path() -> Path:
    """Return the first CSV file found in the project's data directory."""

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    return csv_files[0]


@_cache_data(show_spinner=False)
def load_csv(path: str | Path | None) -> pd.DataFrame:
    """Load a CSV from an absolute or project-relative path.

    - If `path` is None or empty, auto-pick the first CSV in `<project>/data`.
    - If `path` is relative, resolve it against the project root.
    - Provides a clearer error listing available CSV files.
    """
    candidates: list[Path] = []

    if path:
        p = Path(path)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(PROJECT_ROOT / p)
    else:
        candidates.extend(sorted(DATA_DIR.glob("*.csv")))

    for c in candidates:
        if c.exists() and c.is_file():
            return pd.read_csv(c)

    available = sorted(DATA_DIR.glob("*.csv"))
    hint = "\n".join(f" - {a}" for a in available) or "(no CSV files found in ./data)"
    raise FileNotFoundError(
        f"Could not locate CSV. Tried: {', '.join(map(str, candidates)) or '(none)'}\nAvailable in ./data:\n{hint}"
    )


def ensure_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
