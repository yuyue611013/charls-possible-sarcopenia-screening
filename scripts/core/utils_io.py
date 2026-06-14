from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import pandas as pd


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_output_dirs() -> None:
    for relative in ("output/logs", "output/tables", "output/figures"):
        project_path(relative).mkdir(parents=True, exist_ok=True)


def load_json(relative_path: str) -> dict[str, Any]:
    with project_path(relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(project_path(relative_path), low_memory=False)


def write_text(relative_path: str, text: str) -> None:
    path = project_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_dataframe_csv(df: pd.DataFrame, relative_path: str) -> None:
    path = project_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def column_exists(df: pd.DataFrame, column_name: str | None) -> bool:
    return bool(column_name) and column_name in df.columns


def non_missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.notna()
    return series.notna() & (series.astype(str).str.strip() != "")


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")
