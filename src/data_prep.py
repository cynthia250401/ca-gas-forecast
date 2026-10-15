"""Load the raw EIA CSVs, profile them, and align them to a weekly Monday panel with no look-ahead.

Timing convention (everything is indexed on the Monday retail-price date T):
  * CA / US retail prices for Monday T are known at T.
  * Crude for Monday T is the last daily close on or before the prior Friday (T - 3 days).
  * EIA weekly stocks / utilisation for the week ending Friday F are released the
    following Wednesday, so they are usable from Monday F + 10. At origin T we may only
    use observations with F <= T - 10 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------- loading
def load_raw() -> dict[str, pd.Series]:
    """Return {name: float Series indexed by observation date}, blanks dropped."""
    raw = {}
    for name, (fname, _) in C.RAW_FILES.items():
        df = pd.read_csv(C.RAW_DIR / fname)
        df.columns = ["date", "value"]
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                      index=pd.to_datetime(df["date"]), name=name).sort_index()
        raw[name] = s.dropna()
    return raw


def profile(raw: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """One row per series: coverage, frequency, gaps, range."""
    raw = raw or load_raw()
    rows = []
    for name, (fname, lag) in C.RAW_FILES.items():
        full = pd.read_csv(C.RAW_DIR / fname)
        n_blank = pd.to_numeric(full.iloc[:, 1], errors="coerce").isna().sum()
        s = raw[name]
        gaps = s.index.to_series().diff().dt.days.dropna()
        weekday = s.index.day_name().value_counts()
        rows.append({
            "series": name, "file": fname,
            "start": s.index.min().date(), "end": s.index.max().date(),
            "n_obs": len(s), "blank_rows": int(n_blank),
            "frequency": "daily (business)" if gaps.median() == 1 else "weekly",
            "weekday": ", ".join(weekday.index[:2]) if gaps.median() > 1 else "Mon-Fri",
            "max_gap_days": int(gaps.max()),
            "min": round(s.min(), 2), "max": round(s.max(), 2), "last": round(s.iloc[-1], 3),
            "usable_lag_days": lag,
        })
    return pd.DataFrame(rows)


def truncate_to_origin(raw: dict[str, pd.Series], origin) -> dict[str, pd.Series]:
    """Keep only what was *published* by Monday `origin` (applies the release lags)."""
    origin = pd.Timestamp(origin)
    return {k: s[s.index <= origin - pd.Timedelta(days=C.RAW_FILES[k][1])] for k, s in raw.items()}


# ----------------------------------------------------------------------------- helpers
def _asof(s: pd.Series, dates: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    """Value of `s` at the latest observation date <= date - lag_days."""
    lookup = dates - pd.Timedelta(days=lag_days)
    out = s.reindex(s.index.union(lookup)).ffill().reindex(lookup)
    out.index = dates
    return out


# ----------------------------------------------------------------------------- weekly panel
WEEKLY_COLUMNS = ["ca", "us", "brent", "wti", "stocks_p5", "stocks_us", "util_p5", "util_us"]


def align_weekly(raw: dict[str, pd.Series] | None = None, origin=None) -> pd.DataFrame:
    """All eight series on the Monday CA price dates, each as of its release lag:
    retail prices same day, crude T - 3 days, EIA weekly data T - 10 days."""
    raw = raw or load_raw()
    if origin is not None:
        raw = truncate_to_origin(raw, origin)
    idx = raw["ca"].index
    w = pd.DataFrame(index=idx)
    w.index.name = "date"
    w["ca"] = raw["ca"]
    w["us"] = raw["us"].reindex(idx)
    for k in ("brent", "wti"):
        w[k] = _asof(raw[k], idx, C.CRUDE_LAG_DAYS)
    for k in ("stocks_p5", "stocks_us", "util_p5", "util_us"):
        w[k] = _asof(raw[k], idx, C.EIA_WEEKLY_LAG_DAYS)
    return w[WEEKLY_COLUMNS]


def save_weekly_panel(raw: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Write data/processed/weekly_panel.csv."""
    w = align_weekly(raw)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    w.to_csv(C.WEEKLY_PANEL_CSV)
    return w


if __name__ == "__main__":
    w = save_weekly_panel()
    print(f"Wrote {C.WEEKLY_PANEL_CSV.relative_to(C.ROOT)}: {w.shape}")
