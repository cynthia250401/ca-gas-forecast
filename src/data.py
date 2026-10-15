"""Load the EIA CSVs, profile them, and build the weekly feature panel with no look-ahead.

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
        df = pd.read_csv(C.DATA_DIR / fname)
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
        full = pd.read_csv(C.DATA_DIR / fname)
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


def _years_back(k: int) -> int:
    return int(round(k * C.WEEKS_PER_YEAR))


def _five_year_dev(s: pd.Series) -> pd.Series:
    """% deviation of a weekly (Friday) series from its same-week average of the prior 5 years."""
    past = pd.concat([s.shift(_years_back(k)) for k in range(1, 6)], axis=1)
    avg = past.mean(axis=1).where(past.notna().sum(axis=1) == 5)
    return 100 * (s / avg - 1)


# ----------------------------------------------------------------------------- panel
FEATURES_BASE = [
    "ca_chg_1w", "ca_chg_4w",
    "brent_chg_1w", "brent_chg_4w", "brent_chg_12w",
    "margin_dev", "spread_dev",
    "inv_p5_dev", "inv_us_dev",
    "util_p5", "util_p5_chg_4w",
    "woy_sin", "woy_cos",
]
FEATURE_LABELS = {
    "ca_chg_1w": "CA log chg 1w", "ca_chg_4w": "CA log chg 4w",
    "brent_chg_1w": "Brent log chg 1w", "brent_chg_4w": "Brent log chg 4w",
    "brent_chg_12w": "Brent log chg 12w",
    "margin_dev": "Margin vs 52w mean", "spread_dev": "CA-US spread vs 52w mean",
    "inv_p5_dev": "PADD5 stocks vs 5y avg", "inv_us_dev": "US stocks vs 5y avg",
    "util_p5": "PADD5 utilisation", "util_p5_chg_4w": "PADD5 util chg 4w",
    "woy_sin": "Week-of-year sin", "woy_cos": "Week-of-year cos",
    "seas": "Seasonal expectation (h)",
}


def build_panel(raw: dict[str, pd.Series] | None = None, origin=None) -> pd.DataFrame:
    """Weekly panel on Monday CA price dates.

    If `origin` is given, raw data are first truncated to what was published by then,
    so the last row is exactly what a forecaster standing at `origin` could compute.
    Columns `seas_h` hold the seasonal expectation for horizon h, `tgt_h` the realised
    log change ln(CA_{t+h}) - ln(CA_t) (NaN where not yet observed).
    """
    raw = raw or load_raw()
    if origin is not None:
        raw = truncate_to_origin(raw, origin)
    ca = raw["ca"]
    idx = ca.index
    p = pd.DataFrame(index=idx)
    p.index.name = "date"
    p["ca"] = ca
    p["us"] = raw["us"].reindex(idx)
    p["brent"] = _asof(raw["brent"], idx, C.CRUDE_LAG_DAYS)
    p["wti"] = _asof(raw["wti"], idx, C.CRUDE_LAG_DAYS)
    # EIA weekly: transform on the native Friday series, then align with the release lag
    p["stocks_p5"] = _asof(raw["stocks_p5"], idx, C.EIA_WEEKLY_LAG_DAYS)
    p["stocks_us"] = _asof(raw["stocks_us"], idx, C.EIA_WEEKLY_LAG_DAYS)
    p["inv_p5_dev"] = _asof(_five_year_dev(raw["stocks_p5"]), idx, C.EIA_WEEKLY_LAG_DAYS)
    p["inv_us_dev"] = _asof(_five_year_dev(raw["stocks_us"]), idx, C.EIA_WEEKLY_LAG_DAYS)
    util = raw["util_p5"]
    p["util_p5"] = _asof(util, idx, C.EIA_WEEKLY_LAG_DAYS)
    p["util_p5_chg_4w"] = _asof(util - util.shift(4), idx, C.EIA_WEEKLY_LAG_DAYS)
    p["util_us"] = _asof(raw["util_us"], idx, C.EIA_WEEKLY_LAG_DAYS)

    ly = np.log(p["ca"])
    lb = np.log(p["brent"])
    p["ly"] = ly
    p["ca_chg_1w"] = ly.diff(1)
    p["ca_chg_4w"] = ly.diff(4)
    p["brent_chg_1w"] = lb.diff(1)
    p["brent_chg_4w"] = lb.diff(4)
    p["brent_chg_12w"] = lb.diff(12)
    p["margin"] = p["ca"] - p["brent"] / C.BBL_TO_GAL
    p["margin_dev"] = p["margin"] - p["margin"].rolling(52).mean()
    p["spread"] = p["ca"] - p["us"]
    p["spread_dev"] = p["spread"] - p["spread"].rolling(52).mean()
    angle = 2 * np.pi * (idx.dayofyear - 1) / 365.25
    p["woy_sin"], p["woy_cos"] = np.sin(angle), np.cos(angle)

    # Seasonal expectation: mean log change over the next h weeks at this time of year,
    # across the prior 10 calendar years. Every term ends at t - 52k + h <= t - 40: past only.
    for h in C.HORIZONS:
        terms = pd.concat([ly.shift(_years_back(k) - h) - ly.shift(_years_back(k))
                           for k in range(1, C.SEASONAL_YEARS + 1)], axis=1)
        p[f"seas_{h}"] = terms.mean(axis=1).where(terms.notna().sum(axis=1) >= C.SEASONAL_MIN_YEARS)
        p[f"tgt_{h}"] = ly.shift(-h) - ly
    return p


def check_no_lookahead(panel: pd.DataFrame, raw: dict[str, pd.Series] | None = None,
                       n_checks: int = 25, seed: int = C.SEED) -> pd.DataFrame:
    """Rebuild the panel from data truncated at sampled origins and compare the last row.

    If any feature at origin T used information published after T, the truncated rebuild
    would differ from the full-sample panel. Returns one row per origin with the max abs diff.
    """
    raw = raw or load_raw()
    feats = FEATURES_BASE + [f"seas_{h}" for h in C.HORIZONS]
    cand = panel.index[(panel.index >= "2012-01-01")]
    rng = np.random.default_rng(seed)
    origins = sorted(set(rng.choice(cand, size=n_checks, replace=False)) | {panel.index[-1]})
    rows = []
    for T in origins:
        trunc = build_panel(raw, origin=T)
        a, b = trunc.loc[T, feats].astype(float), panel.loc[T, feats].astype(float)
        diff = (a - b).abs().max()
        rows.append({"origin": pd.Timestamp(T).date(), "max_abs_diff": diff,
                     "ok": bool(np.isclose(a, b, equal_nan=True, atol=1e-12).all())})
    out = pd.DataFrame(rows)
    if not out["ok"].all():
        raise AssertionError(f"Look-ahead detected:\n{out[~out.ok]}")
    return out
