"""Engineered features on the weekly Monday panel, the look-ahead check, and model_features.csv.

Every feature at Monday T uses only data published by T (see data_prep.py for the release lags);
`check_no_lookahead` verifies this by rebuilding the panel from data truncated at sampled origins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data_prep as D


# ----------------------------------------------------------------------------- helpers
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
    raw = raw or D.load_raw()
    if origin is not None:
        raw = D.truncate_to_origin(raw, origin)
    w = D.align_weekly(raw)
    idx = w.index
    p = w[["ca", "us", "brent", "wti", "stocks_p5", "stocks_us"]].copy()
    # EIA weekly: transform on the native Friday series, then align with the release lag
    p["inv_p5_dev"] = D._asof(_five_year_dev(raw["stocks_p5"]), idx, C.EIA_WEEKLY_LAG_DAYS)
    p["inv_us_dev"] = D._asof(_five_year_dev(raw["stocks_us"]), idx, C.EIA_WEEKLY_LAG_DAYS)
    util = raw["util_p5"]
    p["util_p5"] = w["util_p5"]
    p["util_p5_chg_4w"] = D._asof(util - util.shift(4), idx, C.EIA_WEEKLY_LAG_DAYS)
    p["util_us"] = w["util_us"]

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
    raw = raw or D.load_raw()
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


def lookahead_mutation_test(raw, panel) -> bool:
    """Sanity-check the checker: if EIA data were aligned with a 3-day lag instead of the
    true 10-day release lag, the look-ahead check must fail."""
    saved = C.EIA_WEEKLY_LAG_DAYS
    try:
        C.EIA_WEEKLY_LAG_DAYS = 3
        leaky = build_panel(raw)
        try:
            check_no_lookahead(leaky, raw, n_checks=10)
            return False
        except AssertionError:
            return True
    finally:
        C.EIA_WEEKLY_LAG_DAYS = saved


def save_model_features(raw: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Write data/processed/model_features.csv (levels, features, seas_h and tgt_h)."""
    panel = build_panel(raw)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(C.MODEL_FEATURES_CSV)
    return panel


if __name__ == "__main__":
    panel = save_model_features()
    print(f"Wrote {C.MODEL_FEATURES_CSV.relative_to(C.ROOT)}: {panel.shape}")
