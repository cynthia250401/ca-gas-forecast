"""Final 12-week forecast from the last data date, empirical prediction intervals,
holdout interval coverage, and the 'what if Brent moves +/-X%?' scenario."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import config as C
from .backtest import add_ensembles, attach_actuals
from .models import H


def final_forecast(panel: pd.DataFrame, models: list) -> tuple[pd.DataFrame, dict]:
    """Fit every model on all data available at the last date and forecast 12 weeks."""
    T = panel.index[-1]
    rows, fitted = [], {}
    for m in models:
        m = copy.deepcopy(m)
        m.fit(panel, T)
        fitted[m.name] = m
        rows.append(pd.DataFrame({"origin": T, "model": m.name, "h": H, "fc_chg": m.predict(panel, T)}))
    fc = attach_actuals(add_ensembles(pd.concat(rows, ignore_index=True)), panel)
    return fc.drop(columns="actual_price"), fitted


def interval_quantiles(bt: pd.DataFrame, model: str) -> pd.DataFrame:
    """Empirical quantiles of backtest log errors ln(actual) - ln(forecast) by horizon."""
    d = bt[(bt.model == model) & bt.actual_price.notna()]
    err = np.log(d.actual_price) - np.log(d.fc_price)
    q = err.groupby(d.h).quantile([0.05, 0.10, 0.90, 0.95]).unstack()
    q.columns = ["q05", "q10", "q90", "q95"]
    q["n_errors"] = err.groupby(d.h).size()
    return q


def add_intervals(final: pd.DataFrame, q: pd.DataFrame, model: str) -> pd.DataFrame:
    f = final[final.model == model].set_index("h").copy()
    for lo, hi, lab in [("q10", "q90", "80"), ("q05", "q95", "90")]:
        f[f"lo{lab}"] = f.fc_price * np.exp(q[lo])
        f[f"hi{lab}"] = f.fc_price * np.exp(q[hi])
    return f.reset_index()


def holdout_coverage(bt: pd.DataFrame, model: str) -> pd.DataFrame:
    """Share (%) of holdout outcomes inside bands built from validation-period errors only."""
    q_val = interval_quantiles(bt[bt.target_date.dt.year <= C.VALID_YEARS[1]], model)
    h_rows = bt[(bt.model == model) & bt.actual_price.notna() & (bt.target_date.dt.year >= C.HOLDOUT_YEARS[0])]
    err = np.log(h_rows.actual_price) - np.log(h_rows.fc_price)
    return pd.DataFrame({
        "cover80": (err >= h_rows.h.map(q_val.q10)) & (err <= h_rows.h.map(q_val.q90)),
        "cover90": (err >= h_rows.h.map(q_val.q05)) & (err <= h_rows.h.map(q_val.q95)),
        "h": h_rows.h}).groupby("h").mean().mul(100).round(1)


# ----------------------------------------------------------------------------- Brent scenario
def brent_shock_override(panel: pd.DataFrame, pct: float, T=None) -> dict:
    """Feature shifts if Brent had moved `pct`% in the latest week (retail not yet adjusted).

    All three Brent log-change features rise by ln(1+pct), and the retail-minus-crude margin
    falls by Brent * pct / 42 (the pump price has not moved yet, crude cost per gallon has).
    """
    T = T or panel.index[-1]
    dlog = np.log1p(pct / 100)
    dmargin = -panel.loc[T, "brent"] * (pct / 100) / C.BBL_TO_GAL
    return {"brent_chg_1w": dlog, "brent_chg_4w": dlog, "brent_chg_12w": dlog, "margin_dev": dmargin}


def brent_scenario(panel: pd.DataFrame, ridge_model, pct: float) -> pd.DataFrame:
    T = panel.index[-1]
    base = ridge_model.predict(panel, T)
    shocked = ridge_model.predict(panel, T, override=brent_shock_override(panel, pct, T))
    p0 = panel.loc[T, "ca"]
    out = pd.DataFrame({
        "h": C.HORIZONS,
        "target_date": [T + pd.Timedelta(weeks=h) for h in C.HORIZONS],
        "baseline": p0 * np.exp(base),
        "scenario": p0 * np.exp(shocked),
    })
    out["difference"] = out.scenario - out.baseline
    # Mechanical full pass-through benchmark: crude cost per gallon changes 1-for-1
    out["full_passthrough_diff"] = panel.loc[T, "brent"] * (pct / 100) / C.BBL_TO_GAL
    return out
