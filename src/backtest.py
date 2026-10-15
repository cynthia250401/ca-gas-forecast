"""Rolling-origin, expanding-window backtest and the final forecast."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from .models import ENSEMBLES, H


def _run_one(model, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    C.set_seeds()
    model = copy.deepcopy(model)
    rows = []
    for i, T in enumerate(origins):
        if i % C.REFIT_EVERY == 0:
            model.fit(panel, T)
        fc = model.predict(panel, T)
        rows.append(pd.DataFrame({"origin": T, "model": model.name, "h": H, "fc_chg": fc}))
    return pd.concat(rows)


def add_ensembles(fc: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average of member log-change forecasts (members fixed a priori)."""
    wide = fc.pivot_table(index=["origin", "h"], columns="model", values="fc_chg")
    extra = []
    for name, members in ENSEMBLES.items():
        s = wide[members].mean(axis=1).rename("fc_chg").reset_index()
        s["model"] = name
        extra.append(s)
    return pd.concat([fc] + extra, ignore_index=True)


def attach_actuals(fc: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    fc["origin"] = pd.to_datetime(fc["origin"])
    fc["target_date"] = fc["origin"] + pd.to_timedelta(7 * fc["h"], unit="D")
    fc["origin_price"] = fc["origin"].map(panel["ca"])
    fc["fc_price"] = fc["origin_price"] * np.exp(fc["fc_chg"])
    fc["actual_price"] = fc["target_date"].map(panel["ca"])
    return fc


def run_backtest(panel: pd.DataFrame, models: list, start=C.BACKTEST_START, n_jobs: int = -1) -> pd.DataFrame:
    origins = panel.index[panel.index >= start]
    parts = Parallel(n_jobs=n_jobs)(delayed(_run_one)(m, panel, origins) for m in models)
    fc = add_ensembles(pd.concat(parts, ignore_index=True))
    return attach_actuals(fc, panel)


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
