"""Accuracy metrics, Diebold-Mariano tests, validation/holdout split, regime breakdown."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

RW = "Random walk"


def dm_test(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss with the Harvey-Leybourne-Newbold
    small-sample correction and Newey-West variance (lag h-1, for overlapping h-step errors).
    Negative statistic => model has lower squared error than the benchmark."""
    d = e_model ** 2 - e_bench ** 2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    lrv = dc @ dc / n
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def _metrics(g: pd.DataFrame) -> pd.Series:
    e = g.actual_price - g.fc_price
    fc_dir = np.sign(g.fc_price - g.origin_price)
    act_dir = np.sign(g.actual_price - g.origin_price)
    valid = fc_dir != 0
    return pd.Series({
        "n": len(g),
        "MAE": e.abs().mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "MAPE": 100 * (e.abs() / g.actual_price).mean(),
        "DirAcc": 100 * (fc_dir[valid] == act_dir[valid]).mean() if valid.any() else np.nan,
    })


def accuracy_table(bt: pd.DataFrame, mask=None, horizons=C.REPORT_H) -> pd.DataFrame:
    """Metrics by model x horizon on the rows selected by `mask` (default: all scored rows)."""
    d = bt[bt.actual_price.notna() & bt.h.isin(horizons)]
    if mask is not None:
        d = d[mask(d)]
    tab = d.groupby(["model", "h"]).apply(_metrics, include_groups=False).reset_index()
    rw = tab[tab.model == RW].set_index("h")["RMSE"]
    tab["RelRMSE"] = tab.RMSE / tab.h.map(rw)
    # Diebold-Mariano vs random walk on the same target dates
    dm_rows = []
    rw_e = d[d.model == RW].set_index(["h", "origin"])
    for (m, h), g in d.groupby(["model", "h"]):
        g = g.set_index(["h", "origin"])
        common = g.index.intersection(rw_e.index)
        e1 = (g.loc[common].actual_price - g.loc[common].fc_price).to_numpy()
        e0 = (rw_e.loc[common].actual_price - rw_e.loc[common].fc_price).to_numpy()
        stat, p = dm_test(e1, e0, h) if m != RW else (np.nan, np.nan)
        dm_rows.append({"model": m, "h": h, "DM_stat": stat, "DM_p": p})
    return tab.merge(pd.DataFrame(dm_rows), on=["model", "h"])


def by_year(lo: int, hi: int):
    return lambda d: d.target_date.dt.year.between(lo, hi)


def by_dates(start, end):
    return lambda d: d.target_date.between(pd.Timestamp(start), pd.Timestamp(end))


def select_top_model(valid_tab: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Rank models by mean RMSE relative to the random walk over the reported horizons."""
    rank = (valid_tab.groupby("model")
            .agg(mean_RelRMSE=("RelRMSE", "mean"), mean_MAE=("MAE", "mean"), mean_DirAcc=("DirAcc", "mean"))
            .sort_values("mean_RelRMSE"))
    return rank.index[0], rank


def regime_table(bt: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name, (s, e) in C.REGIMES.items():
        t = accuracy_table(bt, by_dates(s, e))
        t.insert(0, "regime", name)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def wide(tab: pd.DataFrame, value: str = "RelRMSE") -> pd.DataFrame:
    return tab.pivot(index="model", columns="h", values=value).add_prefix("h=")
