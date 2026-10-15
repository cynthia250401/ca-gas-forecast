"""'What if Brent moves +/-X%?' scenario using the Ridge driver model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


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
