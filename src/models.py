"""Forecasting models. Each model forecasts the log change ln(CA_{T+h}) - ln(CA_T) for h = 1..12.

Interface:
    m.fit(panel, T)     re-estimate parameters with data available at origin T
    m.predict(panel, T) -> np.ndarray of 12 log-change forecasts from origin T
`panel` may contain future rows; models only ever read rows <= T, and the direct
regressions only train on rows whose target date t + h <= T.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config as C
from .features import FEATURES_BASE

H = np.array(C.HORIZONS)
warnings.filterwarnings("ignore", module="statsmodels")


def _ly_upto(panel: pd.DataFrame, T) -> np.ndarray:
    return panel.loc[:T, "ly"].to_numpy()


class RandomWalk:
    name = "Random walk"

    def fit(self, panel, T):
        pass

    def predict(self, panel, T):
        return np.zeros(len(H))


class SeasonalNaive:
    """Last year's change over the same weeks, applied to today's level:
    y_{T+h} = y_T + (y_{T-52+h} - y_{T-52})."""
    name = "Seasonal naive"

    def fit(self, panel, T):
        pass

    def predict(self, panel, T):
        ly = panel.loc[:T, "ly"]
        base = ly.iloc[-53]
        return np.array([ly.iloc[-53 + h] - base for h in H])


class SeasonalMean:
    """Average same-season change over the prior 10 years (the `seas_h` feature)."""
    name = "Seasonal mean (10y)"

    def fit(self, panel, T):
        pass

    def predict(self, panel, T):
        return panel.loc[T, [f"seas_{h}" for h in H]].to_numpy(dtype=float)


def select_arima_order(panel: pd.DataFrame, end=C.ARIMA_SELECT_END, max_pq: int = 3):
    """Grid-search ARIMA(p,1,q) +/- drift on log CA price by AIC, using data <= `end` only."""
    y = panel.loc[:end, "ly"].to_numpy()
    rows = []
    for p, q, trend in itertools.product(range(max_pq + 1), range(max_pq + 1), ["n", "t"]):
        try:
            res = ARIMA(y, order=(p, 1, q), trend=trend).fit()
            rows.append({"p": p, "d": 1, "q": q, "trend": trend, "aic": res.aic, "bic": res.bic})
        except Exception:  # noqa: BLE001 - some orders fail to converge; skip them
            continue
    tab = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    return tab


class ArimaModel:
    """ARIMA with order fixed from pre-2022 AIC; parameters re-estimated on refit,
    and between refits the fitted parameters are re-filtered over the newer data."""
    name = "ARIMA"

    def __init__(self, order=(1, 1, 1), trend="n"):
        self.order, self.trend, self.res = order, trend, None
        self.name = f"ARIMA{order}" + ("+drift" if trend == "t" else "")

    def fit(self, panel, T):
        self.res = ARIMA(_ly_upto(panel, T), order=self.order, trend=self.trend).fit()

    def predict(self, panel, T):
        y = _ly_upto(panel, T)
        res = self.res.apply(y) if len(y) != self.res.nobs else self.res
        return res.forecast(len(H)) - y[-1]


class EtsDamped:
    """Holt's additive damped-trend exponential smoothing on log price."""
    name = "ETS damped"

    def __init__(self):
        self.params = None

    def fit(self, panel, T):
        r = ExponentialSmoothing(_ly_upto(panel, T), trend="add", damped_trend=True,
                                 initialization_method="estimated").fit()
        self.params = {k: r.params[k] for k in ("smoothing_level", "smoothing_trend", "damping_trend")}

    def predict(self, panel, T):
        y = _ly_upto(panel, T)
        r = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                 initialization_method="estimated").fit(**self.params, optimized=False)
        return r.forecast(len(H)) - y[-1]


class DirectRegression:
    """One regression per horizon: tgt_h ~ features at origin. Trains only on rows whose
    target date t + h <= T (no target leakage)."""

    def __init__(self, kind: str = "ridge", seasonal: bool = True):
        self.kind, self.seasonal = kind, seasonal
        self.name = {"ridge": "Ridge", "gbm": "GBM"}[kind] + (" + seasonal" if seasonal else "")
        self.models: dict[int, object] = {}
        self.train_info: dict[int, tuple] = {}

    def features(self, h):
        return FEATURES_BASE + [f"seas_{h}"] if self.seasonal else list(FEATURES_BASE)

    def _make(self):
        if self.kind == "ridge":
            return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        return GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2,
                                         subsample=0.8, min_samples_leaf=20, random_state=C.SEED)

    def training_rows(self, panel, T, h):
        cols = self.features(h)
        hist = panel.loc[:T]
        n = len(hist) - h                       # rows t with t + h <= T
        X = hist[cols].iloc[:n]
        y = hist[f"tgt_{h}"].iloc[:n]
        ok = X.notna().all(axis=1) & y.notna()
        return X[ok], y[ok]

    def fit(self, panel, T):
        T = pd.Timestamp(T)
        for h in H:
            X, y = self.training_rows(panel, T, h)
            last_target = panel.index[panel.index.get_loc(X.index[-1]) + h]
            assert last_target <= T, "training target after forecast origin"
            self.models[h] = self._make().fit(X.to_numpy(), y.to_numpy())
            self.train_info[h] = (X.index[0], X.index[-1], last_target, len(X))

    def predict(self, panel, T, override: dict | None = None):
        out = []
        for h in H:
            x = panel.loc[[T], self.features(h)].copy()
            if override:
                for k, v in override.items():
                    if k in x:
                        x[k] = x[k] + v
            out.append(self.models[h].predict(x.to_numpy())[0])
        return np.array(out)

    def coefficients(self) -> pd.DataFrame:
        """Standardised Ridge coefficients or GBM impurity importances, features x horizon."""
        cols = {}
        for h, m in self.models.items():
            names = self.features(h)
            names = [n if not n.startswith("seas_") else "seas" for n in names]
            vals = m[-1].coef_ if self.kind == "ridge" else m.feature_importances_
            cols[h] = pd.Series(vals, index=names)
        return pd.DataFrame(cols)


ENSEMBLES = {
    "Ens: RW + seasonal mean": ["Random walk", "Seasonal mean (10y)"],
    "Ens: statistical": ["ARIMA", "ETS damped", "Seasonal mean (10y)"],
    "Ens: ML (+seasonal)": ["Ridge + seasonal", "GBM + seasonal"],
    "Ens: all models": ["Random walk", "Seasonal naive", "Seasonal mean (10y)", "ARIMA",
                        "ETS damped", "Ridge", "Ridge + seasonal", "GBM", "GBM + seasonal"],
}


def make_models(arima_order=(1, 1, 1), arima_trend="n") -> list:
    arima = ArimaModel(arima_order, arima_trend)
    arima.name = "ARIMA"  # short display name; the order is reported separately
    return [RandomWalk(), SeasonalNaive(), SeasonalMean(), arima, EtsDamped(),
            DirectRegression("ridge", False), DirectRegression("ridge", True),
            DirectRegression("gbm", False), DirectRegression("gbm", True)]
