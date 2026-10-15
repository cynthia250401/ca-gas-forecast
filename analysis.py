"""End-to-end pipeline: profile -> panel -> look-ahead checks -> backtest -> evaluation -> forecast -> figures.

    python analysis.py              # full run (~4 min on a laptop)
    python analysis.py --use-cache  # reuse outputs/backtest_forecasts.csv if present
"""
from __future__ import annotations

import argparse
import warnings
import json

import joblib
import numpy as np
import pandas as pd

from src import backtest as B
from src import config as C
from src import data as D
from src import evaluate as E
from src import models as M
from src import plots as P
from src import scenario as S

warnings.filterwarnings("ignore", message="obj.round has no effect")


def lookahead_mutation_test(raw, panel) -> bool:
    """Sanity-check the checker: if EIA data were aligned with a 3-day lag instead of the
    true 10-day release lag, the look-ahead check must fail."""
    saved = C.EIA_WEEKLY_LAG_DAYS
    try:
        C.EIA_WEEKLY_LAG_DAYS = 3
        leaky = D.build_panel(raw)
        try:
            D.check_no_lookahead(leaky, raw, n_checks=10)
            return False
        except AssertionError:
            return True
    finally:
        C.EIA_WEEKLY_LAG_DAYS = saved


def run_all(use_cache: bool = False, verbose: bool = True) -> dict:
    C.set_seeds()
    C.OUT_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    # 1. data
    raw = D.load_raw()
    prof = D.profile(raw)
    prof.to_csv(C.OUT_DIR / "data_profile.csv", index=False)
    panel = D.build_panel(raw)
    panel.to_csv(C.OUT_DIR / "panel.csv")
    log(f"Panel: {panel.shape}, {panel.index[0]:%Y-%m-%d} .. {panel.index[-1]:%Y-%m-%d}")

    # 2. look-ahead checks
    la = D.check_no_lookahead(panel, raw)
    la.to_csv(C.OUT_DIR / "lookahead_check.csv", index=False)
    mutation_caught = lookahead_mutation_test(raw, panel)
    log(f"Look-ahead check: {la.ok.sum()}/{len(la)} origins identical; leak-injection test caught: {mutation_caught}")

    # 3. ARIMA order on pre-2022 data
    aic = M.select_arima_order(panel)
    aic.to_csv(C.OUT_DIR / "arima_order_selection.csv", index=False)
    best = aic.iloc[0]
    order, trend = (int(best.p), 1, int(best.q)), best.trend
    log(f"ARIMA order by AIC (data <= {C.ARIMA_SELECT_END}): {order}, trend={trend}")
    models = M.make_models(order, trend)

    # 4. backtest
    cache = C.OUT_DIR / "backtest_forecasts.csv"
    if use_cache and cache.exists():
        bt = pd.read_csv(cache, parse_dates=["origin", "target_date"])
    else:
        bt = B.run_backtest(panel, models)
        bt.to_csv(cache, index=False)
    log(f"Backtest: {bt.origin.nunique()} origins x {bt.model.nunique()} models x 12 horizons")

    # 5. evaluation: choose on 2022-24 targets, report 2025-26 separately
    t_valid = E.accuracy_table(bt, E.by_year(*C.VALID_YEARS))
    t_hold = E.accuracy_table(bt, E.by_year(*C.HOLDOUT_YEARS))
    t_all = E.accuracy_table(bt)
    top, ranking = E.select_top_model(t_valid)
    hold_rank = (t_hold.groupby("model").agg(holdout_mean_RelRMSE=("RelRMSE", "mean")))
    ranking = ranking.join(hold_rank)
    reg = E.regime_table(bt)
    for name, t in [("validation_2022_2024", t_valid), ("holdout_2025_2026", t_hold), ("all_2022_2026", t_all),
                    ("by_regime", reg)]:
        t.round(5).to_csv(C.OUT_DIR / f"accuracy_{name}.csv", index=False)
    ranking.round(4).to_csv(C.OUT_DIR / "model_ranking.csv")
    log(f"Top model on validation: {top}")

    # 6. final forecast + empirical intervals
    final, fitted = B.final_forecast(panel, models)
    final.to_csv(C.OUT_DIR / "final_forecast_all_models.csv", index=False)
    q = B.interval_quantiles(bt, top)
    q.to_csv(C.OUT_DIR / "interval_error_quantiles.csv")
    fan = B.add_intervals(final, q, top)
    fan.round(4).to_csv(C.OUT_DIR / "final_forecast_top_model.csv", index=False)

    # interval coverage check on the holdout (bands built from validation-period errors only)
    q_val = B.interval_quantiles(bt[bt.target_date.dt.year <= C.VALID_YEARS[1]], top)
    h_rows = bt[(bt.model == top) & bt.actual_price.notna() & (bt.target_date.dt.year >= C.HOLDOUT_YEARS[0])]
    err = np.log(h_rows.actual_price) - np.log(h_rows.fc_price)
    cov = pd.DataFrame({
        "cover80": (err >= h_rows.h.map(q_val.q10)) & (err <= h_rows.h.map(q_val.q90)),
        "cover90": (err >= h_rows.h.map(q_val.q05)) & (err <= h_rows.h.map(q_val.q95)),
        "h": h_rows.h}).groupby("h").mean().mul(100).round(1)
    cov.to_csv(C.OUT_DIR / "interval_coverage_holdout.csv")

    # 7. drivers
    ridge, gbm = fitted["Ridge + seasonal"], fitted["GBM + seasonal"]
    ridge_coef, gbm_imp = ridge.coefficients(), gbm.coefficients()
    ridge_coef.to_csv(C.OUT_DIR / "ridge_coefficients.csv")
    gbm_imp.to_csv(C.OUT_DIR / "gbm_importance.csv")
    joblib.dump({"ridge": ridge, "top": top, "arima_order": order}, C.OUT_DIR / "driver_model.joblib")
    scen = pd.concat([S.brent_scenario(panel, ridge, x).assign(brent_move_pct=x) for x in (-20, -10, 10, 20)])
    scen.round(4).to_csv(C.OUT_DIR / "brent_scenarios.csv", index=False)

    # 8. figures
    reg_all_h = reg
    figs = [
        P.fig_prices(panel), P.fig_premium(panel), P.fig_seasonal(panel), P.fig_margin(panel),
        P.fig_inventory(panel), P.fig_corr(panel),
        P.fig_error_by_horizon(E.accuracy_table(bt, E.by_year(*C.VALID_YEARS), C.HORIZONS),
                               E.accuracy_table(bt, E.by_year(*C.HOLDOUT_YEARS), C.HORIZONS), top),
        P.fig_backtest(bt, panel, top), P.fig_fan(panel, final, fan, top),
        P.fig_importance(ridge_coef, gbm_imp), P.fig_regimes(reg_all_h, top),
    ]
    log(f"Saved {len(figs)} figures to {C.FIG_DIR}")

    # 9. headline summary
    def pick(t, m, h, col):
        return float(t[(t.model == m) & (t.h == h)][col].iloc[0])

    summary = {
        "latest_date": f"{panel.index[-1]:%Y-%m-%d}",
        "latest_ca_price": round(float(panel.ca.iloc[-1]), 3),
        "latest_us_price": round(float(panel.us.iloc[-1]), 3),
        "latest_brent": round(float(panel.brent.iloc[-1]), 2),
        "arima_order": list(order), "arima_trend": trend,
        "top_model": top,
        "lookahead_checks_passed": int(la.ok.sum()), "lookahead_mutation_caught": mutation_caught,
        "holdout": {h: {k: round(pick(t_hold, top, h, k), 4) for k in ["MAE", "RMSE", "MAPE", "DirAcc", "RelRMSE", "DM_stat", "DM_p"]}
                    | {"RW_MAE": round(pick(t_hold, "Random walk", h, "MAE"), 4),
                       "RW_RMSE": round(pick(t_hold, "Random walk", h, "RMSE"), 4)}
                    for h in C.REPORT_H},
        "validation": {h: {k: round(pick(t_valid, top, h, k), 4) for k in ["MAE", "RMSE", "RelRMSE", "DM_p"]}
                       for h in C.REPORT_H},
        "forecast": fan[["h", "target_date", "fc_price", "lo80", "hi80", "lo90", "hi90"]]
        .assign(target_date=lambda d: d.target_date.dt.strftime("%Y-%m-%d")).round(3).to_dict("records"),
        "holdout_interval_coverage": cov.to_dict(),
    }
    with open(C.OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return {"panel": panel, "bt": bt, "t_valid": t_valid, "t_hold": t_hold, "reg": reg, "ranking": ranking,
            "top": top, "final": final, "fan": fan, "summary": summary, "fitted": fitted, "profile": prof,
            "lookahead": la, "aic": aic, "coverage": cov, "scenarios": scen}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true")
    res = run_all(use_cache=ap.parse_args().use_cache)
    pd.set_option("display.width", 200)
    print("\nModel ranking (validation 2022-24 mean RelRMSE, holdout 2025-26 alongside):")
    print(res["ranking"].round(3))
    print("\n12-week forecast (top model):")
    print(res["fan"][["h", "target_date", "fc_price", "lo80", "hi80", "lo90", "hi90"]].round(3).to_string(index=False))
