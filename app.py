"""California gasoline forecasting dashboard (Option B).

    streamlit run app.py

Reads the outputs written by `python analysis.py`; the Brent scenario re-uses the fitted
Ridge driver model (outputs/driver_model.joblib).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import config as C
from src import evaluate as E
from src.scenario import brent_scenario

st.set_page_config(page_title="CA Gasoline Forecast", page_icon="⛽", layout="wide")

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK2, MUTED, CONTEXT, GRID = "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
MODEL_COLORS = {  # fixed per model so a filter never repaints a survivor
    "Random walk": INK2, "Seasonal naive": MAGENTA, "Seasonal mean (10y)": ORANGE, "ARIMA": AQUA,
    "ETS damped": YELLOW, "Ridge": "#86b6ef", "Ridge + seasonal": VIOLET, "GBM": "#f2a7a6",
    "GBM + seasonal": RED, "Ens: RW + seasonal mean": "#9085e9", "Ens: statistical": GREEN,
    "Ens: ML (+seasonal)": BLUE, "Ens: all models": "#199e70",
}


def style(fig, height=420, ytitle=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False, linecolor=CONTEXT)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle)
    return fig


@st.cache_data
def load():
    o = C.OUT_DIR
    panel = pd.read_csv(o / "panel.csv", index_col=0, parse_dates=True)
    bt = pd.read_csv(o / "backtest_forecasts.csv", parse_dates=["origin", "target_date"])
    final = pd.read_csv(o / "final_forecast_all_models.csv", parse_dates=["origin", "target_date"])
    fan = pd.read_csv(o / "final_forecast_top_model.csv", parse_dates=["origin", "target_date"])
    summary = json.loads((o / "summary.json").read_text())
    return panel, bt, final, fan, summary


@st.cache_resource
def load_driver():
    return joblib.load(C.OUT_DIR / "driver_model.joblib")


try:
    panel, bt, final, fan, summary = load()
except FileNotFoundError:
    st.error("Outputs not found. Run `python analysis.py` first, then reload this page.")
    st.stop()

TOP = summary["top_model"]
MODELS = sorted(bt.model.unique(), key=lambda m: (m != TOP, m))
T = panel.index[-1]

st.title("Forecasting California gasoline prices")
st.caption(f"EIA weekly data through {T:%B %d, %Y}. Top model (chosen on 2022–24 validation): **{TOP}**. "
           "All forecasts use only information published by each forecast date.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("CA regular, latest week", f"${panel.ca.iloc[-1]:.2f}",
          f"{100 * (panel.ca.iloc[-1] / panel.ca.iloc[-2] - 1):+.1f}% w/w")
c2.metric("CA premium over US", f"${panel.spread.iloc[-1]:.2f}")
c3.metric("Brent (prior Friday)", f"${panel.brent.iloc[-1]:.0f}/bbl",
          f"{100 * (np.exp(panel.brent_chg_12w.iloc[-1]) - 1):+.0f}% over 12w")
c4.metric("12-week forecast", f"${fan.fc_price.iloc[-1]:.2f}",
          f"80%: ${fan.lo80.iloc[-1]:.2f}–{fan.hi80.iloc[-1]:.2f}", delta_color="off")

tab_hist, tab_bt, tab_acc, tab_fc, tab_scen, tab_about = st.tabs(
    ["History & drivers", "Backtest", "Accuracy", "12-week forecast", "Brent scenario", "Method"])

# ------------------------------------------------------------------ history
DRIVERS = {
    "Retail prices (CA, US) & crude per gallon": ("$/gal", None),
    "CA premium over US": ("$/gal", "spread"),
    "Retail-minus-crude margin (CA − Brent/42)": ("$/gal", "margin"),
    "Brent crude": ("$/bbl", "brent"),
    "PADD5 gasoline stocks vs 5-yr avg": ("%", "inv_p5_dev"),
    "US gasoline stocks vs 5-yr avg": ("%", "inv_us_dev"),
    "PADD5 refinery utilisation": ("%", "util_p5"),
    "US refinery utilisation": ("%", "util_us"),
}
with tab_hist:
    a, b = st.columns([2, 1])
    picks = a.multiselect("Drivers (each gets its own panel and unit)", list(DRIVERS),
                          default=list(DRIVERS)[:2] + ["PADD5 gasoline stocks vs 5-yr avg"])
    start = b.slider("From year", 2000, T.year, 2015)
    d = panel.loc[str(start):]
    if picks:
        fig = make_subplots(rows=len(picks), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=[f"{p} ({DRIVERS[p][0]})" for p in picks])
        for i, p in enumerate(picks, 1):
            col = DRIVERS[p][1]
            if col is None:
                for s, c, n in [(d.ca, BLUE, "California"), (d.us, ORANGE, "US average"),
                                (d.brent / 42, AQUA, "Brent ÷ 42")]:
                    fig.add_trace(go.Scatter(x=d.index, y=s, name=n, line=dict(color=c, width=2)), i, 1)
            else:
                fig.add_trace(go.Scatter(x=d.index, y=d[col], name=p, line=dict(color=BLUE, width=2),
                                         showlegend=False), i, 1)
            for name, s, e in C.SHOCKS:
                if pd.Timestamp(e or T) >= d.index[0]:
                    fig.add_vrect(x0=max(pd.Timestamp(s), d.index[0]), x1=e or T, fillcolor="#f0efec",
                                  line_width=0, layer="below", row=i, col=1)
        st.plotly_chart(style(fig, height=260 * len(picks)), width="stretch")
        st.caption("Shaded: 2008 crude spike, 2022 Russia–Ukraine, 2026 crude shock. Inventories and "
                   "utilisation are shown as of their release (10-day lag).")

# ------------------------------------------------------------------ backtest
with tab_bt:
    a, b = st.columns([2, 1])
    sel = a.multiselect("Models", MODELS, default=[TOP, "Random walk", "Seasonal mean (10y)"])
    h = b.select_slider("Horizon (weeks ahead)", options=list(C.HORIZONS), value=4)
    act = panel.loc["2021-10-01":, "ca"]
    fig = go.Figure(go.Scatter(x=act.index, y=act, name="Actual", line=dict(color="#0b0b0b", width=2.5)))
    for m in sel:
        g = bt[(bt.model == m) & (bt.h == h) & bt.actual_price.notna()].sort_values("target_date")
        fig.add_trace(go.Scatter(x=g.target_date, y=g.fc_price, name=m,
                                 line=dict(color=MODEL_COLORS.get(m, MUTED), width=2 if m == TOP else 1.4)))
    fig.add_vline(x=pd.Timestamp("2025-01-01"), line=dict(color=INK2, dash="dot", width=1))
    fig.add_annotation(x=pd.Timestamp("2025-01-01"), y=1.02, yref="paper", text="validation | holdout",
                       showarrow=False, font=dict(color=INK2, size=11))
    st.plotly_chart(style(fig, 460, "$ per gallon"), width="stretch")
    st.caption(f"Each point is the {h}-week-ahead forecast made {h} weeks before the date shown. "
               "Rolling origin every week since Jan 2022; parameters re-estimated every 4 weeks.")

# ------------------------------------------------------------------ accuracy
with tab_acc:
    a, b = st.columns(2)
    period = a.radio("Evaluation period (by target date)",
                     ["Validation 2022–24", "Holdout 2025–26", "All 2022–26", "2022 spike",
                      "2023-25 normal", "2026 shock"], horizontal=True)
    metric = b.radio("Sort by", ["RelRMSE", "MAE", "RMSE", "MAPE", "DirAcc"], horizontal=True)
    masks = {"Validation 2022–24": E.by_year(2022, 2024), "Holdout 2025–26": E.by_year(2025, 2026),
             "All 2022–26": None} | {k: E.by_dates(*v) for k, v in C.REGIMES.items()}
    hs = st.multiselect("Horizons", list(C.HORIZONS), default=C.REPORT_H)
    if hs:
        tab = E.accuracy_table(bt, masks[period], horizons=hs)
        hpick = st.radio("Show horizon", hs, horizontal=True, index=min(1, len(hs) - 1))
        t = tab[tab.h == hpick].drop(columns="h").set_index("model")
        t = t.sort_values(metric, ascending=metric != "DirAcc")
        st.dataframe(t.style.format({"n": "{:.0f}", "MAE": "${:.3f}", "RMSE": "${:.3f}", "MAPE": "{:.2f}%",
                                     "DirAcc": "{:.1f}%", "RelRMSE": "{:.3f}", "DM_stat": "{:.2f}",
                                     "DM_p": "{:.3f}"}, na_rep="—")
                     .background_gradient(subset=["RelRMSE"], cmap="Blues_r"), width="stretch")
        st.caption("RelRMSE < 1 beats the random walk. DM = Diebold–Mariano test vs random walk "
                   "(squared error, HLN correction); negative statistic = better than random walk.")
        w = E.wide(tab).round(3)
        st.markdown("**RMSE relative to random walk, all selected horizons**")
        st.dataframe(w, width="stretch")

# ------------------------------------------------------------------ forecast
with tab_fc:
    show_all = st.toggle("Show every model's path", value=True)
    hist = panel.loc["2025-06-01":, "ca"]
    x = [T] + list(fan.target_date)
    y0 = hist.iloc[-1]
    fig = go.Figure(go.Scatter(x=hist.index, y=hist, name="Actual", line=dict(color="#0b0b0b", width=2.5)))
    fig.add_trace(go.Scatter(x=x + x[::-1], y=[y0] + list(fan.hi90) + list(fan.lo90[::-1]) + [y0],
                             fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="90% interval"))
    fig.add_trace(go.Scatter(x=x + x[::-1], y=[y0] + list(fan.hi80) + list(fan.lo80[::-1]) + [y0],
                             fill="toself", fillcolor="rgba(42,120,214,0.30)", line=dict(width=0), name="80% interval"))
    if show_all:
        for m, g in final.groupby("model"):
            if m != TOP:
                fig.add_trace(go.Scatter(x=[T] + list(g.target_date), y=[y0] + list(g.fc_price), name=m,
                                         line=dict(color=MODEL_COLORS.get(m, MUTED), width=1.2), opacity=0.8))
    fig.add_trace(go.Scatter(x=x, y=[y0] + list(fan.fc_price), name=f"{TOP} (top)", line=dict(color=BLUE, width=3.5)))
    st.plotly_chart(style(fig, 480, "$ per gallon"), width="stretch")
    show = fan[["h", "target_date", "fc_price", "lo80", "hi80", "lo90", "hi90"]].copy()
    show["target_date"] = show.target_date.dt.strftime("%Y-%m-%d")
    st.dataframe(show.set_index("h").style.format("${:.2f}", subset=["fc_price", "lo80", "hi80", "lo90", "hi90"]),
                 width="stretch")
    st.caption("Intervals are empirical quantiles of the top model's backtest errors at each horizon "
               "(log scale). On the 2025–26 holdout, bands built from 2022–24 errors covered "
               "roughly their nominal share of outcomes.")

# ------------------------------------------------------------------ scenario
with tab_scen:
    st.markdown("**What if Brent moves ±X% this week?** The Ridge + seasonal driver model is re-run with "
                "all Brent change features shifted by ln(1+X) and the retail-minus-crude margin reduced by "
                "Brent × X / 42 (pump prices have not adjusted yet).")
    pct = st.slider("Brent move (%)", -40, 40, 10, step=5)
    drv = load_driver()["ridge"]
    sc = brent_scenario(panel, drv, pct)
    fig = go.Figure(go.Scatter(x=hist.index, y=hist, name="Actual", line=dict(color="#0b0b0b", width=2.5)))
    fig.add_trace(go.Scatter(x=[T] + list(sc.target_date), y=[y0] + list(sc.baseline), name="Ridge baseline",
                             line=dict(color=MUTED, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[T] + list(sc.target_date), y=[y0] + list(sc.scenario),
                             name=f"Brent {pct:+d}%", line=dict(color=RED if pct > 0 else BLUE, width=3)))
    st.plotly_chart(style(fig, 440, "$ per gallon"), width="stretch")
    a, b, c = st.columns(3)
    a.metric("Brent after move", f"${panel.brent.iloc[-1] * (1 + pct / 100):.0f}/bbl")
    b.metric("Effect at 4 weeks", f"{100 * sc.difference.iloc[3]:+.0f}¢/gal")
    c.metric("Effect at 12 weeks", f"{100 * sc.difference.iloc[11]:+.0f}¢/gal",
             f"full pass-through would be {100 * sc.full_passthrough_diff.iloc[0]:+.0f}¢", delta_color="off")
    st.caption("A linear model fitted on 2010–2026 history: treat large moves (>±30%) as extrapolation. "
               "Pass-through below 100% reflects the margin cushion and lagged retail adjustment.")

# ------------------------------------------------------------------ method
with tab_about:
    st.markdown(f"""
**Question.** Can we forecast California retail gasoline 1–12 weeks ahead, and what drives it?

**Data.** EIA weekly CA and US retail gasoline (Mondays), daily Brent and WTI, weekly PADD5 and US gasoline
inventories and refinery utilisation (Fridays).

**No look-ahead.** Crude for Monday *T* is the last close on or before the prior Friday. EIA weekly data for
the week ending Friday *F* is used only from Monday *F*+10 (it is released the following Wednesday).
An automated check rebuilds the panel from data truncated at {summary['lookahead_checks_passed']} origins
and confirms identical features.

**Models.** Random walk, seasonal naive, 10-year seasonal mean, ARIMA{tuple(summary['arima_order'])} (order by AIC
on pre-2022 data), damped-trend ETS, direct multi-horizon Ridge and gradient boosting (with and without the
seasonal-expectation feature), and four equal-weight ensembles.

**Validation.** Weekly rolling origins from Jan 2022, expanding window, refit every 4 weeks. Model chosen on
2022–24 targets; 2025–26 is an untouched holdout.

**AI use.** Claude Code helped write the pipeline, tests and this app; `llm_kit.py` benchmarks an LLM's own
forecasts against these models with leakage controls (Option A).
""")
