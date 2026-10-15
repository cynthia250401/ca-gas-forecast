"""All project figures. One visual system: single y-axis per panel, thin 2px lines, recessive
grid, fixed categorical colour order, direct labels plus a legend for >= 2 series."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config as C
from .features import FEATURE_LABELS, FEATURES_BASE

# ---------------------------------------------------------------- palette (validated default)
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SHADE = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
CONTEXT = "#c3c2b7"   # de-emphasised series
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#184f95", "#3987e5", "#9ec5f4", SHADE, "#f2a7a6", RED, "#a52a2a"])


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 22,
        "axes.labelsize": 10, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    })


set_style()


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def _shade_shocks(ax, start=None, end=None, label=True):
    lo, hi = ax.get_xlim()
    for i, (name, s, e) in enumerate(C.SHOCKS):
        s = pd.Timestamp(s)
        e = pd.Timestamp(e) if e else pd.Timestamp.today()
        if (start and e < pd.Timestamp(start)) or (end and s > pd.Timestamp(end)):
            continue
        ax.axvspan(s, e, color=SHADE, zorder=0, lw=0)
        if label:
            ax.text(max(s, pd.Timestamp(start)) if start else s, 1.0 - 0.045 * (i % 2), " " + name,
                    transform=ax.get_xaxis_transform(), fontsize=8, color=MUTED, va="top", ha="left")
    ax.set_xlim(lo, hi)


def _spread(values, min_gap):
    """Nudge label y-positions apart so end labels never overlap (keeps order)."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def _end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", fontsize=8.5,
                color=INK2, va="center", ha="left")
    ax.plot([x], [y], "o", ms=4, color=color, mec=SURF, mew=1)


# ---------------------------------------------------------------- 1. prices + crude
def fig_prices(p: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    crude = p["brent"] / C.BBL_TO_GAL
    for ax, start in zip(axes, ["2000-01-01", "2025-01-01"]):
        d = p.loc[start:]
        cr = crude.loc[start:]
        ax.plot(d.index, d.ca, color=BLUE, label="California retail gasoline")
        ax.plot(d.index, d.us, color=ORANGE, label="US average retail gasoline")
        ax.plot(cr.index, cr, color=AQUA, label="Brent crude ÷ 42 (crude cost per gallon)")
        ax.set_ylabel("$ per gallon")
        ax.set_ylim(0, 6.9)
        _shade_shocks(ax, start=start)
        last = d.index[-1]
        if start != "2000-01-01":
            for s, c, dy in [(d.ca, BLUE, 0), (d.us, ORANGE, 0), (cr, AQUA, 0)]:
                _end_label(ax, last, s.iloc[-1], f"${s.iloc[-1]:.2f}", c, dy)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[0].set_title("California pays a widening premium — and crude drives the swings")
    _subtitle(axes[0], "Weekly retail prices and Brent crude expressed in $/gal, 2000–2026 (shaded: crude shocks)")
    axes[1].set_title("Zoom: the 2026 crude shock")
    _subtitle(axes[1], "Jan 2025 – latest week")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    return _save(fig, "fig01_prices_crude")


# ---------------------------------------------------------------- 2. CA premium
def fig_premium(p: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    prem = p["spread"]
    roll = prem.rolling(52).mean()
    ax.plot(prem.index, prem, color=BLUE, lw=1.2, label="CA minus US, weekly")
    ax.plot(roll.index, roll, color=ORANGE, lw=2, label="52-week average")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("$ per gallon")
    _shade_shocks(ax, start="2000-01-01")
    for yr in [2005, 2015, 2025]:
        v = roll.loc[str(yr)].mean()
        ax.annotate(f"{yr} avg ${v:.2f}", (pd.Timestamp(f"{yr}-07-01"), v), xytext=(0, -34),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    _end_label(ax, prem.index[-1], prem.iloc[-1], f"${prem.iloc[-1]:.2f} latest", BLUE)
    ax.set_title("California's premium over the US average has grown more than five-fold since 2005")
    _subtitle(ax, "CA retail minus US retail gasoline price; premium spikes when West Coast refining is disrupted")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    return _save(fig, "fig02_ca_premium")


# ---------------------------------------------------------------- 3. seasonal profile
def fig_seasonal(p: pd.DataFrame, end="2021-12-31"):
    d = p.loc[:end].copy()
    d["wk"] = np.minimum(d.index.isocalendar().week.astype(int), 52)
    prof = d.groupby("wk")[["ca_chg_1w"]].mean() * 100
    us = np.log(d["us"]).diff()
    prof["us"] = us.groupby(d["wk"]).mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    colors = [BLUE if v >= 0 else "#86b6ef" for v in prof.ca_chg_1w]
    ax.bar(prof.index, prof.ca_chg_1w, color=colors, width=0.8, edgecolor=SURF, linewidth=0.5)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Average weekly change (%)")
    ax.set_title("Prices rise Feb–May and fall Oct–Dec")
    _subtitle(ax, f"California average weekly log change by week of year, {d.index[0].year}–{d.index[-1].year}")
    ax = axes[1]
    for col, c, lab in [("ca_chg_1w", BLUE, "California"), ("us", ORANGE, "US average")]:
        cum = prof[col].cumsum()
        ax.plot(cum.index, cum, color=c, lw=2, label=lab)
        _end_label(ax, cum.index[-1], cum.iloc[-1], lab, c)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Cumulative seasonal change (%)")
    ax.set_title("Cumulative seasonal path: California's spring run-up is larger")
    _subtitle(ax, "Running sum of the average weekly changes (trend included)")
    ax.legend(loc="upper left")
    for a in axes:
        a.set_xticks([1, 9, 18, 27, 36, 44, 52])
        a.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"])
    fig.tight_layout()
    return _save(fig, "fig03_seasonal_profile")


# ---------------------------------------------------------------- 4. margin vs crude
def fig_margin(p: pd.DataFrame):
    d = p.loc["2005":].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.plot(d.index, d.margin, color=BLUE, lw=1.4, label="CA retail − Brent/42")
    ax.plot(d.index, d.margin.rolling(52).mean(), color=ORANGE, lw=2, label="52-week average")
    ax.set_ylabel("$ per gallon")
    _shade_shocks(ax, start="2005-01-01")
    ax.set_title("Retail-minus-crude margin: squeezed when crude spikes, rebuilt after")
    _subtitle(ax, "Gap between California pump price and crude cost per gallon (taxes, refining, distribution, retail)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    ax = axes[1]
    x = 100 * d.brent_chg_4w
    y = d.margin.diff(4)
    groups = [("2005–2021", d.index.year <= 2021, CONTEXT, 10),
              ("2022–2025", (d.index.year >= 2022) & (d.index.year <= 2025), BLUE, 18),
              ("2026", d.index.year == 2026, ORANGE, 26)]
    for lab, m, c, s in groups:
        ax.scatter(x[m], y[m], s=s, color=c, alpha=0.75 if c != CONTEXT else 0.5,
                   edgecolor=SURF, linewidth=0.5, label=lab)
    ok = x.notna() & y.notna()
    b = np.polyfit(x[ok], y[ok], 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK2, lw=1.2, ls="--")
    ax.text(0.98, 0.04, f"slope: {100 * b[0]:.1f}¢/gal per 1% crude move",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Brent 4-week change (%)")
    ax.set_ylabel("Margin 4-week change ($/gal)")
    ax.set_title("Crude jumps first compress the margin")
    _subtitle(ax, "Each dot is one week; retail adjusts with a lag")
    ax.legend(loc="upper right", title=None, markerscale=1.2)
    fig.tight_layout()
    return _save(fig, "fig04_margin_vs_crude")


# ---------------------------------------------------------------- 5. inventories vs price change
def fig_inventory(p: pd.DataFrame):
    d = p.loc["2005":"2025", ["inv_p5_dev", "tgt_4"]].dropna()
    d["fwd"] = 100 * d.tgt_4
    d["bin"] = pd.qcut(d.inv_p5_dev, 10)
    b = d.groupby("bin", observed=True).agg(x=("inv_p5_dev", "mean"), y=("fwd", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(d.inv_p5_dev, d.fwd, s=9, color=CONTEXT, alpha=0.6, edgecolor="none", label="Weekly observations")
    ax.plot(b.x, b.y, "-o", color=BLUE, lw=2, ms=6, mec=SURF, mew=1, label="Decile average")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    corr = d[["inv_p5_dev", "fwd"]].corr().iloc[0, 1]
    ax.text(0.98, 0.04, f"correlation = {corr:.2f}", transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
    ax.set_xlabel("PADD5 gasoline stocks vs same-week 5-year average (%), as published at origin")
    ax.set_ylabel("CA price change over next 4 weeks (%)")
    ax.set_title("West Coast inventory surprises carry only a weak signal for next-month prices")
    _subtitle(ax, "2005–2025; inventories lagged to their release date. Very tight stocks (left) were followed by declines, i.e. mean reversion")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig05_inventory_vs_price")


# ---------------------------------------------------------------- 6. correlation heatmap
def fig_corr(p: pd.DataFrame):
    cols = FEATURES_BASE + ["seas_4", "tgt_4"]
    d = p.loc[:"2021-12-31", cols].dropna()
    corr = d.corr()
    labels = [FEATURE_LABELS.get(c, c) for c in cols[:-2]] + ["Seasonal expectation (4w)", "TARGET: CA chg next 4w"]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(cols)), labels, rotation=55, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cols)), labels, fontsize=8.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=SURF if abs(v) > 0.6 else INK2)
    ax.add_patch(plt.Rectangle((-0.5, len(cols) - 1.5), len(cols), 1, fill=False, ec=INK, lw=1.2))
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.outline.set_visible(False)
    cb.set_label("Pearson correlation")
    ax.set_title("Feature correlations (training era 2010–2021)")
    _subtitle(ax, "Bottom row: correlation with the realised 4-week-ahead CA price change")
    fig.tight_layout()
    return _save(fig, "fig06_feature_correlation")


# ---------------------------------------------------------------- 7. error by horizon


def _highlights(top):
    base = {top: BLUE, "Seasonal mean (10y)": ORANGE, "ARIMA": AQUA, "Ridge + seasonal": VIOLET}
    return {k: v for k, v in base.items()}


def fig_error_by_horizon(tab_valid: pd.DataFrame, tab_hold: pd.DataFrame, top: str):
    hl = _highlights(top)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, tab, title in [(axes[0], tab_valid, "Validation: targets in 2022–2024 (model chosen here)"),
                           (axes[1], tab_hold, "Holdout: targets in 2025–2026 (untouched)")]:
        for m, g in tab.groupby("model"):
            g = g.sort_values("h")
            if m in hl or m == "Random walk":
                continue
            ax.plot(g.h, g.RelRMSE, color=CONTEXT, lw=1, alpha=0.9)
        ax.axhline(1, color=INK2, lw=1.2, ls="--", label="Random walk (benchmark)")
        names, ys = ["Random walk"], [1.0]
        for m, c in hl.items():
            g = tab[tab.model == m].sort_values("h")
            ax.plot(g.h, g.RelRMSE, color=c, lw=2.2 if m == top else 1.8, label=m, marker="o", ms=3.5)
            names.append(m)
            ys.append(g.RelRMSE.iloc[-1])
        for m, y in zip(names, _spread(ys, 0.032)):
            ax.text(12.3, y, m, fontsize=8.5, color=INK2, va="center")
        ax.set_xlim(0.5, 15.5)
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
        ax.set_xlabel("Forecast horizon (weeks)")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("RMSE relative to random walk (<1 = better)")
    axes[0].plot([], [], color=CONTEXT, lw=1, label="Other models")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Error by horizon: every model in grey, key models highlighted", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    return _save(fig, "fig07_error_by_horizon")


# ---------------------------------------------------------------- 8. backtest vs actual
def fig_backtest(bt: pd.DataFrame, p: pd.DataFrame, top: str, horizons=(4, 12)):
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 3.4 * len(horizons)), sharex=True)
    for ax, h in zip(axes, horizons):
        d = bt[(bt.model == top) & (bt.h == h) & bt.actual_price.notna()].sort_values("target_date")
        rw = bt[(bt.model == "Random walk") & (bt.h == h) & bt.actual_price.notna()].sort_values("target_date")
        act = p.loc["2021-10-01":, "ca"]
        ax.plot(act.index, act, color=INK2, lw=1.8, label="Actual CA price")
        ax.plot(rw.target_date, rw.fc_price, color=CONTEXT, lw=1.2, label="Random walk forecast")
        ax.plot(d.target_date, d.fc_price, color=BLUE, lw=1.6, label=f"{top} forecast")
        ax.axvline(pd.Timestamp("2025-01-01"), color=INK2, lw=0.8, ls=":")
        ax.text(pd.Timestamp("2024-12-20"), 0.97, "< validation", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=8.5, color=INK2)
        ax.text(pd.Timestamp("2025-01-12"), 0.97, "holdout >", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=8.5, color=INK2)
        ax.set_ylabel("$ per gallon")
        _shade_shocks(ax, start="2021-10-01", label=False)
        mae = (d.actual_price - d.fc_price).abs().mean()
        mae_rw = (rw.actual_price - rw.fc_price).abs().mean()
        ax.set_title(f"{h}-week-ahead forecasts, plotted at the target date", fontsize=11)
        _subtitle(ax, f"MAE {top}: \\${mae:.3f}/gal vs random walk \\${mae_rw:.3f}/gal (2022–2026)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Backtest: {top} vs actual (rolling origin, refit every 4 weeks)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    return _save(fig, "fig08_backtest_top_model")


# ---------------------------------------------------------------- 9. fan chart
def fig_fan(p: pd.DataFrame, final: pd.DataFrame, fan: pd.DataFrame, top: str):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    hist = p.loc["2025-07-01":, "ca"]
    T = p.index[-1]
    ax.plot(hist.index, hist, color=INK2, lw=2, label="Actual CA price")
    x = [T] + list(fan.target_date)
    y0 = hist.iloc[-1]
    ax.fill_between(x, [y0] + list(fan.lo90), [y0] + list(fan.hi90), color="#cde2fb", lw=0, label="90% interval")
    ax.fill_between(x, [y0] + list(fan.lo80), [y0] + list(fan.hi80), color="#86b6ef", lw=0, label="80% interval")
    labelled = {"Random walk", "Seasonal mean (10y)", "ARIMA"}
    lab_pos = []
    for m, g in final.groupby("model"):
        if m == top:
            continue
        g = g.sort_values("h")
        ax.plot([T] + list(g.target_date), [y0] + list(g.fc_price), color=MUTED if m in labelled else CONTEXT,
                lw=1.1, alpha=0.9)
        if m in labelled:
            lab_pos.append((m, g.target_date.iloc[-1], g.fc_price.iloc[-1]))
    for (m, xd, _), yy in zip(lab_pos, _spread([v for *_, v in lab_pos], 0.11)):
        ax.text(xd + pd.Timedelta(days=3), yy, m, fontsize=8, color=MUTED, va="center")
    ax.plot(x, [y0] + list(fan.fc_price), color=BLUE, lw=2.4, label=f"{top} (top model)")
    _end_label(ax, fan.target_date.iloc[-1], fan.fc_price.iloc[-1],
               f"${fan.fc_price.iloc[-1]:.2f} ({fan.lo80.iloc[-1]:.2f}–{fan.hi80.iloc[-1]:.2f}, 80%)", BLUE, dy=-14)
    ax.annotate(f"${y0:.2f} on {T:%b %d}", (T, y0), xytext=(-6, 8), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK2)
    ax.axvline(T, color=AXIS, lw=0.8)
    ax.set_ylabel("$ per gallon")
    ax.set_xlim(hist.index[0], fan.target_date.iloc[-1] + pd.Timedelta(days=60))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_title("12-week California gasoline forecast with empirical 80% / 90% intervals")
    _subtitle(ax, "Bands from quantiles of the top model's backtest errors; grey lines: every other model's path")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig09_forecast_fan")


# ---------------------------------------------------------------- 10. coefficients / importance
def fig_importance(ridge_coef: pd.DataFrame, gbm_imp: pd.DataFrame):
    order = ridge_coef.abs().mean(axis=1).sort_values(ascending=False).index
    rc = ridge_coef.loc[order]
    labels = [FEATURE_LABELS.get(i, i) for i in order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    lim = np.abs(rc.values).max()
    im = ax.imshow(rc.values * 100, cmap=DIVERGING, vmin=-lim * 100, vmax=lim * 100, aspect="auto")
    ax.grid(False)
    ax.set_yticks(range(len(order)), labels, fontsize=9)
    ax.set_xticks(range(rc.shape[1]), [f"{h}" for h in rc.columns])
    ax.set_xlabel("Horizon (weeks)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.set_label("Effect of +1 s.d. on forecast change (pp)")
    ax.set_title("Ridge + seasonal: standardised coefficients")
    _subtitle(ax, "Fitted on all data to the latest week; red = pushes price up")
    ax = axes[1]
    imp = gbm_imp.mean(axis=1).sort_values()
    ax.barh([FEATURE_LABELS.get(i, i) for i in imp.index], imp.values * 100, color=BLUE, height=0.65,
            edgecolor=SURF)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of impurity reduction (%), averaged over horizons")
    ax.set_title("GBM + seasonal: feature importance")
    fig.tight_layout()
    return _save(fig, "fig10_feature_importance")


# ---------------------------------------------------------------- 11. regimes
def fig_regimes(reg: pd.DataFrame, top: str, h: int = 4):
    models = [top, "Seasonal mean (10y)", "ARIMA", "Ridge + seasonal", "ETS damped"]
    cols = [BLUE, ORANGE, AQUA, VIOLET, YELLOW]
    d = reg[(reg.h == h) & reg.model.isin(models)]
    regimes = list(C.REGIMES)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    w = 0.16
    for i, (m, c) in enumerate(zip(models, cols)):
        vals = [d[(d.regime == r) & (d.model == m)].RelRMSE.squeeze() for r in regimes]
        xs = np.arange(len(regimes)) + (i - 2) * w
        ax.bar(xs, vals, width=w - 0.02, color=c, label=m, edgecolor=SURF)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.axhline(1, color=INK2, ls="--", lw=1)
    ax.text(-0.5, 1.01, "Random walk = 1", ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(regimes)), regimes)
    ax.set_ylabel("RMSE relative to random walk")
    ax.set_title(f"Accuracy by regime ({h}-week horizon): drivers win in normal years, the seasonal mean in the 2026 shock")
    _subtitle(ax, "Regimes defined by target date")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "fig11_regimes")


# ---------------------------------------------------------------- 12. LLM comparison
def fig_llm(scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    order = scores.groupby("model").MAE.mean().sort_values().index
    hs = sorted(scores.h.unique())
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    w = 0.8 / len(order)
    for i, (m, c) in enumerate(zip(order, cols + [MAGENTA, GREEN])):
        g = scores[scores.model == m].set_index("h").reindex(hs)
        ax.bar(np.arange(len(hs)) + (i - (len(order) - 1) / 2) * w, g.MAE, width=w - 0.02, color=c, label=m,
               edgecolor=SURF)
    ax.set_xticks(range(len(hs)), [f"h={h}" for h in hs])
    ax.set_ylabel("MAE ($ per gallon)")
    ax.set_title("LLM forecasts vs statistical models on the same origins")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "fig12_llm_comparison")
