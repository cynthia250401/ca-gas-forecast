# Forecasting California Retail Gasoline Prices (1–12 weeks ahead)

UCLA Anderson MSBA · Time Series Analysis & Forecasting · Group Project 2026

**Question:** Can we forecast California retail gasoline prices 1–12 weeks ahead, and what drives them?
The story covers the 2026 crude shock (Brent rose from about $62 to more than $125, then to about $135) and California's growing premium over the US average.

| Deliverable | File |
|---|---|
| Written report (executive summary first) | `report.md` |
| Slide outline (≤10 slides, with speaker notes) | `slides_outline.md` |
| End-to-end notebook | `project.ipynb` |
| Pipeline script | `analysis.py` + `src/` |
| Interactive app (Option B) | `app.py` |
| LLM forecast comparison kit (Option A) | `llm_kit.py` |
| Results | `outputs/*.csv`, `outputs/figures/*.png`, `outputs/summary.json` |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python analysis.py                 # full pipeline, ~4 min: tables, forecasts, figures in outputs/
python analysis.py --use-cache     # reuse outputs/backtest_forecasts.csv (seconds)
jupyter nbconvert --to notebook --execute --inplace project.ipynb   # or open it in Jupyter / VS Code
```

## Launch the app

```bash
streamlit run app.py               # opens http://localhost:8501
```

Run `analysis.py` first, because the app reads `outputs/`. The tabs are:
- **History & drivers:** a driver selector, with each driver drawn in its own panel and units.
- **Backtest:** model and horizon pickers, showing forecasts against actuals.
- **Accuracy:** the metrics table by period or regime.
- **12-week forecast:** the forecast with 80% and 90% bands.
- **Brent scenario:** a "what if Brent moves ±X%" slider driven by the Ridge model.

To share it, push the repo to GitHub and deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), with `app.py` as the entry point. Commit `outputs/` so the app has data.

## LLM comparison (Option A)

```bash
python llm_kit.py prepare          # writes 20 prompts to outputs/llm/prompts/
                                   #   and calls the Anthropic API if ANTHROPIC_API_KEY is set
python llm_kit.py prepare --mask-dates   # variant: years relabelled Y0, Y-1, ... (harder to "remember")
python llm_kit.py score            # scores saved answers vs random walk, seasonal mean, ARIMA, top model
```

**Without an API key:**
1. Paste each prompt into a **fresh** chat with memory, web search and file tools off.
2. Save the reply as `outputs/llm/responses/<origin>.json`. Plain text is fine, because the kit parses the first `{...}` block.
3. Run `score`.

Seven of the origins fall after the model's knowledge cutoff (`LLM_KNOWLEDGE_CUTOFF` in `src/config.py`). Scores are reported separately for them, because earlier origins may be contaminated by training data.

## Project layout

```
src/config.py     dates, horizons, publication lags, seeds
src/data.py       loading, profiling, weekly panel with release lags, look-ahead check
src/models.py     random walk, seasonal naive and mean, ARIMA, ETS, direct Ridge and GBM, ensembles
src/backtest.py   rolling-origin backtest, final forecast, empirical intervals
src/evaluate.py   MAE/RMSE/MAPE/directional accuracy, RelRMSE, Diebold-Mariano, regimes
src/scenario.py   Brent ±X% scenario
src/plots.py      all figures (one consistent style)
```

## Data

There are eight U.S. EIA series (columns: date, value):
- **Retail gasoline:** CA and US, weekly on Mondays.
- **Crude:** Brent and WTI, daily, with blank holiday rows.
- **Gasoline inventories:** PADD5 and US, weekly on Fridays.
- **Refinery utilisation:** PADD5 and US, weekly on Fridays; PADD5 starts 2010-06.

## Reproducibility and leakage controls

- **Seeds:** fixed (`SEED = 42`). GBM uses `random_state`.
- **Publication lags:**
  - Crude for a Monday is the prior Friday's close.
  - EIA weekly data for the week ending Friday F is used only from Monday F+10.
  - `data.check_no_lookahead` rebuilds the panel from data truncated at 26 origins and requires identical features.
  - A leak-injection test confirms that the check catches a 3-day lag instead of the true 10-day lag.
- **Model selection:**
  - The ARIMA order is chosen by AIC on data up to 2021-12-31.
  - The top model is chosen on 2022–24 targets only.
  - 2025–26 targets are a holdout.
- **Training targets:** direct regressions assert that every training target date is on or before the forecast origin.
