# Forecasting California Retail Gasoline Prices 1–12 Weeks Ahead

UCLA Anderson MSBA · Time Series Analysis & Forecasting · Group Project 2026

Can we forecast California's average retail gasoline price 1–12 weeks ahead, and what drives it? We
use eight public EIA weekly and daily series: CA and US retail gasoline, Brent and WTI crude, and West
Coast (PADD 5) and US gasoline inventories and refinery utilisation. These are built into a weekly
Monday panel where every input is lagged to its real publication date, and an automated test confirms
there is no look-ahead. We compare 13 models in a weekly rolling-origin backtest from January 2022:
naive benchmarks, a 10-year seasonal mean, ARIMA, damped ETS, direct Ridge and gradient-boosting
regressions, and four ensembles. The top model is chosen on 2022–24 targets only and then judged on an
untouched 2025–26 holdout that includes the 2026 crude shock (Brent ~$62 → $138 → $70 → $135).

## Key results

**Top model: `Ens: ML (+seasonal)`**, an equal-weight ensemble of Ridge and gradient boosting, both
with a seasonal-expectation feature. It ranked first on 2022–24 validation (mean RMSE 0.81× random walk).

**Holdout 2025–26** (by target date, never used for model choice):

| Horizon | MAE, top model | MAE, random walk | RMSE vs random walk | Direction correct | Diebold–Mariano p |
|---|---|---|---|---|---|
| 1 week | **$0.056** | $0.078 | 0.81 (−19%) | 79% | 0.005 |
| 4 weeks | **$0.206** | $0.234 | 0.89 (−11%) | 71% | 0.24 |
| 8 weeks | **$0.303** | $0.362 | 0.83 (−17%) | 74% | 0.17 |
| 12 weeks | **$0.342** | $0.458 | 0.82 (−18%) | 78% | 0.15 |

**12-week forecast** from $6.23/gal on October 5, 2026 (empirical bands from backtest errors):

| Target date | Forecast | 80% interval | 90% interval |
|---|---|---|---|
| Oct 12, 2026 | $6.24 | $6.13–6.34 | $6.09–6.42 |
| Nov 2, 2026 | $6.17 | $5.79–6.65 | $5.72–6.92 |
| Nov 30, 2026 | $5.95 | $5.44–6.90 | $5.20–7.10 |
| Dec 28, 2026 | $5.58 | $4.99–6.58 | $4.87–7.06 |

Seasonality is the strongest single signal. After it come CA price momentum, lagged Brent pass-through
(a 10% Brent move adds ~24¢/gal by week 8), margin compression and mean reversion in the CA–US premium.
The full write-up is in [`reports/report.md`](reports/report.md), and all weekly forecasts are in
[`outputs/forecasts/final_forecast_top_model.csv`](outputs/forecasts/final_forecast_top_model.csv).

## Repository layout

```
ca-gas-forecast/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # the 8 original EIA CSVs, untouched (+ README: source, units, ranges)
│   └── processed/            # weekly_panel.csv, model_features.csv (+ README: column dictionary)
├── src/                      # all model and pipeline code
│   ├── config.py             # paths, dates, horizons, release lags, seeds
│   ├── data_prep.py          # load raw → weekly Monday panel with release lags → weekly_panel.csv
│   ├── features.py           # engineered features, look-ahead test → model_features.csv
│   ├── models.py             # naive, seasonal naive/mean, ARIMA, ETS, Ridge, GBM, ensembles
│   ├── backtest.py           # rolling-origin evaluation + metrics, DM tests, regimes
│   ├── forecast.py           # final 12-week forecast, intervals, coverage, Brent scenario
│   ├── visualize.py          # all figures
│   ├── llm_kit.py            # Option A: LLM-as-forecaster comparison kit
│   └── run_all.py            # runs the whole pipeline in order
├── notebooks/
│   └── project.ipynb         # narrative + charts, imports from src/
├── app/
│   └── app.py                # Streamlit dashboard (Option B)
├── outputs/
│   ├── figures/              # PNG charts (fig01–fig11)
│   ├── tables/               # accuracy, model ranking, holdout, coverage, summary.json
│   ├── forecasts/            # backtest forecasts, final 12-week forecast, Brent scenarios
│   ├── models/               # driver_model.joblib (fitted Ridge for the app's scenario tab)
│   └── llm/                  # prompts / responses for Option A
└── reports/
    ├── report.md
    └── slides_outline.md
```

## Setup

Tested with Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

All commands run from the repo root.

```bash
python src/run_all.py              # full pipeline, ~5 min: data/processed/ + every table, forecast, figure
python src/run_all.py --use-cache  # reuse outputs/forecasts/backtest_forecasts.csv (seconds)

jupyter nbconvert --to notebook --execute --inplace notebooks/project.ipynb   # or open it in Jupyter / VS Code
```

`run_all.py` rebuilds everything from `data/raw/`, including `data/processed/` and
`outputs/models/driver_model.joblib`. To rebuild only the processed data, run
`python -m src.data_prep` or `python -m src.features`. Seeds are fixed (42), so reruns reproduce the
committed outputs.

## Launch the app

```bash
streamlit run app/app.py           # opens http://localhost:8501
```

The app reads the committed `data/processed/` and `outputs/`, so it works without rerunning the pipeline.
Its tabs are:
- **History & drivers:** a driver selector, with each driver in its own panel and units.
- **Backtest:** model and horizon pickers, showing forecasts against actuals.
- **Accuracy:** the metrics table by period or regime.
- **12-week forecast:** the forecast with 80% and 90% bands.
- **Brent scenario:** a "what if Brent moves ±X%" slider driven by the Ridge model.

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), use `app/app.py` as the entry point.

## LLM comparison (Option A)

```bash
python src/llm_kit.py prepare               # writes 20 prompts to outputs/llm/prompts/
                                            #   and calls the Anthropic API if ANTHROPIC_API_KEY is set
python src/llm_kit.py prepare --mask-dates  # variant: years relabelled Y0, Y-1, ... (harder to "remember")
python src/llm_kit.py score                 # scores saved answers vs random walk, seasonal mean, ARIMA, top model
```

Without an API key, paste each prompt into a **fresh** chat with memory, web search and file tools off.
Save each reply as `outputs/llm/responses/<origin>.json` (plain text is fine), then run `score`. Seven
origins fall after the model's knowledge cutoff (`LLM_KNOWLEDGE_CUTOFF` in `src/config.py`) and are scored
separately, because earlier origins may be contaminated by training data.

## Leakage controls

- **Release lags:** crude enters as the prior Friday's close. EIA weekly stocks and utilisation for the
  week ending Friday F enter only from Monday F + 10.
- **Look-ahead test:** `features.check_no_lookahead` rebuilds the panel from data truncated at 26 origins
  and requires identical features. A leak-injection test, which uses a 3-day instead of a 10-day lag,
  confirms the check catches leaks.
- **Model selection:** the ARIMA order is chosen by AIC on data to 2021-12-31. The top model is chosen on
  2022–24 targets only, and 2025–26 is a holdout.
- **Training targets:** direct regressions assert that every training target date is ≤ the forecast origin.

## AI tools used

| Tool | Role |
|---|---|
| **Claude Code (Claude Opus 5.5)** | Pair programmer. It profiled the data and implemented the pipeline in `src/` (panel and look-ahead test, models, backtest, DM tests, figures), the Streamlit app and the LLM kit. It also ran everything end to end and restructured the repository. Its output was checked by automated tests (look-ahead, leak injection, target-date assertions) and by the team. |
| **Streamlit** (`app/app.py`, Option B) | Interactive dashboard over the results |
| **LLM forecast kit** (`src/llm_kit.py`, Option A) | Benchmarks an LLM *as a forecaster* against our models, with leakage and contamination controls |

Data: U.S. Energy Information Administration (EIA). See [`data/raw/README.md`](data/raw/README.md).
