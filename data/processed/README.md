# Processed data

Both files are derived from `../raw/` and are fully reproducible:

```bash
python src/run_all.py          # whole pipeline, writes both files first
python -m src.data_prep        # weekly_panel.csv only (from the repo root)
python -m src.features         # model_features.csv only
```

Both are indexed by `date`, the **Monday** of each EIA California retail-price survey
(1,377 weeks, 2000-05-22 to 2026-10-05). Each row holds only information that was **published by
that Monday**. `features.check_no_lookahead` verifies this on every run by rebuilding the panel from raw
data truncated at 26 origins (results in `outputs/tables/lookahead_check.csv`).

## `weekly_panel.csv`

Built by `src/data_prep.py::align_weekly`. These are the eight raw series on the Monday grid with the
release-lag rules applied: an *as-of* lookup of the latest observation on or before T − lag. There are
no transformations.

| Column | Definition | Units | Lag rule |
|---|---|---|---|
| `date` | Monday price date T (index) | | |
| `ca` | California regular retail gasoline | $/gal | same day |
| `us` | U.S. regular retail gasoline | $/gal | same day |
| `brent` | Brent spot, last close ≤ T − 3 days (prior Friday) | $/bbl | 3 days |
| `wti` | WTI spot, last close ≤ T − 3 days | $/bbl | 3 days |
| `stocks_p5` | PADD 5 gasoline stocks, latest week ending ≤ T − 10 days | thousand bbl | 10 days |
| `stocks_us` | U.S. gasoline stocks, latest week ending ≤ T − 10 days | thousand bbl | 10 days |
| `util_p5` | PADD 5 refinery utilisation, latest week ending ≤ T − 10 days | % | 10 days |
| `util_us` | U.S. refinery utilisation, latest week ending ≤ T − 10 days | % | 10 days |

Early rows are blank where a series had not started yet (for example `util_p5` before mid-2010).

## `model_features.csv`

Built by `src/features.py::build_panel`. This is the table every model in `src/models.py` reads: the
aligned levels above plus engineered features, seasonal expectations and targets (47 columns). The
Streamlit app also reads it for its history charts.

| Column | Definition |
|---|---|
| `ca`, `us`, `brent`, `wti`, `stocks_p5`, `stocks_us`, `util_p5`, `util_us` | As in `weekly_panel.csv` |
| `inv_p5_dev`, `inv_us_dev` | Gasoline stocks, % above or below the average of the same week in the prior 5 years. Computed on the native Friday series, then aligned with the 10-day lag |
| `util_p5_chg_4w` | 4-week change in PADD 5 utilisation (percentage points), computed on the native series, then lagged 10 days |
| `ly` | ln(`ca`) |
| `ca_chg_1w`, `ca_chg_4w` | Change in ln CA price over 1 and 4 weeks |
| `brent_chg_1w`, `brent_chg_4w`, `brent_chg_12w` | Change in ln Brent over 1, 4 and 12 weeks |
| `margin` | `ca` − `brent` / 42, retail minus crude cost per gallon ($/gal) |
| `margin_dev` | `margin` minus its trailing 52-week mean |
| `spread` | `ca` − `us`, California premium over the U.S. ($/gal) |
| `spread_dev` | `spread` minus its trailing 52-week mean |
| `woy_sin`, `woy_cos` | sin and cos of 2π·(day-of-year − 1)/365.25 |
| `seas_1` … `seas_12` | Seasonal expectation for horizon h: mean ln CA change over the next h weeks at this time of year, across the prior 10 calendar years (at least 5 required). It uses only data ≥ 40 weeks old |
| `tgt_1` … `tgt_12` | **Target:** ln CA(T+h) − ln CA(T). Blank where T+h is not yet observed. Regressions train only on rows with T+h ≤ forecast origin |

The model inputs are `FEATURES_BASE` in `src/features.py` (13 columns), plus `seas_h` for the "+ seasonal" variants.
