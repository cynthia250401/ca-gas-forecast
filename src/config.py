"""Project-wide constants: paths, dates, horizons, seeds."""
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
LLM_DIR = OUT_DIR / "llm"

SEED = 42

# Backtest design
BACKTEST_START = "2022-01-03"      # first forecast origin (Monday)
REFIT_EVERY = 4                    # re-estimate model parameters every 4 weekly origins
HORIZONS = list(range(1, 13))      # all horizons are forecast (needed for fan chart)
REPORT_H = [1, 4, 8, 12]           # horizons reported in tables
VALID_YEARS = (2022, 2024)         # model selection: target dates in 2022-2024
HOLDOUT_YEARS = (2025, 2026)       # untouched evaluation: target dates in 2025-2026
ARIMA_SELECT_END = "2021-12-31"    # ARIMA order chosen by AIC on data up to here only

# Publication lags (days, relative to the Monday price date)
CRUDE_LAG_DAYS = 3     # last close on or before the prior Friday
EIA_WEEKLY_LAG_DAYS = 10  # week ending Friday F is released Wed F+5, usable Monday F+10

REGIMES = {
    "2022 spike": ("2022-01-01", "2022-12-31"),
    "2023-25 normal": ("2023-01-01", "2025-12-31"),
    "2026 shock": ("2026-01-01", "2026-12-31"),
}

# Shaded shock periods for charts
SHOCKS = [
    ("2008 crude spike", "2008-01-01", "2008-09-30"),
    ("2022 Russia-Ukraine", "2022-02-21", "2022-11-30"),
    ("2026 crude shock", "2026-03-01", None),
]


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

# Raw files: name -> (filename, publication lag in days relative to the Monday origin)
RAW_FILES = {
    "ca": ("ca_gas.csv", 0),
    "us": ("us_gas.csv", 0),
    "brent": ("brent.csv", CRUDE_LAG_DAYS),
    "wti": ("wti.csv", CRUDE_LAG_DAYS),
    "stocks_p5": ("gas_stocks_padd5.csv", EIA_WEEKLY_LAG_DAYS),
    "stocks_us": ("gas_stocks_us.csv", EIA_WEEKLY_LAG_DAYS),
    "util_p5": ("refinery_util_padd5.csv", EIA_WEEKLY_LAG_DAYS),
    "util_us": ("refinery_util_us.csv", EIA_WEEKLY_LAG_DAYS),
}

BBL_TO_GAL = 42.0
SEASONAL_YEARS = 10      # seasonal-expectation feature: prior 10 calendar years
SEASONAL_MIN_YEARS = 5   # require at least 5 of them
WEEKS_PER_YEAR = 52.1775

# Option A (LLM) settings
LLM_MODEL = "claude-opus-5-5"
LLM_KNOWLEDGE_CUTOFF = "2026-06-30"  # edit if you use a different model
