"""Option A: an honest LLM-vs-model forecast comparison kit.

For ~20 test origins it writes a prompt containing ONLY data published by that origin
(same release lags as the models) and asks for 1/4/8/12-week CA gasoline forecasts in JSON.

    python src/llm_kit.py prepare [--mask-dates]   # write prompts; calls the API if ANTHROPIC_API_KEY is set
    python src/llm_kit.py call                     # (re)call the API for prompts without a response
    python src/llm_kit.py score                    # score whatever responses exist vs the backtest models

Without an API key, paste each prompt in outputs/llm/prompts/ into a FRESH chat (no memory,
no web search, no file uploads) and save the reply as outputs/llm/responses/<origin>.json
(raw text is fine; the first {...} block is parsed).

Contamination risk: a model trained on data past an origin may "remember" the outcome.
Origins after LLM_KNOWLEDGE_CUTOFF are flagged and scored separately; --mask-dates replaces
years with relative labels (Y0, Y-1, ...) to make memorised prices harder to look up.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python src/llm_kit.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data_prep as D
from src import features as F

PROMPT_DIR = C.LLM_DIR / "prompts"
RESP_DIR = C.LLM_DIR / "responses"
REPORT_H = [1, 4, 8, 12]
HISTORY_WEEKS = 156
COMPARE = ["Random walk", "Seasonal mean (10y)", "ARIMA"]  # plus the top model from summary.json

SCHEMA = {
    "type": "object",
    "properties": {
        "h1": {"type": "number"}, "h4": {"type": "number"},
        "h8": {"type": "number"}, "h12": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["h1", "h4", "h8", "h12", "rationale"],
    "additionalProperties": False,
}


def choose_origins(panel: pd.DataFrame) -> pd.DataFrame:
    """13 origins before the LLM's knowledge cutoff (every 6 weeks) + 7 after it (every 2 weeks)."""
    cutoff = pd.Timestamp(C.LLM_KNOWLEDGE_CUTOFF)
    dates = panel.index
    last_scorable = dates[-2]                       # needs at least the 1-week actual
    pre_end = dates[dates <= cutoff - pd.Timedelta(weeks=4)][-1]
    pre = [pre_end - pd.Timedelta(weeks=6 * k) for k in range(13)][::-1]
    post_start = dates[dates > cutoff][0]
    post = [d for d in (post_start + pd.Timedelta(weeks=2 * k) for k in range(7)) if d <= last_scorable]
    o = pd.DataFrame({"origin": pre + post})
    o["post_cutoff"] = o.origin > cutoff
    return o


def _year_label(d: pd.Timestamp, T: pd.Timestamp, mask: bool) -> str:
    if not mask:
        return f"{d:%Y-%m-%d}"
    return f"Y{d.year - T.year:+d}-{d:%m-%d}".replace("Y+0", "Y0")


def build_prompt(raw, T: pd.Timestamp, mask_dates: bool = False) -> str:
    """Prompt from data truncated at T (publication lags applied) - nothing after T can leak."""
    p = F.build_panel(raw, origin=T)
    assert p.index[-1] == T
    h = p.iloc[-HISTORY_WEEKS:]
    lines = ["week_monday,ca_retail_usd_gal,us_retail_usd_gal,brent_prior_fri_usd_bbl,"
             "padd5_gas_stocks_kbbl_lagged,padd5_refinery_util_pct_lagged"]
    for d, r in h.iterrows():
        lines.append(f"{_year_label(d, T, mask_dates)},{r.ca:.3f},{r.us:.3f},{r.brent:.2f},"
                     f"{r.stocks_p5:.0f},{r.util_p5:.1f}")
    targets = ", ".join(f"h{k} = {_year_label(T + pd.Timedelta(weeks=k), T, mask_dates)}" for k in REPORT_H)
    origin_lab = _year_label(T, T, mask_dates)
    return f"""You are a commodity-price forecaster. Forecast the California average retail regular
gasoline price (EIA weekly series, $/gallon, Monday survey).

Forecast origin: {origin_lab}. You may use ONLY the data below, which is exactly what was published
by that date. Do not use any knowledge of prices or events after the origin date.
{"Years are relabelled relative to the origin year (Y0 = origin year, Y-1 = the year before)." if mask_dates else ""}

Data notes:
- ca/us retail: weekly Monday prices.
- brent_prior_fri: Brent crude close on the Friday before each Monday ($/barrel; /42 = $/gallon).
- padd5 stocks (thousand barrels) and refinery utilisation (%) are for the week ending ~10 days
  before each Monday, because EIA releases them with a lag.

Last {len(h)} weeks of data (CSV):
{chr(10).join(lines)}

Task: forecast the California retail price on these target Mondays: {targets}.
Return ONLY a JSON object of the form
{{"h1": <price>, "h4": <price>, "h8": <price>, "h12": <price>, "rationale": "<2-3 sentences>"}}
with prices in $/gallon to 3 decimals.
"""


def prepare(mask_dates: bool = False) -> pd.DataFrame:
    raw = D.load_raw()
    panel = F.build_panel(raw)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    RESP_DIR.mkdir(parents=True, exist_ok=True)
    o = choose_origins(panel)
    for T in o.origin:
        (PROMPT_DIR / f"{T:%Y-%m-%d}.txt").write_text(build_prompt(raw, T, mask_dates))
    o["mask_dates"] = mask_dates
    o["model"] = C.LLM_MODEL
    o["knowledge_cutoff"] = C.LLM_KNOWLEDGE_CUTOFF
    o.to_csv(C.LLM_DIR / "manifest.csv", index=False)
    print(f"Wrote {len(o)} prompts to {PROMPT_DIR} ({o.post_cutoff.sum()} after the knowledge cutoff)")
    return o


def call_api(overwrite: bool = False):
    """Send each prompt to Claude with structured JSON output. Fresh request per origin (no memory)."""
    import anthropic

    client = anthropic.Anthropic()
    for path in sorted(PROMPT_DIR.glob("*.txt")):
        out = RESP_DIR / f"{path.stem}.json"
        if out.exists() and not overwrite:
            continue
        try:
            resp = client.beta.messages.create(
                model=C.LLM_MODEL,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": path.read_text()}],
            )
        except anthropic.RateLimitError as e:
            print(f"{path.stem}: rate limited ({e}); rerun `python src/llm_kit.py call` later")
            break
        except anthropic.APIStatusError as e:
            print(f"{path.stem}: API error {e.status_code}: {e.message}")
            continue
        except anthropic.APIConnectionError as e:
            print(f"{path.stem}: connection error: {e}")
            break
        if resp.stop_reason == "refusal":
            print(f"{path.stem}: request declined; skipped")
            continue
        text = next(b.text for b in resp.content if b.type == "text")
        record = json.loads(text) | {"served_by": resp.model, "origin": path.stem}
        out.write_text(json.dumps(record, indent=2))
        print(f"{path.stem}: {record['h1']:.3f} / {record['h4']:.3f} / {record['h8']:.3f} / {record['h12']:.3f}")


def _parse(path) -> dict | None:
    m = re.search(r"\{.*\}", path.read_text(), flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def score() -> pd.DataFrame | None:
    from src import backtest as E

    files = sorted(RESP_DIR.glob("*.json")) + sorted(RESP_DIR.glob("*.txt"))
    if not files:
        print(f"No responses in {RESP_DIR}. Paste LLM answers there, then rerun `python src/llm_kit.py score`.")
        return None
    manifest = pd.read_csv(C.LLM_DIR / "manifest.csv", parse_dates=["origin"])
    rows = []
    for f in files:
        d = _parse(f)
        if d is None:
            print(f"Could not parse {f.name}; skipped")
            continue
        T = pd.Timestamp(f.stem)
        rows += [{"origin": T, "h": h, "fc_price": float(d[f"h{h}"])} for h in REPORT_H if f"h{h}" in d]
    llm = pd.DataFrame(rows).assign(model=f"LLM ({C.LLM_MODEL})")

    bt = pd.read_csv(C.FORECAST_DIR / "backtest_forecasts.csv", parse_dates=["origin", "target_date"])
    top = json.loads((C.TABLE_DIR / "summary.json").read_text())["top_model"]
    keep = bt[bt.model.isin(COMPARE + [top]) & bt.origin.isin(llm.origin) & bt.h.isin(REPORT_H)]
    ref = keep[keep.model == "Random walk"][["origin", "h", "target_date", "origin_price", "actual_price"]]
    llm = llm.merge(ref, on=["origin", "h"], how="inner")
    allf = pd.concat([keep[llm.columns], llm], ignore_index=True)
    allf = allf.merge(manifest[["origin", "post_cutoff"]], on="origin", how="left")
    allf = allf[allf.actual_price.notna()]
    allf.to_csv(C.LLM_DIR / "llm_forecasts_vs_models.csv", index=False)

    parts = []
    for label, sub in [("all origins", allf), ("pre-cutoff (contamination risk)", allf[~allf.post_cutoff]),
                       ("post-cutoff (clean)", allf[allf.post_cutoff])]:
        if sub.empty:
            continue
        t = E.accuracy_table(sub, horizons=REPORT_H)
        t.insert(0, "subset", label)
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    tab.round(4).to_csv(C.LLM_DIR / "llm_scores.csv", index=False)
    print(tab[["subset", "model", "h", "n", "MAE", "RMSE", "MAPE", "DirAcc", "RelRMSE", "DM_p"]]
          .round(3).to_string(index=False))
    from src import visualize as P
    P.fig_llm(tab[tab.subset == "all origins"])
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "call", "score"])
    ap.add_argument("--mask-dates", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    np.random.seed(C.SEED)
    if a.command == "prepare":
        prepare(a.mask_dates)
        if os.environ.get("ANTHROPIC_API_KEY"):
            call_api(a.overwrite)
        else:
            print("ANTHROPIC_API_KEY not set: prompts saved. Paste each into a fresh chat and save replies "
                  f"as {RESP_DIR}/<origin>.json, then run `python src/llm_kit.py score`.")
    elif a.command == "call":
        call_api(a.overwrite)
    else:
        score()
