# Slide outline: 10 slides, 10 minutes

**Title:** *Forecasting California Gas Prices Through the 2026 Crude Shock*

**Timing:** about 1 minute per slide. The live app demo takes about 90 seconds on slide 9. Figures are in `outputs/figures/`.

---

### Slide 1: $6.23 and rising: can we see what's coming?
- **Visual:** a big number, **$6.23/gal** (CA, Oct 5, 2026), beside **$4.35** (US). Subtitle: "+54% since January · Brent $62 → $135".
- **On slide:** research question: *Can we forecast CA gasoline 1–12 weeks ahead, and what drives it?*
- **Speaker notes (~45s):** "California drivers are paying $1.87 a gallon more than the rest of the country, during the biggest crude shock since 2022. Households budgeting fuel, retailers setting prices and a state that now regulates refiner margins all want the same thing: a credible 1–3-month outlook, with honest uncertainty. That is our question today."

### Slide 2: The 2026 shock, in one chart
- **Chart:** `fig01_prices_crude.png`.
- **On slide:** "Brent ×2 in 5 weeks (Mar–Apr) → CA +$1.30" · "Second surge in Sept–Oct".
- **Speaker notes (~60s):**
  - "Blue is California, orange is the US, and green is Brent converted to dollars per gallon, so everything is on one axis. The gap between blue and green is what drivers pay beyond the crude itself."
  - "In March Brent doubled and California went from $4.47 to $5.77 in five weeks. Brent fell back to about $70 by June, then surged again to $135 on October 2."
  - "So crude clearly matters. But notice the blue line moves more than crude alone would suggest."

### Slide 3: California's premium is structural, and its spikes are local
- **Chart:** `fig02_ca_premium.png`.
- **On slide:** "Premium $0.20 (2005) → $1.31 (2025) → $1.87 now" · "Spikes = West Coast refinery outages".
- **Speaker notes (~50s):**
  - "California's own fuel blend, higher taxes and climate-programme costs, and an isolated refining market have pushed the premium up more than five-fold."
  - "The spikes, like autumn 2022 and 2023, are refinery outages, not crude."
  - "Two lessons for modelling: the premium tends to revert to its trend, and today's $1.87 is well above its $1.45 norm."

### Slide 4: Seasonality is the strongest signal
- **Chart:** `fig03_seasonal_profile.png`, plus an inset or callout from `fig04_margin_vs_crude.png` (right panel).
- **On slide:**
  - "Feb–May: about +18% (maintenance + summer blend)".
  - "Oct–Dec: decline (winter blend)".
  - "Crude jumps squeeze the margin first; the pump catches up over weeks".
- **Speaker notes (~60s):**
  - "Every year prices climb from February to May and fall in the autumn. In our first pass, a simple seasonal benchmark beat most of our models, so we turned seasonality into an explicit feature: the average change over the next h weeks at this time of year, over the previous 10 years."
  - "The second pattern is lagged pass-through. When crude jumps, retail margins get squeezed first and recover over the next weeks. So an unusually wide or narrow margin today predicts where prices go next."

### Slide 5: Our approach: 13 models, no peeking at the future
- **Visual:** a simple diagram. Weekly panel (Monday origin) → features → 13 models → rolling backtest. Show the timeline: Validation 2022–24 | Holdout 2025–26.
- **On slide (three columns):**
  - **Data:** 8 EIA series; crude = prior-Friday close; inventories usable only 10 days after the reference week.
  - **Models:** random walk, seasonal naive, seasonal mean, ARIMA, ETS, Ridge, gradient boosting (± seasonal), 4 ensembles.
  - **Validation:** 249 weekly origins, refit every 4 weeks; choose on 2022–24, test on 2025–26; DM tests.
- **Speaker notes (~60s):**
  - "Forecasting backtests are easy to cheat by accident, for example by using inventory data a week before EIA publishes it. We aligned every input to its release date."
  - "We then wrote a test that rebuilds the features using only what was published at 26 different dates. They match exactly, and we checked that the test catches a deliberately injected leak."
  - "We chose the best model using 2022–24 only, and kept 2025–26 as an untouched holdout."

### Slide 6: Results: seasonality plus drivers beats the random walk at every horizon
- **Chart:** `fig07_error_by_horizon.png`.
- **On slide:** holdout table for the top model, *Ens: ML (+seasonal)*:

  | | 1 wk | 4 wk | 8 wk | 12 wk |
  |---|---|---|---|---|
  | MAE ($/gal) | 0.056 | 0.206 | 0.303 | 0.342 |
  | Random walk | 0.078 | 0.234 | 0.362 | 0.458 |
  | RMSE vs RW | −19% | −11% | −17% | −18% |

- **Speaker notes (~70s):**
  - "Below the dashed line means better than 'no change'. ARIMA helps only at 1 week, and ETS is worse than doing nothing beyond a few weeks."
  - "The winner on validation was an equal-weight ensemble of Ridge and gradient boosting, both using the seasonal feature. On the untouched holdout it beat the random walk at every horizon by 11–19%, and called the direction right three times out of four."
  - "Honest caveats: only the 1-week gain is statistically significant on the holdout alone, and the plain 10-year seasonal mean is nearly as good at 8–12 weeks."

### Slide 7: Shocks: where models struggle, and what still works
- **Chart:** `fig08_backtest_top_model.png`, with `fig11_regimes.png` as a small inset or backup.
- **On slide:** "Normal years: −26% RMSE at 12 wk" · "Shocks: errors double; nobody forecasts the onset" · "In 2026 the simple seasonal mean was most robust".
- **Speaker notes (~60s):**
  - "Here are the 4- and 12-week-ahead forecasts against reality. The model anticipates the spring run-ups and autumn declines that the random walk misses."
  - "No model sees a war or an OPEC shock coming: March 2022 and March 2026 are missed at 12 weeks."
  - "In 2026, crude reversed twice, so momentum-based models got whipsawed, and the simple seasonal average was the most robust. Every key model still beat the random walk."

### Slide 8: What drives California prices?
- **Chart:** `fig10_feature_importance.png`.
- **On slide:**
  1. Season (grows with horizon).
  2. Recent CA momentum (short horizons).
  3. Brent 4-week change: a 10% move gives about +24¢ by week 8 (~75% pass-through).
  4. Wide margin or premium → prices fall next.
  5. Inventories and utilisation add little.
- **Speaker notes (~50s):**
  - "Linear and tree models agree. Season dominates long horizons and momentum dominates short ones."
  - "Crude passes through with a lag: about half within a month and three-quarters by two months."
  - "When California's premium or margin is stretched, prices come back down."
  - "Surprisingly, weekly inventory data adds little once you know the premium and margin, which react to refinery problems faster."

### Slide 9: The forecast, and our app (live demo)
- **Chart:** `fig09_forecast_fan.png`, then switch to the **Streamlit app** (`streamlit run app/app.py`).
- **On slide:** "$6.24 → $6.17 (Nov 2) → **$5.58** (Dec 28) · 80%: $4.99–6.58 · 90%: $4.87–7.06".
- **Speaker notes (~90s):**
  - "From $6.23, the model sees prices holding near $6.20 into mid-November, as crude's October spike offsets an over-stretched premium. Then the seasonal autumn decline takes over, toward $5.58 by late December."
  - "The bands come from the model's own historical errors, and they covered about their stated share in 2025–26. They are skewed upward, because historical surprises were mostly upside."
  - **Demo:** pick drivers on the History tab → pick a horizon on Backtest → show the Accuracy table → **slide Brent to +20%**: December becomes about $6.02 instead of $5.62.
- **Backup if the demo fails:** screenshots of the Forecast and Scenario tabs.

### Slide 10: Takeaways and how we used AI
- **On slide (left), takeaways:**
  1. Seasonality is the backbone: about −22% RMSE at 12 weeks from a seasonal average alone.
  2. Drivers add the most at 1–4 weeks; the ensemble beat the random walk at every horizon on unseen data.
  3. During shocks, trust the bands and scenarios, not the point forecast.
- **On slide (right), AI:**
  - **Option B:** Streamlit dashboard.
  - **Option A:** an LLM-as-forecaster kit, with 20 origins, only data available at each date, 7 origins after the model's training cutoff, and the same metrics.
  - **Claude Code:** built and tested the pipeline. We verified it with leakage tests and corrected an AI-written chart claim that the data contradicted.
- **Speaker notes (~45s):** "Three takeaways… [read]. On AI: we used Claude Code as a pair programmer, but we didn't trust it blindly. Every leakage claim is tested in code, and when an AI-drafted chart title claimed tight inventories predict price rises, the data said otherwise, so we changed it. Our LLM kit asks whether an LLM can forecast gas prices, and it controls for the LLM simply remembering them. Thank you."
- **[Team: if LLM results are in, add one line, e.g. "LLM MAE at 4 weeks: $X vs model $0.21 (post-cutoff origins)".]**
