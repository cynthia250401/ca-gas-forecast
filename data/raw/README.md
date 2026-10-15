# Raw data

Eight public series from the **U.S. Energy Information Administration (EIA)**, downloaded as CSVs
and kept exactly as downloaded. Do not edit these files; every derived dataset is rebuilt from them by
`python src/run_all.py` (see `../processed/README.md`).

Each file has two columns: a date (`date` or `period`, ISO `YYYY-MM-DD`) and `value`.

| File | Series | Units | Frequency | Date range | Obs | Usable at Monday T |
|---|---|---|---|---|---|---|
| `ca_gas.csv` | California regular gasoline, retail price (all formulations) | $/gallon | Weekly, Monday | 2000-05-22 to 2026-10-05 | 1,377 | T (the **forecast target**) |
| `us_gas.csv` | U.S. regular gasoline, retail price (all formulations) | $/gallon | Weekly, Monday | 2000-01-03 to 2026-10-05 | 1,397 | T |
| `brent.csv` | Europe Brent crude oil, spot price FOB | $/barrel | Daily, business days | 2000-01-04 to 2026-10-06 | 6,481 (+500 blank holiday rows) | prior Friday close (T − 3 days) |
| `wti.csv` | WTI Cushing, OK crude oil, spot price FOB | $/barrel | Daily, business days | 2000-01-04 to 2026-10-06 | 6,332 (+649 blank rows) | prior Friday close (T − 3 days) |
| `gas_stocks_padd5.csv` | West Coast (PADD 5) total motor gasoline ending stocks | thousand barrels | Weekly, Friday (week ending) | 1990-01-05 to 2026-10-02 | 1,918 | week ending ≤ T − 10 days |
| `gas_stocks_us.csv` | U.S. total motor gasoline ending stocks | thousand barrels | Weekly, Friday (week ending) | 1990-01-05 to 2026-10-02 | 1,918 | week ending ≤ T − 10 days |
| `refinery_util_padd5.csv` | West Coast (PADD 5) percent utilization of refinery operable capacity | % | Weekly, Friday (week ending) | 2010-06-04 to 2026-10-02 | 853 | week ending ≤ T − 10 days |
| `refinery_util_us.csv` | U.S. percent utilization of refinery operable capacity | % | Weekly, Friday (week ending) | 1990-11-02 to 2026-10-02 | 1,875 | week ending ≤ T − 10 days |

The observation counts exclude blank rows. A full profile with min, max and last value is in
`outputs/tables/data_profile.csv`, regenerated on every run.

**Release lags.** Crude prices for Monday T are the last close on or before the prior Friday. EIA weekly
stocks and utilisation for the week ending Friday F are published the following Wednesday, so they are
used only from Monday F + 10. These lags are set in `src/config.py` (`CRUDE_LAG_DAYS`, `EIA_WEEKLY_LAG_DAYS`).

**Source pages.** EIA *Weekly Retail Gasoline and Diesel Prices*, *Spot Prices for Crude Oil and Petroleum
Products*, and the *Weekly Petroleum Status Report* (stocks, refinery utilisation), all at eia.gov.
The CSVs do not record their EIA series IDs. They most likely correspond to `EMM_EPMR_PTE_SCA_DPG`,
`EMM_EPMR_PTE_NUS_DPG`, `RBRTE`, `RWTC`, `WGTSTP51`, `WGTSTUS1`, `WPULEP52` and `WPULEUS3` (in table order),
but these IDs are inferred, not recorded.

**Notes.**
- Crude blanks are market holidays and are dropped on load.
- WTI includes the April 2020 negative print (−$36.98). It is aligned into the panel but not used by the models.
- PADD 5 refinery utilisation starts in June 2010, which sets the start of the regression training sample.
