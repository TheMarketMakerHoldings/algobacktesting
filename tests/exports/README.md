# Sample trade exports

Small, hand-made files that imitate each platform's trade export. They test the website's
importer (`tests/test_site_import.py`). None of them is a real account or a real export.

| File | Modelled on | Source of the column names |
| --- | --- | --- |
| `quantower.csv` | Quantower Trades panel, "Export data" | Column list published by the TradeZella import guide |
| `topstepx.csv` | TopstepX Trades tab, "Export" | Assumed; not confirmed against a real file |
| `ninjatrader.csv` | NinjaTrader 8 Trade Performance, Trades view | Assumed; not confirmed against a real file |
| `thinkorswim.csv` | thinkorswim Account Statement, "Export to File" | Section and column names from journal import guides |

Each test lists the expected per-trade results, worked out by hand from the rows. Replace a
file with a real (anonymised) export when one is available, and keep the expected values in
step.
