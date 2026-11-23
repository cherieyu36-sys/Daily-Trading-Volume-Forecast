# Adaptive Intraday Volume Forecasting and Optimal Trade Execution

## Research question
Can information available at 11:00 New York time improve forecasts of the remaining intraday volume profile and reduce modelled execution costs relative to TWAP and historical VWAP?

## Quick start
Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python research.py --synthetic
python research.py --ticker AAPL --order 10000
python -m pytest -q
```
Other US equity tickers (e.g. MSFT, SPY) can be substituted. The Yahoo Finance adapter uses yfinance and one month of 15-minute regular-session bars. Yahoo limits intraday history to the recent past (typically 60 days); availability can vary. See https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html. Do not redistribute market data without checking applicable terms.

## Methodology
- Use six completed morning bars (09:30–11:00 ET), then forecast the twenty remaining bars.
- Exclude incomplete sessions and half-days. Fit on up to 30 previous complete sessions; evaluate sequentially after a ten-day warm-up.
- Compare historical mean volume, morning-scaled historical volume and multi-output ridge regression on log volume. Features include morning volume, return, realised volatility and recent volume share.
- Compare TWAP, three forecast-weighted VWAP schedules, and a risk-aware quadratic execution optimisation. The scaled forecast changes total volume but not relative weights, so its VWAP schedule matches historical VWAP.
- Score volume MAE and L1 error in volume shares. For sell orders, compare implementation shortfall against the 11:00 arrival-price proxy, remaining-session VWAP slippage, modelled costs and maximum participation.

## Outputs
`outputs/` contains per-day strategy metrics, execution schedules, forecast errors, a strategy summary and a volume-profile chart. Schedule files include actual volume for retrospective analysis only; actual volume is never a forecasting input. Quantities are continuous shares in this first research prototype.

## Assumptions and limitations
This is an offline research prototype, not a live execution system. Bar-close prices proxy for benchmark and fill prices. The assumed half-spread (1 bp) and linear participation impact (20 bps times participation) are illustrative, not fitted market-impact estimates. All orders are assumed filled, without liquidity caps or queue modelling; large participation rates flag unrealistic schedules. The risk coefficient is dimensionless and not calibrated to market volatility. No claims about actual trading profitability are warranted. Ridge log predictions are not bias-corrected. The short Yahoo sample limits statistical inference; use the assignment's simulated data for formal evaluation and add a data adapter as needed.

## Next steps
1. Add a simulated-data CSV adapter once the assignment schema is available.
2. Test rolling window length and risk parameters on a separate chronological validation sample.
3. Add participation constraints, integer-share rounding and incomplete-fill handling.
4. Compare gradient boosting with ridge on a longer dataset; add uncertainty estimates and cost sensitivity tests.
5. Document out-of-sample results and prepare the final research report.

## Repository layout
- `research.py`: data ingestion, feature engineering, forecasting, optimisation, backtest and CLI.
- `tests/test_research.py`: leakage, allocation and integration checks.
- `requirements.txt`: dependencies.
- `example_results/`: synthetic demonstration only.
