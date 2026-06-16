# Current Modeling Insights

## Main Signal
- Lagged `total_cases` is the strongest signal by far.
- Autocorrelation: `sj` lag 1 = `0.965`, `iq` lag 1 = `0.747`.
- Use lagged target features as a core model input, not an afterthought.

## City Differences
- `sj`: strongest weather signal is delayed temperature/humidity, mostly lag `7-11`.
- `sj` top feature: `station_avg_temp_c` lag `10`, `r = 0.369`.
- `iq`: weather signal is weaker and shorter-lagged, mostly lag `0-3`.
- `iq` top feature: `reanalysis_specific_humidity_g_per_kg` lag `0`, `r = 0.235`.

## Feature Engineering Direction
- Add seasonality: `weekofyear_sin`, `weekofyear_cos`.
- Add selected weather lags instead of every lag.
- Add delayed rolling weather means for exposure windows.
- Add target lags: `1, 2, 3, 4, 8, 12`.
- Add rolling target history: past `2`, `4`, `8` week mean/max.

## Modeling Caution
- These are only linear correlations, not proof of causal value.
- Validate features with strict forward time splits by city.
- Next useful test: baseline vs lagged target vs lagged weather vs both.
