# Current Modeling Insights

## Forecast-Relevant Signal
- `total_cases` is only the dependent variable / model target, never an input signal.
- Exclude lagged, rolling, or autocorrelated `total_cases` from analysis and modeling.
- Main usable signals are seasonality plus lagged/rolling weather, humidity, temperature, NDVI, and precipitation.

## City Differences
- `sj`: strongest weather signal is delayed temperature/humidity, mostly lag `7-11`.
- `sj` top feature: `station_avg_temp_c` lag `10`, `r = 0.369`.
- `iq`: weather signal is weaker and shorter-lagged, mostly lag `0-3`.
- `iq` top feature: `reanalysis_specific_humidity_g_per_kg` lag `0`, `r = 0.235`.

## Feature Engineering Direction
- Add seasonality: `weekofyear_sin`, `weekofyear_cos`.
- Add selected weather lags instead of every lag.
- Add cumulative rolling weather means over the last `1..14` available weeks.
- Keep feature engineering restricted to exogenous predictors available for both train and test rows.

## Rolling Feature Update
- Rolling windows now mean cumulative current/past feature means: week `t` over the last `1..14` weeks.
- `sj`: broad station-temperature means help; best rolling weather signal is `station_min_temp_c` over `14` weeks, `r = 0.395`.
- `iq`: shorter humidity/dew/min-temp windows dominate; best rolling feature is `reanalysis_min_air_temp_k` over `6` weeks, `r = 0.262`.
- NDVI/precipitation rolling signals are weaker under this corrected cumulative-window definition.

## Modeling Caution
- These are only linear correlations, not proof of causal value.
- Rolling windows can mostly duplicate seasonality and lag effects; validate before trusting them.
- Validate features with strict forward time splits by city.
- Next useful test: baseline vs lagged weather vs rolling weather vs combined exogenous feature set.
