# Individual Lag/Rolling Feature Selection

This is an analysis-only output for a later modelling variant.
`total_cases` is used only as the target for correlations, never as an input feature.

## Selection Rule

- Default absolute-correlation threshold is 0.150; iq uses 0.150, sj uses 0.200.
- For each city, feature, and transform type, keep up to 3 lags or 3 rolling windows within 95% of that feature's best absolute correlation.
- Lag 0 and rolling window 1 are de-duplicated into the raw feature in the final plan.

## Selected Feature Plan

### iq

- Selected generated features: 46
- raw: 4
  - reanalysis_dew_point_temp_k: raw, abs corr 0.229
  - reanalysis_min_air_temp_k: raw, abs corr 0.212
  - reanalysis_specific_humidity_g_per_kg: raw, abs corr 0.235
  - station_min_temp_c: raw, abs corr 0.207

- lag: 9
  - ndvi_sw: lags 11, best abs corr 0.155
  - reanalysis_air_temp_k: lags 7, best abs corr 0.151
  - reanalysis_dew_point_temp_k: lags 1, best abs corr 0.221
  - reanalysis_min_air_temp_k: lags 2, best abs corr 0.205
  - reanalysis_specific_humidity_g_per_kg: lags 1, best abs corr 0.226
  - station_avg_temp_c: lags 4, 5, best abs corr 0.156
  - station_max_temp_c: lags 6, best abs corr 0.160
  - station_min_temp_c: lags 1, best abs corr 0.209

- rolling_mean: 33
  - ndvi_nw: rolling windows 14, best abs corr 0.161
  - ndvi_sw: rolling windows 13, 14, best abs corr 0.162
  - precipitation_amt_mm: rolling windows 6, 7, best abs corr 0.161
  - reanalysis_air_temp_k: rolling windows 12, 13, 14, best abs corr 0.198
  - reanalysis_avg_temp_k: rolling windows 12, 13, 14, best abs corr 0.193
  - reanalysis_dew_point_temp_k: rolling windows 3, 4, 6, best abs corr 0.255
  - reanalysis_min_air_temp_k: rolling windows 6, 7, 8, best abs corr 0.262
  - reanalysis_sat_precip_amt_mm: rolling windows 6, 7, best abs corr 0.161
  - reanalysis_specific_humidity_g_per_kg: rolling windows 3, 4, 6, best abs corr 0.259
  - station_avg_temp_c: rolling windows 8, 10, 11, best abs corr 0.225
  - station_max_temp_c: rolling windows 10, 11, 12, best abs corr 0.188
  - station_min_temp_c: rolling windows 3, 4, 5, best abs corr 0.258
  - station_precip_mm: rolling windows 6, 7, best abs corr 0.157

### sj

- Selected generated features: 59
- lag: 27
  - reanalysis_air_temp_k: lags 7, 8, 9, best abs corr 0.298
  - reanalysis_avg_temp_k: lags 7, 8, 9, best abs corr 0.295
  - reanalysis_dew_point_temp_k: lags 7, 8, 9, best abs corr 0.302
  - reanalysis_max_air_temp_k: lags 6, 7, 8, best abs corr 0.296
  - reanalysis_min_air_temp_k: lags 7, 8, 9, best abs corr 0.303
  - reanalysis_specific_humidity_g_per_kg: lags 7, 8, 9, best abs corr 0.302
  - station_avg_temp_c: lags 9, 10, 11, best abs corr 0.369
  - station_max_temp_c: lags 10, 11, 12, best abs corr 0.316
  - station_min_temp_c: lags 9, 10, 11, best abs corr 0.362

- rolling_mean: 32
  - reanalysis_air_temp_k: rolling windows 12, 13, 14, best abs corr 0.309
  - reanalysis_avg_temp_k: rolling windows 12, 13, 14, best abs corr 0.307
  - reanalysis_dew_point_temp_k: rolling windows 12, 13, 14, best abs corr 0.332
  - reanalysis_max_air_temp_k: rolling windows 12, 13, 14, best abs corr 0.323
  - reanalysis_min_air_temp_k: rolling windows 12, 13, 14, best abs corr 0.333
  - reanalysis_precip_amt_kg_per_m2: rolling windows 8, 9, 10, best abs corr 0.215
  - reanalysis_relative_humidity_percent: rolling windows 12, 13, 14, best abs corr 0.259
  - reanalysis_specific_humidity_g_per_kg: rolling windows 12, 13, 14, best abs corr 0.329
  - station_avg_temp_c: rolling windows 12, 13, 14, best abs corr 0.386
  - station_max_temp_c: rolling windows 13, 14, best abs corr 0.381
  - station_min_temp_c: rolling windows 12, 13, 14, best abs corr 0.395
