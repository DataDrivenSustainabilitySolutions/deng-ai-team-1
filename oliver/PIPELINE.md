1. The script reads the train features, test features, labels, and submission format CSV files. It checks that row keys are unique and that the submission rows match the test rows.

2. It merges labels onto the training features only for training and validation. `total_cases` is kept as the target, not as an engineered input feature.

3. It joins train and test rows temporarily so preprocessing and feature engineering are applied in the same way. The original row order is preserved for later output.

4. Missing numeric weather values are filled by linear interpolation within each city separately. San Juan and Iquitos are never blended.

5. The script adds seasonal features using sine and cosine transforms of `weekofyear`. This lets the model treat week 52 and week 1 as close together.

6. It adds lag columns named `<feature>_lag_<n>` for selected humidity, temperature, NDVI, and precipitation features. These lags represent delayed dengue effects from earlier weather conditions.

7. It adds rolling mean columns named `<feature>_rolling_<n>_mean` for selected weather and precipitation features. These summarize recent exposure over the last several weeks.

8. It builds separate feature sets for San Juan and Iquitos.

San Juan (`sj`):
- `weekofyear_sin`, `weekofyear_cos`: encode yearly seasonality while keeping week 52 close to week 1.
- `reanalysis_air_temp_k`: uses raw air temperature plus 6/8/10/12-week lags and 8/10/12/14-week rolling means.
- `reanalysis_avg_temp_k`: uses raw average reanalysis temperature plus 6/8/10/12-week lags and 8/10/12/14-week rolling means.
- `reanalysis_dew_point_temp_k`: uses raw dew point plus 1/2/3/5/6/8/10/12-week lags and 3/4/6/8/10/12/14-week rolling means.
- `reanalysis_max_air_temp_k`: uses raw maximum air temperature plus 6/8/10/12-week lags and 8/10/12/14-week rolling means.
- `reanalysis_min_air_temp_k`: uses raw minimum air temperature plus 1/2/3/5/6/8/10/12-week lags and 3/4/5/6/7/8/10/12/14-week rolling means.
- `reanalysis_relative_humidity_percent`: uses raw relative humidity as a direct humidity signal.
- `reanalysis_specific_humidity_g_per_kg`: uses raw specific humidity plus 1/2/3/5/6/8/10/12-week lags and 3/4/6/8/10/12/14-week rolling means.
- `station_avg_temp_c`: uses raw station average temperature plus 2/4/5/6/8/10/12-week lags and 3/4/5/6/8/10/12/14-week rolling means.
- `station_max_temp_c`: uses raw station maximum temperature plus 4/6/8/10/12-week lags and 5/6/8/10/12/14-week rolling means.
- `station_min_temp_c`: uses raw station minimum temperature plus 1/2/3/4/6/8/10/12-week lags and 3/4/5/6/8/10/12/14-week rolling means.

Iquitos (`iq`):
- `weekofyear_sin`, `weekofyear_cos`: encode yearly seasonality while keeping week 52 close to week 1.
- `reanalysis_specific_humidity_g_per_kg`: uses raw specific humidity plus 1/2/3/5-week lags and 3/4/6/8-week rolling means.
- `reanalysis_dew_point_temp_k`: uses raw dew point plus 1/2/3/5-week lags and 3/4/6/8-week rolling means.
- `reanalysis_min_air_temp_k`: uses raw minimum air temperature plus 1/2/3/5-week lags and 3/4/5/6/7/8-week rolling means.
- `station_min_temp_c`: uses raw station minimum temperature plus 1/2/3/4/6-week lags and 3/4/5/6-week rolling means.
- `station_avg_temp_c`: uses raw station average temperature plus 2/4/6-week lags and 3/4/5/6-week rolling means.
- `station_max_temp_c`: uses raw station maximum temperature plus 4/6-week lags and 5/6-week rolling means.
- `ndvi_ne`: uses raw northeast NDVI as a vegetation signal.
- `ndvi_nw`: uses raw northwest NDVI plus 10/11/14-week lags.
- `ndvi_se`: uses raw southeast NDVI as a vegetation signal.
- `ndvi_sw`: uses raw southwest NDVI plus 10/11/14-week lags.
- `precipitation_amt_mm`: uses raw precipitation plus 2/3-week lags and 4/7/10/14-week rolling means.
- `reanalysis_sat_precip_amt_mm`: uses raw satellite precipitation plus 2/3-week lags and 4/7/10/14-week rolling means.
- `reanalysis_precip_amt_kg_per_m2`: uses raw reanalysis precipitation as an additional rainfall signal.

9. It trains separate LightGBM models for each city. The script tests candidate losses, including Tweedie, Poisson, and L1, and chooses the best one by forward-year validation.

10. If hyperparameter tuning is enabled, Optuna searches LightGBM settings separately for each city. Otherwise, the script uses fixed default settings.

11. Validation uses expanding full-year folds. Each fold trains on past years only and validates on a later full year, avoiding random splits.

12. Final models are trained on all available training rows for each city. Test predictions are clipped at zero, rounded to integers, and inserted into the submission format.

13. The script writes outputs under `oliver/outputs/main/`. These include preprocessed data, missing-value summaries, validation scores, feature importances, loss-selection results, an append-only experiment log, and the final submission CSV.

14. Optional CSV preview images are created so the generated output tables can be inspected quickly.
