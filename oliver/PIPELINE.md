1. The script reads the train features, test features, labels, and submission format CSV files. It checks that row keys are unique and that the submission rows match the test rows.

2. It merges labels onto the training features only for training and validation. `total_cases` is kept as the target, not as an engineered input feature.

3. It joins train and test rows temporarily so preprocessing and feature engineering are applied in the same way. The original row order is preserved for later output.

4. Missing numeric weather values are filled by linear interpolation within each city separately. San Juan and Iquitos are never blended.

5. The script adds seasonal features using sine and cosine transforms of `weekofyear`. This lets the model treat week 52 and week 1 as close together.

6. It adds lag columns named `<feature>_lag_<n>` for selected humidity, temperature, NDVI, and precipitation features. These lags represent delayed dengue effects from earlier weather conditions.

7. It adds rolling mean columns named `<feature>_rolling_<n>_mean` for selected weather and precipitation features. These summarize recent exposure over the last several weeks.

8. It builds separate feature sets for San Juan and Iquitos.
   - `sj`: `weekofyear_sin/cos`, raw temperature and humidity features, temperature/humidity lags up to 12 weeks, and rolling means up to 14 weeks.
   - `iq`: `weekofyear_sin/cos`, raw humidity/dew/temp/NDVI/precipitation features, short humidity/temp lags and rolling means, `ndvi_sw/nw_lag_10/11/14`, and precipitation lag 2/3 plus rolling 4/7/10/14.

9. It trains separate LightGBM models for each city. The script tests candidate losses, including Tweedie, Poisson, and L1, and chooses the best one by forward-year validation.

10. If hyperparameter tuning is enabled, Optuna searches LightGBM settings separately for each city. Otherwise, the script uses fixed default settings.

11. Validation uses expanding full-year folds. Each fold trains on past years only and validates on a later full year, avoiding random splits.

12. Final models are trained on all available training rows for each city. Test predictions are clipped at zero, rounded to integers, and inserted into the submission format.

13. The script writes outputs under `oliver/outputs/main/`. These include preprocessed data, missing-value summaries, validation scores, feature importances, loss-selection results, and the final submission CSV.

14. Optional CSV preview images are created so the generated output tables can be inspected quickly.
