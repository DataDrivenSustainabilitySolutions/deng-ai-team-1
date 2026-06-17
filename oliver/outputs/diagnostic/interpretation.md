# Adversarial Validation Interpretation

This diagnostic trains city-specific classifiers to predict whether a row came from the original training features (`1`) or the competition test features (`0`). It reads only feature CSVs, never loads labels, and rejects columns containing `total_cases`.

The default scenarios exclude absolute time fields (`year`, `week_start_date`, and raw `weekofyear`) from model inputs. `weather_only` uses only weather, climate, vegetation, and humidity columns. `weather_plus_seasonality` adds cyclic `weekofyear` sine/cosine features.

Interpretation guide: ROC AUC near 0.50 means the classifier cannot separate train from test. Around 0.65-0.80 means meaningful covariate shift. Above 0.80 means the train/test feature distributions are highly distinguishable.

## Results

### iq - weather_only

- ROC AUC: 0.739 +/- 0.046 across 5 folds (meaningful distinguishability).
- Balanced accuracy: 0.633.
- Rows: 520 train and 156 test.
- Top classifier features: station_precip_mm (13.6%), reanalysis_precip_amt_kg_per_m2 (9.9%), ndvi_nw (7.9%), station_diur_temp_rng_c (6.8%), precipitation_amt_mm (5.7%)
- Largest direct mean shifts: station_precip_mm (SMD -0.57), ndvi_nw (SMD 0.41), reanalysis_precip_amt_kg_per_m2 (SMD 0.29), precipitation_amt_mm (SMD -0.19), reanalysis_sat_precip_amt_mm (SMD -0.19)

### iq - weather_plus_seasonality

- ROC AUC: 0.741 +/- 0.056 across 5 folds (meaningful distinguishability).
- Balanced accuracy: 0.630.
- Rows: 520 train and 156 test.
- Top classifier features: station_precip_mm (13.3%), reanalysis_precip_amt_kg_per_m2 (9.4%), ndvi_nw (7.6%), station_diur_temp_rng_c (6.4%), precipitation_amt_mm (5.6%)
- Largest direct mean shifts: station_precip_mm (SMD -0.57), ndvi_nw (SMD 0.41), reanalysis_precip_amt_kg_per_m2 (SMD 0.29), precipitation_amt_mm (SMD -0.19), reanalysis_sat_precip_amt_mm (SMD -0.19)

### sj - weather_only

- ROC AUC: 0.869 +/- 0.015 across 5 folds (strong distinguishability).
- Balanced accuracy: 0.749.
- Rows: 936 train and 260 test.
- Top classifier features: station_diur_temp_rng_c (20.1%), reanalysis_precip_amt_kg_per_m2 (12.2%), ndvi_ne (7.7%), station_precip_mm (7.3%), station_avg_temp_c (6.6%)
- Largest direct mean shifts: station_diur_temp_rng_c (SMD -0.79), station_min_temp_c (SMD 0.33), ndvi_nw (SMD -0.32), ndvi_ne (SMD -0.29), precipitation_amt_mm (SMD -0.23)

### sj - weather_plus_seasonality

- ROC AUC: 0.872 +/- 0.020 across 5 folds (strong distinguishability).
- Balanced accuracy: 0.761.
- Rows: 936 train and 260 test.
- Top classifier features: station_diur_temp_rng_c (20.7%), reanalysis_precip_amt_kg_per_m2 (11.7%), station_precip_mm (7.1%), ndvi_ne (7.0%), station_avg_temp_c (6.2%)
- Largest direct mean shifts: station_diur_temp_rng_c (SMD -0.79), station_min_temp_c (SMD 0.33), ndvi_nw (SMD -0.32), ndvi_ne (SMD -0.29), precipitation_amt_mm (SMD -0.23)

## Project Takeaway

At least one city/scenario is strongly distinguishable, so train/test covariates are not IID. This does not make the competition pure gambling, but it does mean random validation would be misleading.

Use this as a diagnostic only. It should not replace forward-chaining dengue validation, and the classifier features should not be optimized directly unless they also improve future-holdout MAE.

Generated files:
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/domain_classifier_scores.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/fold_scores.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/feature_importance.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/feature_shift_summary.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/missingness_shift_summary.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/adversarial_oof_predictions.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/auc_by_city_scenario.png`
