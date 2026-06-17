# main_goal.py changelog

Run timestamp UTC: 2026-06-17T13:30:17.241369Z

## Guardrails

- Cities are modeled separately.
- Validation is expanding, full-year forward chaining within each city.
- No total_cases, lagged cases, rolling cases, or other target-derived predictors are used as features.
- total_cases is used only as the supervised label and validation ground truth.
- Feature imputation is forward-fill plus training-fold medians to avoid backward-looking future leakage.
- Hidden external feedback is incorporated by requiring log1p-target candidates to clear a 1.00 MAE validation margin before final selection.

## Prior saved runs

| Run | Overall MAE | Rows |
| --- | ---: | ---: |
| main_log_target | 18.8758 | 1248 |
| main_reduced_additional_diff | 19.3822 | 1248 |
| main_additional_diff | 19.5409 | 1248 |
| main_reduced_features | 19.6611 | 1248 |
| main | 19.8077 | 1248 |
| main_baseline_rerun | 19.8077 | 1248 |
| main_informed | 19.8942 | 1248 |
| main_rf_optuna | 23.2011 | 1248 |

## Skipped optional candidates

| Iteration | Reason |
| --- | --- |
| 12_catboost_conservative_lag_roll_raw | catboost is not installed; skipped optional CatBoost comparison |

## Iteration summary

| Iteration | Feature set | Loss | Target transform | Overall MAE | SJ MAE | IQ MAE |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 01_calendar_l1_raw | calendar | l1 | none | 18.1458 | 23.6178 | 7.2019 |
| 02_calendar_l1_log | calendar | l1 | log1p | 18.1763 | 23.6671 | 7.1947 |
| 03_raw_weather_l1_raw | raw_weather | l1 | none | 19.8806 | 26.1106 | 7.4207 |
| 04_raw_weather_l1_log | raw_weather | l1 | log1p | 18.9736 | 24.7007 | 7.5192 |
| 05_conservative_lag_roll_l1_raw | conservative_lag_roll | l1 | none | 19.6691 | 25.6274 | 7.7524 |
| 06_conservative_lag_roll_l1_log | conservative_lag_roll | l1 | log1p | 19.0785 | 24.8413 | 7.5529 |
| 07_conservative_lag_roll_poisson | conservative_lag_roll | poisson | none | 21.4856 | 28.0986 | 8.2596 |
| 08_broad_lag_roll_l1_raw | broad_lag_roll | l1 | none | 19.8341 | 26.0397 | 7.4231 |
| 09_broad_lag_roll_l1_log | broad_lag_roll | l1 | log1p | 18.8910 | 24.6322 | 7.4087 |
| 10_broad_lag_roll_poisson | broad_lag_roll | poisson | none | 21.7003 | 28.3726 | 8.3558 |
| 11_calendar_broad_log_blend | calendar_broad_blend | weighted_average | mixed | 18.0312 | 23.4736 | 7.1466 |

## Iteration notes

### 01_calendar_l1_raw

Rationale: Start with a low-capacity seasonal baseline using only cyclic week features.

Result: overall MAE 18.1458; SJ 23.6178; IQ 7.2019.

Hidden-prior selection score: SJ 23.6178; IQ 7.2019.

Interpretation: This is the reference point for later comparisons.

Decision: Selected for final submission city/cities: sj, iq

### 02_calendar_l1_log

Rationale: Test whether the same seasonal-only model benefits from log1p target scaling.

Result: overall MAE 18.1763; SJ 23.6671; IQ 7.1947.

Hidden-prior selection score: SJ 24.6671; IQ 8.1947.

Interpretation: Worse by 0.0304 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 03_raw_weather_l1_raw

Rationale: Add the original climate and vegetation measurements without target transformation.

Result: overall MAE 19.8806; SJ 26.1106; IQ 7.4207.

Hidden-prior selection score: SJ 26.1106; IQ 7.4207.

Interpretation: Worse by 1.7043 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 04_raw_weather_l1_log

Rationale: Test log1p target scaling on the original climate and vegetation measurements.

Result: overall MAE 18.9736; SJ 24.7007; IQ 7.5192.

Hidden-prior selection score: SJ 25.7007; IQ 8.5192.

Interpretation: Improved by 0.9071 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 05_conservative_lag_roll_l1_raw

Rationale: Add a compact, biologically plausible lag/rolling set without target transformation.

Result: overall MAE 19.6691; SJ 25.6274; IQ 7.7524.

Hidden-prior selection score: SJ 25.6274; IQ 7.7524.

Interpretation: Worse by 0.6955 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 06_conservative_lag_roll_l1_log

Rationale: Test log1p target scaling on the compact lag/rolling weather feature set.

Result: overall MAE 19.0785; SJ 24.8413; IQ 7.5529.

Hidden-prior selection score: SJ 25.8413; IQ 8.5529.

Interpretation: Improved by 0.5905 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 07_conservative_lag_roll_poisson

Rationale: Test a count-style objective on the conservative feature set.

Result: overall MAE 21.4856; SJ 28.0986; IQ 8.2596.

Hidden-prior selection score: SJ 28.0986; IQ 8.2596.

Interpretation: Worse by 2.4071 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 08_broad_lag_roll_l1_raw

Rationale: Reproduce the strongest prior Oliver-style feature breadth without target transformation.

Result: overall MAE 19.8341; SJ 26.0397; IQ 7.4231.

Hidden-prior selection score: SJ 26.0397; IQ 7.4231.

Interpretation: Improved by 1.6514 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 09_broad_lag_roll_l1_log

Rationale: Apply the best prior target scaling idea to the broad weather lag/rolling feature set.

Result: overall MAE 18.8910; SJ 24.6322; IQ 7.4087.

Hidden-prior selection score: SJ 25.6322; IQ 8.4087.

Interpretation: Improved by 0.9431 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 10_broad_lag_roll_poisson

Rationale: Check whether the broad feature set benefits from a direct count objective.

Result: overall MAE 21.7003; SJ 28.3726; IQ 8.3558.

Hidden-prior selection score: SJ 28.3726; IQ 8.3558.

Interpretation: Worse by 2.8093 MAE versus the previous iteration.

Decision: Not selected for final submission.

### 11_calendar_broad_log_blend

Rationale: Blend the robust seasonal model with a small broad-weather component because fold diagnostics show weather helps on some outbreak years.

Result: overall MAE 18.0312; SJ 23.4736; IQ 7.1466.

Hidden-prior selection score: SJ 23.8069; IQ 7.4800.

Interpretation: Improved by 3.6691 MAE versus the previous iteration.

Decision: Not selected for final submission.

## Final selection

| City | Selected iteration | Validation MAE | Selection MAE | Feature set | Loss | Target transform |
| --- | --- | ---: | ---: | --- | --- | --- |
| iq | 01_calendar_l1_raw | 7.2019 | 7.2019 | calendar | l1 | none |
| sj | 01_calendar_l1_raw | 23.6178 | 23.6178 | calendar | l1 | none |

## Best overall single iteration

11_calendar_broad_log_blend had the best single-iteration overall MAE at 18.0312. The submission uses per-city selection after the hidden-prior log-target adjustment because the challenge cities behave like separate datasets and the external feedback makes small log-target validation gains less trustworthy.

## Additive no-log merged overview - 20260617T134941Z

Source run: `oliver/sandbox/outputs/main_goal_additive/20260617T134941Z`. This table merges the latest additive all-year scores with the city-specific recent-window selection scores for direct comparison. All rows below use raw `total_cases` as the target, with no log target transform and no target-derived input features.

| Iteration | Model type | Overall MAE | Overall selection | SJ all-year | SJ recent | SJ selection | IQ all-year | IQ recent | IQ selection | Selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a24_recent7_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.5697 | 15.8512 | 22.7296 | 16.7548 | 20.9371 | 7.2500 | 6.9279 | 7.1534 | sj |
| a27_recent4_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.1466 | 15.8519 | 23.7404 | 14.6442 | 21.0115 | 6.9591 | 6.3510 | 6.7767 | iq |
| a28_recent7_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 17.5801 | 15.8599 | 22.7440 | 16.7452 | 20.9444 | 7.2524 | 6.9471 | 7.1608 |  |
| a15_recent7_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.5929 | 15.8631 | 22.7644 | 16.7212 | 20.9514 | 7.2500 | 6.9327 | 7.1548 |  |
| a21_recent4_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.1731 | 15.8841 | 23.7764 | 14.6731 | 21.0454 | 6.9663 | 6.4135 | 6.8005 |  |
| a17_recent10_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.7716 | 15.9385 | 23.0565 | 16.4856 | 21.0852 | 7.2019 | 6.8365 | 7.0923 |  |
| a23_recent6_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.9207 | 15.9505 | 23.3281 | 16.0673 | 21.1499 | 7.1058 | 6.6394 | 6.9659 |  |
| a14_recent6_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0072 | 16.0031 | 23.4519 | 15.9856 | 21.2120 | 7.1178 | 6.6683 | 6.9829 |  |
| a20_recent8_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.8325 | 16.0107 | 23.1130 | 16.5481 | 21.1435 | 7.2716 | 6.9712 | 7.1815 |  |
| a16_recent9_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8654 | 16.0171 | 23.1971 | 16.5721 | 21.2096 | 7.2019 | 6.8365 | 7.0923 |  |
| a13_recent8_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8670 | 16.0384 | 23.1659 | 16.5721 | 21.1877 | 7.2692 | 6.9712 | 7.1798 |  |
| a10_calendar_lgbm_l1_recent_weighted_soft | lgbm_calendar_recent_weighted | 18.0048 | 16.2560 | 23.4135 | 17.5288 | 21.6481 | 7.1875 | 6.8221 | 7.0779 |  |
| a18_recent12_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0609 | 16.2671 | 23.4904 | 17.3269 | 21.6413 | 7.2019 | 6.8365 | 7.0923 |  |
| a02_calendar_lgbm_l1_recent_weighted | lgbm_calendar_recent_weighted | 18.2171 | 16.3607 | 23.7079 | 17.1971 | 21.7547 | 7.2356 | 6.8606 | 7.1231 |  |
| a22_recent5_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.3045 | 16.3735 | 23.8462 | 16.8606 | 21.7505 | 7.2212 | 6.8750 | 7.1173 |  |
| a19_calendar_lgbm_l1_harmonic2 | lgbm_calendar | 18.1170 | 16.3994 | 23.5709 | 17.9375 | 21.8809 | 7.2091 | 6.8462 | 7.1002 |  |
| a01_calendar_lgbm_l1_raw | lgbm_calendar | 18.1458 | 16.4160 | 23.6178 | 17.9231 | 21.9094 | 7.2019 | 6.8365 | 7.0923 |  |
| a09_calendar_lgbm_pm1_median_blend | blend | 18.2348 | 16.4689 | 23.7296 | 17.8269 | 21.9588 | 7.2452 | 6.8702 | 7.1327 |  |
| a11_calendar_lgbm_l1_recent_weighted_strong | lgbm_calendar_recent_weighted | 18.4591 | 16.4702 | 24.0829 | 16.8750 | 21.9206 | 7.2115 | 6.7837 | 7.0832 |  |
| a07_week_median_profile_pm2 | week_profile | 18.3069 | 16.5093 | 23.8425 | 17.7885 | 22.0263 | 7.2356 | 6.8413 | 7.1173 |  |
| a12_calendar_lgbm_l1_recent_weighted_curved | lgbm_calendar_recent_weighted | 18.5377 | 16.5389 | 24.1683 | 16.9231 | 21.9947 | 7.2764 | 6.8269 | 7.1416 |  |
| a06_week_median_profile_pm1 | week_profile | 18.3405 | 16.5437 | 23.8942 | 17.8558 | 22.0827 | 7.2332 | 6.8462 | 7.1171 |  |
| a26_recent3_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.5433 | 16.5565 | 24.1899 | 17.2644 | 22.1123 | 7.2500 | 6.5769 | 7.0481 |  |
| a04_week_median_profile | week_profile | 18.4936 | 16.6760 | 24.0601 | 17.9375 | 22.2233 | 7.3606 | 6.9327 | 7.2322 |  |
| a08_recent5_week_median_profile | week_profile | 18.7212 | 16.7127 | 24.3858 | 17.0577 | 22.1874 | 7.3918 | 6.9952 | 7.2728 |  |
| a25_recent2_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 20.4631 | 17.3942 | 27.2788 | 13.7788 | 23.2288 | 6.8317 | 6.6875 | 6.7885 |  |
| a05_week_mean_profile | week_profile | 23.4367 | 20.9843 | 31.5204 | 23.7260 | 29.1821 | 7.2692 | 6.7981 | 7.1279 |  |
| a03_calendar_lgbm_l1_with_year_index | lgbm_calendar_year_index | 24.3574 | 21.1283 | 32.2188 | 19.6635 | 28.4522 | 8.6346 | 7.5240 | 8.3014 |  |

## Latest additive no-log plus RF merged overview

Source run: `oliver/sandbox/outputs/main_goal_additive/20260617T135901Z`. This table extends the previous regularized LightGBM frontier with matched Random Forest checks. The selected submission is unchanged from `20260617T135709Z`: regularized LightGBM remains best for both cities under the current selection rule.

| Iteration | Model type | Overall MAE | Overall selection | SJ all-year | SJ recent | SJ selection | IQ all-year | IQ recent | IQ selection | Selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 18.0112 | 15.7377 | 23.5625 | 14.5625 | 20.8625 | 6.9087 | 6.3029 | 6.7269 | iq |
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 17.4920 | 15.7679 | 22.6274 | 16.5817 | 20.8137 | 7.2212 | 6.9087 | 7.1274 | sj |
| a31_recent4_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 18.0561 | 15.7965 | 23.6058 | 14.6394 | 20.9159 | 6.9567 | 6.4087 | 6.7923 |  |
| a35_recent4_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 18.1450 | 15.8357 | 23.7596 | 14.5481 | 20.9962 | 6.9159 | 6.3462 | 6.7450 |  |
| a24_recent7_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.5697 | 15.8512 | 22.7296 | 16.7548 | 20.9371 | 7.2500 | 6.9279 | 7.1534 |  |
| a27_recent4_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.1466 | 15.8519 | 23.7404 | 14.6442 | 21.0115 | 6.9591 | 6.3510 | 6.7767 |  |
| a28_recent7_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 17.5801 | 15.8599 | 22.7440 | 16.7452 | 20.9444 | 7.2524 | 6.9471 | 7.1608 |  |
| a15_recent7_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.5929 | 15.8631 | 22.7644 | 16.7212 | 20.9514 | 7.2500 | 6.9327 | 7.1548 |  |
| a21_recent4_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.1731 | 15.8841 | 23.7764 | 14.6731 | 21.0454 | 6.9663 | 6.4135 | 6.8005 |  |
| a34_recent7_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 17.7388 | 15.9321 | 23.0757 | 16.6875 | 21.1593 | 7.0649 | 6.7452 | 6.9690 |  |
| a17_recent10_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.7716 | 15.9385 | 23.0565 | 16.4856 | 21.0852 | 7.2019 | 6.8365 | 7.0923 |  |
| a23_recent6_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.9207 | 15.9505 | 23.3281 | 16.0673 | 21.1499 | 7.1058 | 6.6394 | 6.9659 |  |
| a14_recent6_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0072 | 16.0031 | 23.4519 | 15.9856 | 21.2120 | 7.1178 | 6.6683 | 6.9829 |  |
| a20_recent8_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.8325 | 16.0107 | 23.1130 | 16.5481 | 21.1435 | 7.2716 | 6.9712 | 7.1815 |  |
| a16_recent9_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8654 | 16.0171 | 23.1971 | 16.5721 | 21.2096 | 7.2019 | 6.8365 | 7.0923 |  |
| a13_recent8_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8670 | 16.0384 | 23.1659 | 16.5721 | 21.1877 | 7.2692 | 6.9712 | 7.1798 |  |
| a10_calendar_lgbm_l1_recent_weighted_soft | lgbm_calendar_recent_weighted | 18.0048 | 16.2560 | 23.4135 | 17.5288 | 21.6481 | 7.1875 | 6.8221 | 7.0779 |  |
| a18_recent12_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0609 | 16.2671 | 23.4904 | 17.3269 | 21.6413 | 7.2019 | 6.8365 | 7.0923 |  |
| a02_calendar_lgbm_l1_recent_weighted | lgbm_calendar_recent_weighted | 18.2171 | 16.3607 | 23.7079 | 17.1971 | 21.7547 | 7.2356 | 6.8606 | 7.1231 |  |
| a22_recent5_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.3045 | 16.3735 | 23.8462 | 16.8606 | 21.7505 | 7.2212 | 6.8750 | 7.1173 |  |
| a19_calendar_lgbm_l1_harmonic2 | lgbm_calendar | 18.1170 | 16.3994 | 23.5709 | 17.9375 | 21.8809 | 7.2091 | 6.8462 | 7.1002 |  |
| a01_calendar_lgbm_l1_raw | lgbm_calendar | 18.1458 | 16.4160 | 23.6178 | 17.9231 | 21.9094 | 7.2019 | 6.8365 | 7.0923 |  |
| a09_calendar_lgbm_pm1_median_blend | blend | 18.2348 | 16.4689 | 23.7296 | 17.8269 | 21.9588 | 7.2452 | 6.8702 | 7.1327 |  |
| a11_calendar_lgbm_l1_recent_weighted_strong | lgbm_calendar_recent_weighted | 18.4591 | 16.4702 | 24.0829 | 16.8750 | 21.9206 | 7.2115 | 6.7837 | 7.0832 |  |
| a07_week_median_profile_pm2 | week_profile | 18.3069 | 16.5093 | 23.8425 | 17.7885 | 22.0263 | 7.2356 | 6.8413 | 7.1173 |  |
| a12_calendar_lgbm_l1_recent_weighted_curved | lgbm_calendar_recent_weighted | 18.5377 | 16.5389 | 24.1683 | 16.9231 | 21.9947 | 7.2764 | 6.8269 | 7.1416 |  |
| a06_week_median_profile_pm1 | week_profile | 18.3405 | 16.5437 | 23.8942 | 17.8558 | 22.0827 | 7.2332 | 6.8462 | 7.1171 |  |
| a26_recent3_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.5433 | 16.5565 | 24.1899 | 17.2644 | 22.1123 | 7.2500 | 6.5769 | 7.0481 |  |
| a30_recent3_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.5369 | 16.5571 | 24.1526 | 17.2500 | 22.0819 | 7.3053 | 6.6250 | 7.1012 |  |
| a04_week_median_profile | week_profile | 18.4936 | 16.6760 | 24.0601 | 17.9375 | 22.2233 | 7.3606 | 6.9327 | 7.2322 |  |
| a08_recent5_week_median_profile | week_profile | 18.7212 | 16.7127 | 24.3858 | 17.0577 | 22.1874 | 7.3918 | 6.9952 | 7.2728 |  |
| a29_recent2_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 20.3654 | 17.3243 | 27.1346 | 13.7933 | 23.1322 | 6.8269 | 6.6635 | 6.7779 |  |
| a25_recent2_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 20.4631 | 17.3942 | 27.2788 | 13.7788 | 23.2288 | 6.8317 | 6.6875 | 6.7885 |  |
| a36_recent2_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 22.1955 | 18.6443 | 29.7957 | 13.8990 | 25.0267 | 6.9952 | 6.8173 | 6.9418 |  |
| a05_week_mean_profile | week_profile | 23.4367 | 20.9843 | 31.5204 | 23.7260 | 29.1821 | 7.2692 | 6.7981 | 7.1279 |  |
| a03_calendar_lgbm_l1_with_year_index | lgbm_calendar_year_index | 24.3574 | 21.1283 | 32.2188 | 19.6635 | 28.4522 | 8.6346 | 7.5240 | 8.3014 |  |

## Statistical count-model check - 20260617T141143Z

Source run: `oliver/sandbox/outputs/main_goal_additive/20260617T141143Z`. This run adds Poisson and negative-binomial GLMs as non-tree statistical count baselines. These use raw counts as the response with a count-model log link; they do not use a log-transformed target.

| Iteration | Model type | Overall MAE | SJ all-year | SJ selection | IQ all-year | IQ selection | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| a39_recent7_calendar_quantile_harmonic2 | quantile_calendar_recent_window | 17.4062 | 22.5156 | 20.7542 | 7.1875 | 7.0678 | selected for sj |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 18.0112 | 23.5625 | 20.8625 | 6.9087 | 6.7269 | selected for iq |
| a41_recent7_calendar_poisson_glm_harmonic2 | glm_poisson_calendar_recent_window | 21.7540 | 29.0637 | 25.6465 | 7.1346 | 7.0466 | not selected |
| a42_recent4_calendar_poisson_glm_harmonic2 | glm_poisson_calendar_recent_window | 21.4014 | 28.4904 | 24.3423 | 7.2236 | 7.1075 | not selected |
| a43_recent7_calendar_negbin_glm_harmonic2 | glm_negbin_calendar_recent_window | 21.6987 | 28.9844 | 25.5939 | 7.1274 | 7.0416 | not selected |
| a44_recent4_calendar_negbin_glm_harmonic2 | glm_negbin_calendar_recent_window | 21.2492 | 28.2668 | 24.1700 | 7.2139 | 7.0935 | not selected |

Interpretation: The GLMs are statistically sensible for count data, especially negative binomial for overdispersion, but this shallow seasonal-only specification underfits San Juan badly and does not beat the current IQ winner. The result supports keeping the simple seasonal quantile model for SJ and the regularized seasonal LightGBM for IQ rather than switching to a pure count GLM.

Selection sensitivity from `selection_sensitivity.csv`: SJ is robust, with `a39_recent7_calendar_quantile_harmonic2` selected in 22 of 25 sensitivity scenarios. IQ is less robust: `a29_recent2_calendar_lgbm_l1_harmonic2` is selected in 13 scenarios and `a33_recent4_calendar_lgbm_l1_harmonic2_regularized` in 12 scenarios. The default guarded rule still selects `a33` for IQ because it has better recent MAE while staying close to the all-year frontier.

## Frontier blend check - 20260617T141528Z

Source run: `oliver/sandbox/outputs/main_goal_additive/20260617T141528Z`. This run tests small model averages only on the current frontier, after sensitivity showed IQ was split between short-window and regularized-window seasonal learners.

| Iteration | Purpose | Overall MAE | SJ all-year | SJ selection | IQ all-year | IQ selection | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| a45_iq_lgbm_frontier_blend | 50/50 blend of `a29` and `a33` | 18.9215 | 24.9856 | 21.7187 | 6.7933 | 6.6923 | not selected |
| a46_iq_lgbm_rf_frontier_blend | 1/3 blend of `a29`, `a33`, and `a35` | 18.5873 | 24.4976 | 21.4161 | 6.7668 | 6.6421 | selected for iq |
| a47_sj_quantile_lgbm_frontier_blend | 50/50 blend of `a39` and `a37` | 17.3630 | 22.4495 | 20.7050 | 7.1899 | 7.0853 | selected for sj |

Interpretation: A small average of near-frontier seasonal models improves both cities without adding new features or target-derived inputs. The IQ three-way blend resolves the prior selection-sensitivity split and is selected in all 25 sensitivity scenarios. The SJ quantile/tree blend improves over the pure quantile model and is selected in 23 of 25 scenarios. This is currently the strongest no-log additive submission candidate.

Final selected submission from this run:

| City | Selected iteration | All-year MAE | Recent MAE | Selection MAE | Model type |
| --- | --- | ---: | ---: | ---: | --- |
| iq | a46_iq_lgbm_rf_frontier_blend | 6.7668 | 6.3510 | 6.6421 | blend |
| sj | a47_sj_quantile_lgbm_frontier_blend | 22.4495 | 16.6346 | 20.7050 | blend |

## Adjacent blend-weight check - 20260617T141841Z

Source run: `oliver/sandbox/outputs/main_goal_additive/20260617T141841Z`. This run tests nearby blend weights around the selected frontier blends.

| Iteration | Purpose | SJ selection | IQ selection | Decision |
| --- | --- | ---: | ---: | --- |
| a46_iq_lgbm_rf_frontier_blend | current IQ equal-third frontier blend | 21.4161 | 6.6421 | selected for iq |
| a48_iq_lgbm_rf_frontier_blend_recent_tilt | IQ blend tilted toward `a33` | 21.1578 | 6.6663 | not selected |
| a49_iq_lgbm_rf_frontier_blend_short_tilt | IQ blend tilted toward `a29` | 21.7393 | 6.6858 | not selected |
| a47_sj_quantile_lgbm_frontier_blend | current SJ 50/50 quantile/tree blend | 20.7050 | 7.0853 | selected for sj |
| a50_sj_quantile_lgbm_frontier_blend_quantile_tilt | SJ blend tilted toward `a39` | 20.7218 | 7.0928 | not selected |
| a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt | SJ blend tilted toward `a37` | 20.7149 | 7.0950 | not selected |

Interpretation: The selected equal/half frontier blends remain the best among nearby weights. This supports keeping the current blend choices rather than tuning weights more aggressively.

## Seed stability check - 20260617T142050Z

Source run: `oliver/sandbox/outputs/main_goal_additive/20260617T142050Z`. This reruns the full additive script with `--random-state 7` while keeping the default selection rule and candidate set fixed.

Policy correction: seed tuning is not an allowed optimization axis. This run is therefore not eligible for model selection, promotion, or submission choice. It is retained only as an audit artifact showing that the selected model identities did not change under one alternate seed. The submission candidate remains the fixed-default-seed run unless a non-seed modeling change justifies a newer run.

Implementation update: `oliver/sandbox/main_goal_additive.py` now enforces the fixed seed policy by rejecting any `--random-state` other than `42`.

| City | Seed 42 selected | Seed 42 selection MAE | Seed 7 selected | Seed 7 selection MAE |
| --- | --- | ---: | --- | ---: |
| iq | a46_iq_lgbm_rf_frontier_blend | 6.6421 | a46_iq_lgbm_rf_frontier_blend | 6.6865 |
| sj | a47_sj_quantile_lgbm_frontier_blend | 20.7050 | a47_sj_quantile_lgbm_frontier_blend | 20.6726 |

Sensitivity summary: under seed 7, `a46_iq_lgbm_rf_frontier_blend` is selected in 21 of 25 IQ sensitivity scenarios and `a48_iq_lgbm_rf_frontier_blend_recent_tilt` in 4 of 25; `a47_sj_quantile_lgbm_frontier_blend` remains selected in 23 of 25 SJ scenarios. The submitted prediction file changes 51 rows versus the seed-42 run, with mean absolute changed-row movement of 1 case in both cities.

Interpretation: The selected model identities are seed-stable. Because the score movement is small and mixed by city, this is evidence for keeping the current frontier blend structure, not evidence for chasing a single lucky seed.

Operational rule going forward: keep `--random-state` fixed at the script default and do not compare, tune, average, or select submissions based on random seed.

## Latest guarded additive no-log merged overview

Source run: `oliver/sandbox/outputs/main_goal_additive/20260617T140519Z`. This run adds stronger regularization and simple median linear seasonal quantile models. Selection uses the same 70/30 all-year/recent score, but with a city-level all-year guardrail: candidates must be within 0.35 MAE of that city's best all-year MAE. This keeps the unguarded recent-only SJ spike out while still allowing the simpler quantile seasonal model to win SJ.

| Iteration | Model type | Overall MAE | Overall selection | SJ all-year | SJ recent | SJ selection | IQ all-year | IQ recent | IQ selection | Selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | lgbm_calendar_recent_window | 17.8950 | 15.6628 | 23.3906 | 14.5673 | 20.7436 | 6.9038 | 6.3413 | 6.7351 |  |
| a39_recent7_calendar_quantile_harmonic2 | quantile_calendar_recent_window | 17.4062 | 15.6993 | 22.5156 | 16.6442 | 20.7542 | 7.1875 | 6.7885 | 7.0678 | sj |
| a40_recent4_calendar_quantile_harmonic2 | quantile_calendar_recent_window | 17.8862 | 15.7165 | 23.3546 | 14.9327 | 20.8280 | 6.9495 | 6.3750 | 6.7772 |  |
| a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized | lgbm_calendar_recent_window | 17.4223 | 15.7343 | 22.5312 | 16.6683 | 20.7724 | 7.2043 | 6.9231 | 7.1200 |  |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 18.0112 | 15.7377 | 23.5625 | 14.5625 | 20.8625 | 6.9087 | 6.3029 | 6.7269 | iq |
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 17.4920 | 15.7679 | 22.6274 | 16.5817 | 20.8137 | 7.2212 | 6.9087 | 7.1274 |  |
| a31_recent4_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 18.0561 | 15.7965 | 23.6058 | 14.6394 | 20.9159 | 6.9567 | 6.4087 | 6.7923 |  |
| a35_recent4_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 18.1450 | 15.8357 | 23.7596 | 14.5481 | 20.9962 | 6.9159 | 6.3462 | 6.7450 |  |
| a24_recent7_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.5697 | 15.8512 | 22.7296 | 16.7548 | 20.9371 | 7.2500 | 6.9279 | 7.1534 |  |
| a27_recent4_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.1466 | 15.8519 | 23.7404 | 14.6442 | 21.0115 | 6.9591 | 6.3510 | 6.7767 |  |
| a28_recent7_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 17.5801 | 15.8599 | 22.7440 | 16.7452 | 20.9444 | 7.2524 | 6.9471 | 7.1608 |  |
| a15_recent7_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.5929 | 15.8631 | 22.7644 | 16.7212 | 20.9514 | 7.2500 | 6.9327 | 7.1548 |  |
| a21_recent4_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.1731 | 15.8841 | 23.7764 | 14.6731 | 21.0454 | 6.9663 | 6.4135 | 6.8005 |  |
| a34_recent7_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 17.7388 | 15.9321 | 23.0757 | 16.6875 | 21.1593 | 7.0649 | 6.7452 | 6.9690 |  |
| a17_recent10_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.7716 | 15.9385 | 23.0565 | 16.4856 | 21.0852 | 7.2019 | 6.8365 | 7.0923 |  |
| a23_recent6_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.9207 | 15.9505 | 23.3281 | 16.0673 | 21.1499 | 7.1058 | 6.6394 | 6.9659 |  |
| a14_recent6_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0072 | 16.0031 | 23.4519 | 15.9856 | 21.2120 | 7.1178 | 6.6683 | 6.9829 |  |
| a20_recent8_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.8325 | 16.0107 | 23.1130 | 16.5481 | 21.1435 | 7.2716 | 6.9712 | 7.1815 |  |
| a16_recent9_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8654 | 16.0171 | 23.1971 | 16.5721 | 21.2096 | 7.2019 | 6.8365 | 7.0923 |  |
| a13_recent8_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8670 | 16.0384 | 23.1659 | 16.5721 | 21.1877 | 7.2692 | 6.9712 | 7.1798 |  |
| a10_calendar_lgbm_l1_recent_weighted_soft | lgbm_calendar_recent_weighted | 18.0048 | 16.2560 | 23.4135 | 17.5288 | 21.6481 | 7.1875 | 6.8221 | 7.0779 |  |
| a18_recent12_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0609 | 16.2671 | 23.4904 | 17.3269 | 21.6413 | 7.2019 | 6.8365 | 7.0923 |  |
| a02_calendar_lgbm_l1_recent_weighted | lgbm_calendar_recent_weighted | 18.2171 | 16.3607 | 23.7079 | 17.1971 | 21.7547 | 7.2356 | 6.8606 | 7.1231 |  |
| a22_recent5_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.3045 | 16.3735 | 23.8462 | 16.8606 | 21.7505 | 7.2212 | 6.8750 | 7.1173 |  |
| a19_calendar_lgbm_l1_harmonic2 | lgbm_calendar | 18.1170 | 16.3994 | 23.5709 | 17.9375 | 21.8809 | 7.2091 | 6.8462 | 7.1002 |  |
| a01_calendar_lgbm_l1_raw | lgbm_calendar | 18.1458 | 16.4160 | 23.6178 | 17.9231 | 21.9094 | 7.2019 | 6.8365 | 7.0923 |  |
| a09_calendar_lgbm_pm1_median_blend | blend | 18.2348 | 16.4689 | 23.7296 | 17.8269 | 21.9588 | 7.2452 | 6.8702 | 7.1327 |  |
| a11_calendar_lgbm_l1_recent_weighted_strong | lgbm_calendar_recent_weighted | 18.4591 | 16.4702 | 24.0829 | 16.8750 | 21.9206 | 7.2115 | 6.7837 | 7.0832 |  |
| a07_week_median_profile_pm2 | week_profile | 18.3069 | 16.5093 | 23.8425 | 17.7885 | 22.0263 | 7.2356 | 6.8413 | 7.1173 |  |
| a12_calendar_lgbm_l1_recent_weighted_curved | lgbm_calendar_recent_weighted | 18.5377 | 16.5389 | 24.1683 | 16.9231 | 21.9947 | 7.2764 | 6.8269 | 7.1416 |  |
| a06_week_median_profile_pm1 | week_profile | 18.3405 | 16.5437 | 23.8942 | 17.8558 | 22.0827 | 7.2332 | 6.8462 | 7.1171 |  |
| a26_recent3_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.5433 | 16.5565 | 24.1899 | 17.2644 | 22.1123 | 7.2500 | 6.5769 | 7.0481 |  |
| a30_recent3_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.5369 | 16.5571 | 24.1526 | 17.2500 | 22.0819 | 7.3053 | 6.6250 | 7.1012 |  |
| a04_week_median_profile | week_profile | 18.4936 | 16.6760 | 24.0601 | 17.9375 | 22.2233 | 7.3606 | 6.9327 | 7.2322 |  |
| a08_recent5_week_median_profile | week_profile | 18.7212 | 16.7127 | 24.3858 | 17.0577 | 22.1874 | 7.3918 | 6.9952 | 7.2728 |  |
| a29_recent2_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 20.3654 | 17.3243 | 27.1346 | 13.7933 | 23.1322 | 6.8269 | 6.6635 | 6.7779 |  |
| a25_recent2_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 20.4631 | 17.3942 | 27.2788 | 13.7788 | 23.2288 | 6.8317 | 6.6875 | 6.7885 |  |
| a36_recent2_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 22.1955 | 18.6443 | 29.7957 | 13.8990 | 25.0267 | 6.9952 | 6.8173 | 6.9418 |  |
| a05_week_mean_profile | week_profile | 23.4367 | 20.9843 | 31.5204 | 23.7260 | 29.1821 | 7.2692 | 6.7981 | 7.1279 |  |
| a03_calendar_lgbm_l1_with_year_index | lgbm_calendar_year_index | 24.3574 | 21.1283 | 32.2188 | 19.6635 | 28.4522 | 8.6346 | 7.5240 | 8.3014 |  |
