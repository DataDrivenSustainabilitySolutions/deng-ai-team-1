# main_goal_additive.py changelog

Run timestamp UTC: 2026-06-17T14:18:41.769480Z

## Purpose

This is an additive no-log iteration after the first main_goal run.
It keeps prior outputs intact and tests only shallow seasonal candidates.

## Guardrails

- Cities are modeled separately.
- Validation is expanding, full-year forward chaining within each city.
- No log target transform is used in any candidate.
- No total_cases-derived predictors, lagged cases, or rolling cases are used as inputs.
- total_cases is used only as the supervised label and validation ground truth.
- Final additive selection uses 70% all-year MAE and 30% latest-4-validation-year MAE.
- Final city selection is restricted to candidates within 0.35 MAE of that city's best all-year MAE to avoid chasing a narrow recent-year validation fluctuation.

## Previous main_goal reference

| Prior iteration | Overall MAE |
| --- | ---: |
| 11_calendar_broad_log_blend | 18.0312 |
| 01_calendar_l1_raw | 18.1458 |
| 02_calendar_l1_log | 18.1763 |
| 09_broad_lag_roll_l1_log | 18.8910 |
| 04_raw_weather_l1_log | 18.9736 |

## Additive iteration summary

| Iteration | Model type | Overall MAE | SJ MAE | IQ MAE |
| --- | --- | ---: | ---: | ---: |
| a01_calendar_lgbm_l1_raw | lgbm_calendar | 18.1458 | 23.6178 | 7.2019 |
| a02_calendar_lgbm_l1_recent_weighted | lgbm_calendar_recent_weighted | 18.2171 | 23.7079 | 7.2356 |
| a03_calendar_lgbm_l1_with_year_index | lgbm_calendar_year_index | 24.3574 | 32.2188 | 8.6346 |
| a04_week_median_profile | week_profile | 18.4936 | 24.0601 | 7.3606 |
| a05_week_mean_profile | week_profile | 23.4367 | 31.5204 | 7.2692 |
| a06_week_median_profile_pm1 | week_profile | 18.3405 | 23.8942 | 7.2332 |
| a07_week_median_profile_pm2 | week_profile | 18.3069 | 23.8425 | 7.2356 |
| a08_recent5_week_median_profile | week_profile | 18.7212 | 24.3858 | 7.3918 |
| a10_calendar_lgbm_l1_recent_weighted_soft | lgbm_calendar_recent_weighted | 18.0048 | 23.4135 | 7.1875 |
| a11_calendar_lgbm_l1_recent_weighted_strong | lgbm_calendar_recent_weighted | 18.4591 | 24.0829 | 7.2115 |
| a12_calendar_lgbm_l1_recent_weighted_curved | lgbm_calendar_recent_weighted | 18.5377 | 24.1683 | 7.2764 |
| a13_recent8_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8670 | 23.1659 | 7.2692 |
| a14_recent6_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0072 | 23.4519 | 7.1178 |
| a15_recent7_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.5929 | 22.7644 | 7.2500 |
| a16_recent9_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8654 | 23.1971 | 7.2019 |
| a17_recent10_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.7716 | 23.0565 | 7.2019 |
| a18_recent12_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0609 | 23.4904 | 7.2019 |
| a19_calendar_lgbm_l1_harmonic2 | lgbm_calendar | 18.1170 | 23.5709 | 7.2091 |
| a20_recent8_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.8325 | 23.1130 | 7.2716 |
| a21_recent4_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.1731 | 23.7764 | 6.9663 |
| a22_recent5_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.3045 | 23.8462 | 7.2212 |
| a23_recent6_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.9207 | 23.3281 | 7.1058 |
| a24_recent7_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.5697 | 22.7296 | 7.2500 |
| a25_recent2_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 20.4631 | 27.2788 | 6.8317 |
| a26_recent3_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.5433 | 24.1899 | 7.2500 |
| a27_recent4_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.1466 | 23.7404 | 6.9591 |
| a28_recent7_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 17.5801 | 22.7440 | 7.2524 |
| a29_recent2_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 20.3654 | 27.1346 | 6.8269 |
| a30_recent3_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.5369 | 24.1526 | 7.3053 |
| a31_recent4_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 18.0561 | 23.6058 | 6.9567 |
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 17.4920 | 22.6274 | 7.2212 |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 18.0112 | 23.5625 | 6.9087 |
| a34_recent7_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 17.7388 | 23.0757 | 7.0649 |
| a35_recent4_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 18.1450 | 23.7596 | 6.9159 |
| a36_recent2_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 22.1955 | 29.7957 | 6.9952 |
| a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized | lgbm_calendar_recent_window | 17.4223 | 22.5312 | 7.2043 |
| a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | lgbm_calendar_recent_window | 17.8950 | 23.3906 | 6.9038 |
| a39_recent7_calendar_quantile_harmonic2 | quantile_calendar_recent_window | 17.4062 | 22.5156 | 7.1875 |
| a40_recent4_calendar_quantile_harmonic2 | quantile_calendar_recent_window | 17.8862 | 23.3546 | 6.9495 |
| a41_recent7_calendar_poisson_glm_harmonic2 | glm_poisson_calendar_recent_window | 21.7540 | 29.0637 | 7.1346 |
| a42_recent4_calendar_poisson_glm_harmonic2 | glm_poisson_calendar_recent_window | 21.4014 | 28.4904 | 7.2236 |
| a43_recent7_calendar_negbin_glm_harmonic2 | glm_negbin_calendar_recent_window | 21.6987 | 28.9844 | 7.1274 |
| a44_recent4_calendar_negbin_glm_harmonic2 | glm_negbin_calendar_recent_window | 21.2492 | 28.2668 | 7.2139 |
| a09_calendar_lgbm_pm1_median_blend | blend | 18.2348 | 23.7296 | 7.2452 |
| a45_iq_lgbm_frontier_blend | blend | 18.9215 | 24.9856 | 6.7933 |
| a46_iq_lgbm_rf_frontier_blend | blend | 18.5873 | 24.4976 | 6.7668 |
| a47_sj_quantile_lgbm_frontier_blend | blend | 17.3630 | 22.4495 | 7.1899 |
| a48_iq_lgbm_rf_frontier_blend_recent_tilt | blend | 18.3165 | 24.0709 | 6.8077 |
| a49_iq_lgbm_rf_frontier_blend_short_tilt | blend | 18.9551 | 25.0397 | 6.7861 |
| a50_sj_quantile_lgbm_frontier_blend_quantile_tilt | blend | 17.3790 | 22.4651 | 7.2067 |
| a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt | blend | 17.3694 | 22.4615 | 7.1851 |

## Selection score summary

| Iteration | City | All-year MAE | Recent MAE | Selection MAE |
| --- | --- | ---: | ---: | ---: |
| a46_iq_lgbm_rf_frontier_blend | iq | 6.7668 | 6.3510 | 6.6421 |
| a48_iq_lgbm_rf_frontier_blend_recent_tilt | iq | 6.8077 | 6.3365 | 6.6663 |
| a49_iq_lgbm_rf_frontier_blend_short_tilt | iq | 6.7861 | 6.4519 | 6.6858 |
| a45_iq_lgbm_frontier_blend | iq | 6.7933 | 6.4567 | 6.6923 |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | iq | 6.9087 | 6.3029 | 6.7269 |
| a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | iq | 6.9038 | 6.3413 | 6.7351 |
| a35_recent4_calendar_rf_abs_harmonic2 | iq | 6.9159 | 6.3462 | 6.7450 |
| a27_recent4_calendar_lgbm_l1_harmonic2 | iq | 6.9591 | 6.3510 | 6.7767 |
| a40_recent4_calendar_quantile_harmonic2 | iq | 6.9495 | 6.3750 | 6.7772 |
| a29_recent2_calendar_lgbm_l1_harmonic2 | iq | 6.8269 | 6.6635 | 6.7779 |
| a25_recent2_calendar_lgbm_l1_raw | iq | 6.8317 | 6.6875 | 6.7885 |
| a31_recent4_calendar_lgbm_l1_harmonic3 | iq | 6.9567 | 6.4087 | 6.7923 |
| a21_recent4_calendar_lgbm_l1_raw | iq | 6.9663 | 6.4135 | 6.8005 |
| a36_recent2_calendar_rf_abs_harmonic2 | iq | 6.9952 | 6.8173 | 6.9418 |
| a23_recent6_calendar_lgbm_l1_harmonic2 | iq | 7.1058 | 6.6394 | 6.9659 |
| a34_recent7_calendar_rf_abs_harmonic2 | iq | 7.0649 | 6.7452 | 6.9690 |
| a14_recent6_calendar_lgbm_l1_raw | iq | 7.1178 | 6.6683 | 6.9829 |
| a43_recent7_calendar_negbin_glm_harmonic2 | iq | 7.1274 | 6.8413 | 7.0416 |
| a41_recent7_calendar_poisson_glm_harmonic2 | iq | 7.1346 | 6.8413 | 7.0466 |
| a26_recent3_calendar_lgbm_l1_raw | iq | 7.2500 | 6.5769 | 7.0481 |
| a39_recent7_calendar_quantile_harmonic2 | iq | 7.1875 | 6.7885 | 7.0678 |
| a10_calendar_lgbm_l1_recent_weighted_soft | iq | 7.1875 | 6.8221 | 7.0779 |
| a11_calendar_lgbm_l1_recent_weighted_strong | iq | 7.2115 | 6.7837 | 7.0832 |
| a47_sj_quantile_lgbm_frontier_blend | iq | 7.1899 | 6.8413 | 7.0853 |
| a01_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a16_recent9_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a17_recent10_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a18_recent12_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a50_sj_quantile_lgbm_frontier_blend_quantile_tilt | iq | 7.2067 | 6.8269 | 7.0928 |
| a44_recent4_calendar_negbin_glm_harmonic2 | iq | 7.2139 | 6.8125 | 7.0935 |
| a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt | iq | 7.1851 | 6.8846 | 7.0950 |
| a19_calendar_lgbm_l1_harmonic2 | iq | 7.2091 | 6.8462 | 7.1002 |
| a30_recent3_calendar_lgbm_l1_harmonic2 | iq | 7.3053 | 6.6250 | 7.1012 |
| a42_recent4_calendar_poisson_glm_harmonic2 | iq | 7.2236 | 6.8365 | 7.1075 |
| a06_week_median_profile_pm1 | iq | 7.2332 | 6.8462 | 7.1171 |
| a22_recent5_calendar_lgbm_l1_raw | iq | 7.2212 | 6.8750 | 7.1173 |
| a07_week_median_profile_pm2 | iq | 7.2356 | 6.8413 | 7.1173 |
| a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized | iq | 7.2043 | 6.9231 | 7.1200 |
| a02_calendar_lgbm_l1_recent_weighted | iq | 7.2356 | 6.8606 | 7.1231 |
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | iq | 7.2212 | 6.9087 | 7.1274 |
| a05_week_mean_profile | iq | 7.2692 | 6.7981 | 7.1279 |
| a09_calendar_lgbm_pm1_median_blend | iq | 7.2452 | 6.8702 | 7.1327 |
| a12_calendar_lgbm_l1_recent_weighted_curved | iq | 7.2764 | 6.8269 | 7.1416 |
| a24_recent7_calendar_lgbm_l1_harmonic2 | iq | 7.2500 | 6.9279 | 7.1534 |
| a15_recent7_calendar_lgbm_l1_raw | iq | 7.2500 | 6.9327 | 7.1548 |
| a28_recent7_calendar_lgbm_l1_harmonic3 | iq | 7.2524 | 6.9471 | 7.1608 |
| a13_recent8_calendar_lgbm_l1_raw | iq | 7.2692 | 6.9712 | 7.1798 |
| a20_recent8_calendar_lgbm_l1_harmonic2 | iq | 7.2716 | 6.9712 | 7.1815 |
| a04_week_median_profile | iq | 7.3606 | 6.9327 | 7.2322 |
| a08_recent5_week_median_profile | iq | 7.3918 | 6.9952 | 7.2728 |
| a03_calendar_lgbm_l1_with_year_index | iq | 8.6346 | 7.5240 | 8.3014 |
| a47_sj_quantile_lgbm_frontier_blend | sj | 22.4495 | 16.6346 | 20.7050 |
| a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt | sj | 22.4615 | 16.6394 | 20.7149 |
| a50_sj_quantile_lgbm_frontier_blend_quantile_tilt | sj | 22.4651 | 16.6538 | 20.7218 |
| a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | sj | 23.3906 | 14.5673 | 20.7436 |
| a39_recent7_calendar_quantile_harmonic2 | sj | 22.5156 | 16.6442 | 20.7542 |
| a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized | sj | 22.5312 | 16.6683 | 20.7724 |
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | sj | 22.6274 | 16.5817 | 20.8137 |
| a40_recent4_calendar_quantile_harmonic2 | sj | 23.3546 | 14.9327 | 20.8280 |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | sj | 23.5625 | 14.5625 | 20.8625 |
| a31_recent4_calendar_lgbm_l1_harmonic3 | sj | 23.6058 | 14.6394 | 20.9159 |
| a24_recent7_calendar_lgbm_l1_harmonic2 | sj | 22.7296 | 16.7548 | 20.9371 |
| a28_recent7_calendar_lgbm_l1_harmonic3 | sj | 22.7440 | 16.7452 | 20.9444 |
| a15_recent7_calendar_lgbm_l1_raw | sj | 22.7644 | 16.7212 | 20.9514 |
| a35_recent4_calendar_rf_abs_harmonic2 | sj | 23.7596 | 14.5481 | 20.9962 |
| a27_recent4_calendar_lgbm_l1_harmonic2 | sj | 23.7404 | 14.6442 | 21.0115 |
| a21_recent4_calendar_lgbm_l1_raw | sj | 23.7764 | 14.6731 | 21.0454 |
| a17_recent10_calendar_lgbm_l1_raw | sj | 23.0565 | 16.4856 | 21.0852 |
| a20_recent8_calendar_lgbm_l1_harmonic2 | sj | 23.1130 | 16.5481 | 21.1435 |
| a23_recent6_calendar_lgbm_l1_harmonic2 | sj | 23.3281 | 16.0673 | 21.1499 |
| a48_iq_lgbm_rf_frontier_blend_recent_tilt | sj | 24.0709 | 14.3606 | 21.1578 |
| a34_recent7_calendar_rf_abs_harmonic2 | sj | 23.0757 | 16.6875 | 21.1593 |
| a13_recent8_calendar_lgbm_l1_raw | sj | 23.1659 | 16.5721 | 21.1877 |
| a16_recent9_calendar_lgbm_l1_raw | sj | 23.1971 | 16.5721 | 21.2096 |
| a14_recent6_calendar_lgbm_l1_raw | sj | 23.4519 | 15.9856 | 21.2120 |
| a46_iq_lgbm_rf_frontier_blend | sj | 24.4976 | 14.2260 | 21.4161 |
| a18_recent12_calendar_lgbm_l1_raw | sj | 23.4904 | 17.3269 | 21.6413 |
| a10_calendar_lgbm_l1_recent_weighted_soft | sj | 23.4135 | 17.5288 | 21.6481 |
| a45_iq_lgbm_frontier_blend | sj | 24.9856 | 14.0962 | 21.7187 |
| a49_iq_lgbm_rf_frontier_blend_short_tilt | sj | 25.0397 | 14.0385 | 21.7393 |
| a22_recent5_calendar_lgbm_l1_raw | sj | 23.8462 | 16.8606 | 21.7505 |
| a02_calendar_lgbm_l1_recent_weighted | sj | 23.7079 | 17.1971 | 21.7547 |
| a19_calendar_lgbm_l1_harmonic2 | sj | 23.5709 | 17.9375 | 21.8809 |
| a01_calendar_lgbm_l1_raw | sj | 23.6178 | 17.9231 | 21.9094 |
| a11_calendar_lgbm_l1_recent_weighted_strong | sj | 24.0829 | 16.8750 | 21.9206 |
| a09_calendar_lgbm_pm1_median_blend | sj | 23.7296 | 17.8269 | 21.9588 |
| a12_calendar_lgbm_l1_recent_weighted_curved | sj | 24.1683 | 16.9231 | 21.9947 |
| a07_week_median_profile_pm2 | sj | 23.8425 | 17.7885 | 22.0263 |
| a30_recent3_calendar_lgbm_l1_harmonic2 | sj | 24.1526 | 17.2500 | 22.0819 |
| a06_week_median_profile_pm1 | sj | 23.8942 | 17.8558 | 22.0827 |
| a26_recent3_calendar_lgbm_l1_raw | sj | 24.1899 | 17.2644 | 22.1123 |
| a08_recent5_week_median_profile | sj | 24.3858 | 17.0577 | 22.1874 |
| a04_week_median_profile | sj | 24.0601 | 17.9375 | 22.2233 |
| a29_recent2_calendar_lgbm_l1_harmonic2 | sj | 27.1346 | 13.7933 | 23.1322 |
| a25_recent2_calendar_lgbm_l1_raw | sj | 27.2788 | 13.7788 | 23.2288 |
| a44_recent4_calendar_negbin_glm_harmonic2 | sj | 28.2668 | 14.6106 | 24.1700 |
| a42_recent4_calendar_poisson_glm_harmonic2 | sj | 28.4904 | 14.6635 | 24.3423 |
| a36_recent2_calendar_rf_abs_harmonic2 | sj | 29.7957 | 13.8990 | 25.0267 |
| a43_recent7_calendar_negbin_glm_harmonic2 | sj | 28.9844 | 17.6827 | 25.5939 |
| a41_recent7_calendar_poisson_glm_harmonic2 | sj | 29.0637 | 17.6731 | 25.6465 |
| a03_calendar_lgbm_l1_with_year_index | sj | 32.2188 | 19.6635 | 28.4522 |
| a05_week_mean_profile | sj | 31.5204 | 23.7260 | 29.1821 |

## Iteration notes

### a01_calendar_lgbm_l1_raw

Rationale: Re-run the previous selected no-log calendar LightGBM as the additive baseline.

Result: overall MAE 18.1458; SJ 23.6178; IQ 7.2019.

Selection score: SJ 21.9094; IQ 7.0923.

Decision: Not selected for the additive submission.

### a02_calendar_lgbm_l1_recent_weighted

Rationale: Favor recent years modestly because the test set is a future holdout.

Result: overall MAE 18.2171; SJ 23.7079; IQ 7.2356.

Selection score: SJ 21.7547; IQ 7.1231.

Decision: Not selected for the additive submission.

### a03_calendar_lgbm_l1_with_year_index

Rationale: Check whether a simple non-target time trend helps without adding weather complexity.

Result: overall MAE 24.3574; SJ 32.2188; IQ 8.6346.

Selection score: SJ 28.4522; IQ 8.3014.

Decision: Not selected for the additive submission.

### a04_week_median_profile

Rationale: Use the direct week-of-year median, the natural MAE baseline for a seasonal count pattern.

Result: overall MAE 18.4936; SJ 24.0601; IQ 7.3606.

Selection score: SJ 22.2233; IQ 7.2322.

Decision: Not selected for the additive submission.

### a05_week_mean_profile

Rationale: Compare the week mean against the median to see whether outbreak years should pull predictions up.

Result: overall MAE 23.4367; SJ 31.5204; IQ 7.2692.

Selection score: SJ 29.1821; IQ 7.1279.

Decision: Not selected for the additive submission.

### a06_week_median_profile_pm1

Rationale: Smooth the weekly median over neighboring weeks to reduce noisy week-specific estimates.

Result: overall MAE 18.3405; SJ 23.8942; IQ 7.2332.

Selection score: SJ 22.0827; IQ 7.1171.

Decision: Not selected for the additive submission.

### a07_week_median_profile_pm2

Rationale: Try a slightly smoother seasonal median while staying transparent and low capacity.

Result: overall MAE 18.3069; SJ 23.8425; IQ 7.2356.

Selection score: SJ 22.0263; IQ 7.1173.

Decision: Not selected for the additive submission.

### a08_recent5_week_median_profile

Rationale: Use only the latest five training years for the seasonal median to test recency bias.

Result: overall MAE 18.7212; SJ 24.3858; IQ 7.3918.

Selection score: SJ 22.1874; IQ 7.2728.

Decision: Not selected for the additive submission.

### a10_calendar_lgbm_l1_recent_weighted_soft

Rationale: Use a softer recency weight to see whether a smaller future bias keeps all-year stability.

Result: overall MAE 18.0048; SJ 23.4135; IQ 7.1875.

Selection score: SJ 21.6481; IQ 7.0779.

Decision: Not selected for the additive submission.

### a11_calendar_lgbm_l1_recent_weighted_strong

Rationale: Use a stronger recency weight after SJ improved on the latest validation years.

Result: overall MAE 18.4591; SJ 24.0829; IQ 7.2115.

Selection score: SJ 21.9206; IQ 7.0832.

Decision: Not selected for the additive submission.

### a12_calendar_lgbm_l1_recent_weighted_curved

Rationale: Concentrate extra weight on the newest years without fully discarding older seasons.

Result: overall MAE 18.5377; SJ 24.1683; IQ 7.2764.

Selection score: SJ 21.9947; IQ 7.1416.

Decision: Not selected for the additive submission.

### a13_recent8_calendar_lgbm_l1_raw

Rationale: Fit the same calendar model using only the latest eight training years as a recency stress test.

Result: overall MAE 17.8670; SJ 23.1659; IQ 7.2692.

Selection score: SJ 21.1877; IQ 7.1798.

Decision: Not selected for the additive submission.

### a14_recent6_calendar_lgbm_l1_raw

Rationale: Check whether a shorter six-year calendar window improves future alignment without too little data.

Result: overall MAE 18.0072; SJ 23.4519; IQ 7.1178.

Selection score: SJ 21.2120; IQ 6.9829.

Decision: Not selected for the additive submission.

### a15_recent7_calendar_lgbm_l1_raw

Rationale: Test the neighboring seven-year window around the current eight-year winner.

Result: overall MAE 17.5929; SJ 22.7644; IQ 7.2500.

Selection score: SJ 20.9514; IQ 7.1548.

Decision: Not selected for the additive submission.

### a16_recent9_calendar_lgbm_l1_raw

Rationale: Test the neighboring nine-year window around the current eight-year winner.

Result: overall MAE 17.8654; SJ 23.1971; IQ 7.2019.

Selection score: SJ 21.2096; IQ 7.0923.

Decision: Not selected for the additive submission.

### a17_recent10_calendar_lgbm_l1_raw

Rationale: Check whether a ten-year window keeps the recency benefit with more seasonal history.

Result: overall MAE 17.7716; SJ 23.0565; IQ 7.2019.

Selection score: SJ 21.0852; IQ 7.0923.

Decision: Not selected for the additive submission.

### a18_recent12_calendar_lgbm_l1_raw

Rationale: Check whether a wider twelve-year window remains better than using all years.

Result: overall MAE 18.0609; SJ 23.4904; IQ 7.2019.

Selection score: SJ 21.6413; IQ 7.0923.

Decision: Not selected for the additive submission.

### a19_calendar_lgbm_l1_harmonic2

Rationale: Add second-harmonic seasonal terms to capture asymmetry while staying calendar-only.

Result: overall MAE 18.1170; SJ 23.5709; IQ 7.2091.

Selection score: SJ 21.8809; IQ 7.1002.

Decision: Not selected for the additive submission.

### a20_recent8_calendar_lgbm_l1_harmonic2

Rationale: Combine the best recent-window idea with second-harmonic seasonal terms.

Result: overall MAE 17.8325; SJ 23.1130; IQ 7.2716.

Selection score: SJ 21.1435; IQ 7.1815.

Decision: Not selected for the additive submission.

### a21_recent4_calendar_lgbm_l1_raw

Rationale: Test a shorter four-year recent window after IQ improved with six recent years.

Result: overall MAE 18.1731; SJ 23.7764; IQ 6.9663.

Selection score: SJ 21.0454; IQ 6.8005.

Decision: Not selected for the additive submission.

### a22_recent5_calendar_lgbm_l1_raw

Rationale: Test a five-year recent calendar model between the direct recent profile and six-year winner.

Result: overall MAE 18.3045; SJ 23.8462; IQ 7.2212.

Selection score: SJ 21.7505; IQ 7.1173.

Decision: Not selected for the additive submission.

### a23_recent6_calendar_lgbm_l1_harmonic2

Rationale: Check whether the IQ-favored six-year window benefits from a richer seasonal shape.

Result: overall MAE 17.9207; SJ 23.3281; IQ 7.1058.

Selection score: SJ 21.1499; IQ 6.9659.

Decision: Not selected for the additive submission.

### a24_recent7_calendar_lgbm_l1_harmonic2

Rationale: Check whether the SJ-favored seven-year window benefits from a richer seasonal shape.

Result: overall MAE 17.5697; SJ 22.7296; IQ 7.2500.

Selection score: SJ 20.9371; IQ 7.1534.

Decision: Not selected for the additive submission.

### a25_recent2_calendar_lgbm_l1_raw

Rationale: Test a very short two-year window to bound the IQ recency effect.

Result: overall MAE 20.4631; SJ 27.2788; IQ 6.8317.

Selection score: SJ 23.2288; IQ 6.7885.

Decision: Not selected for the additive submission.

### a26_recent3_calendar_lgbm_l1_raw

Rationale: Test a three-year window just below the current IQ four-year winner.

Result: overall MAE 18.5433; SJ 24.1899; IQ 7.2500.

Selection score: SJ 22.1123; IQ 7.0481.

Decision: Not selected for the additive submission.

### a27_recent4_calendar_lgbm_l1_harmonic2

Rationale: Check whether the IQ-favored four-year window benefits from second-harmonic seasonality.

Result: overall MAE 18.1466; SJ 23.7404; IQ 6.9591.

Selection score: SJ 21.0115; IQ 6.7767.

Decision: Not selected for the additive submission.

### a28_recent7_calendar_lgbm_l1_harmonic3

Rationale: Test one extra harmonic on the current SJ seven-year harmonic winner.

Result: overall MAE 17.5801; SJ 22.7440; IQ 7.2524.

Selection score: SJ 20.9444; IQ 7.1608.

Decision: Not selected for the additive submission.

### a29_recent2_calendar_lgbm_l1_harmonic2

Rationale: Check whether the IQ all-year-strong two-year window benefits from second-harmonic seasonality.

Result: overall MAE 20.3654; SJ 27.1346; IQ 6.8269.

Selection score: SJ 23.1322; IQ 6.7779.

Decision: Not selected for the additive submission.

### a30_recent3_calendar_lgbm_l1_harmonic2

Rationale: Check whether the three-year window gains enough shape from second-harmonic seasonality to compete with the four-year winner.

Result: overall MAE 18.5369; SJ 24.1526; IQ 7.3053.

Selection score: SJ 22.0819; IQ 7.1012.

Decision: Not selected for the additive submission.

### a31_recent4_calendar_lgbm_l1_harmonic3

Rationale: Test whether one extra harmonic improves the current IQ four-year harmonic winner or starts overfitting.

Result: overall MAE 18.0561; SJ 23.6058; IQ 6.9567.

Selection score: SJ 20.9159; IQ 6.7923.

Decision: Not selected for the additive submission.

### a32_recent7_calendar_lgbm_l1_harmonic2_regularized

Rationale: Retest the current SJ seven-year harmonic winner with a smaller, more regularized tree shape.

Result: overall MAE 17.4920; SJ 22.6274; IQ 7.2212.

Selection score: SJ 20.8137; IQ 7.1274.

Decision: Not selected for the additive submission.

### a33_recent4_calendar_lgbm_l1_harmonic2_regularized

Rationale: Retest the current IQ four-year harmonic winner with a smaller, more regularized tree shape.

Result: overall MAE 18.0112; SJ 23.5625; IQ 6.9087.

Selection score: SJ 20.8625; IQ 6.7269.

Decision: Not selected for the additive submission.

### a34_recent7_calendar_rf_abs_harmonic2

Rationale: Mirror the SJ seven-year harmonic setup with an absolute-error Random Forest to test whether bagging beats boosting here.

Result: overall MAE 17.7388; SJ 23.0757; IQ 7.0649.

Selection score: SJ 21.1593; IQ 6.9690.

Decision: Not selected for the additive submission.

### a35_recent4_calendar_rf_abs_harmonic2

Rationale: Mirror the IQ four-year harmonic setup with an absolute-error Random Forest to test model-family sensitivity.

Result: overall MAE 18.1450; SJ 23.7596; IQ 6.9159.

Selection score: SJ 20.9962; IQ 6.7450.

Decision: Not selected for the additive submission.

### a36_recent2_calendar_rf_abs_harmonic2

Rationale: Test whether Random Forest stabilizes the very short IQ-favored two-year harmonic window.

Result: overall MAE 22.1955; SJ 29.7957; IQ 6.9952.

Selection score: SJ 25.0267; IQ 6.9418.

Decision: Not selected for the additive submission.

### a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized

Rationale: Check whether the SJ seven-year harmonic winner improves with one more step of shrinkage.

Result: overall MAE 17.4223; SJ 22.5312; IQ 7.2043.

Selection score: SJ 20.7724; IQ 7.1200.

Decision: Not selected for the additive submission.

### a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized

Rationale: Check whether the IQ four-year harmonic winner improves with one more step of shrinkage.

Result: overall MAE 17.8950; SJ 23.3906; IQ 6.9038.

Selection score: SJ 20.7436; IQ 6.7351.

Decision: Not selected for the additive submission.

### a39_recent7_calendar_quantile_harmonic2

Rationale: Use a simple median linear seasonal curve for the SJ seven-year harmonic setup.

Result: overall MAE 17.4062; SJ 22.5156; IQ 7.1875.

Selection score: SJ 20.7542; IQ 7.0678.

Decision: Not selected for the additive submission.

### a40_recent4_calendar_quantile_harmonic2

Rationale: Use a simple median linear seasonal curve for the IQ four-year harmonic setup.

Result: overall MAE 17.8862; SJ 23.3546; IQ 6.9495.

Selection score: SJ 20.8280; IQ 6.7772.

Decision: Not selected for the additive submission.

### a41_recent7_calendar_poisson_glm_harmonic2

Rationale: Fit a statistical Poisson count GLM for the SJ seven-year harmonic setup using raw counts.

Result: overall MAE 21.7540; SJ 29.0637; IQ 7.1346.

Selection score: SJ 25.6465; IQ 7.0466.

Decision: Not selected for the additive submission.

### a42_recent4_calendar_poisson_glm_harmonic2

Rationale: Fit a statistical Poisson count GLM for the IQ four-year harmonic setup using raw counts.

Result: overall MAE 21.4014; SJ 28.4904; IQ 7.2236.

Selection score: SJ 24.3423; IQ 7.1075.

Decision: Not selected for the additive submission.

### a43_recent7_calendar_negbin_glm_harmonic2

Rationale: Fit a statistical negative-binomial count GLM for the SJ seven-year harmonic setup to allow overdispersion.

Result: overall MAE 21.6987; SJ 28.9844; IQ 7.1274.

Selection score: SJ 25.5939; IQ 7.0416.

Decision: Not selected for the additive submission.

### a44_recent4_calendar_negbin_glm_harmonic2

Rationale: Fit a statistical negative-binomial count GLM for the IQ four-year harmonic setup to allow overdispersion.

Result: overall MAE 21.2492; SJ 28.2668; IQ 7.2139.

Selection score: SJ 24.1700; IQ 7.0935.

Decision: Not selected for the additive submission.

### a09_calendar_lgbm_pm1_median_blend

Rationale: Average the best flexible calendar learner with the smoothed median profile.

Result: overall MAE 18.2348; SJ 23.7296; IQ 7.2452.

Selection score: SJ 21.9588; IQ 7.1327.

Decision: Not selected for the additive submission.

### a45_iq_lgbm_frontier_blend

Rationale: Blend the two IQ models that split the selection-sensitivity grid to reduce selection-rule brittleness.

Result: overall MAE 18.9215; SJ 24.9856; IQ 6.7933.

Selection score: SJ 21.7187; IQ 6.6923.

Decision: Not selected for the additive submission.

### a46_iq_lgbm_rf_frontier_blend

Rationale: Add the close IQ Random Forest check to the frontier blend as a small model-family hedge.

Result: overall MAE 18.5873; SJ 24.4976; IQ 6.7668.

Selection score: SJ 21.4161; IQ 6.6421.

Decision: Selected for final additive submission city/cities: iq

### a47_sj_quantile_lgbm_frontier_blend

Rationale: Blend the two strongest broad-stability SJ seasonal models to test whether averaging improves the selected quantile model.

Result: overall MAE 17.3630; SJ 22.4495; IQ 7.1899.

Selection score: SJ 20.7050; IQ 7.0853.

Decision: Selected for final additive submission city/cities: sj

### a48_iq_lgbm_rf_frontier_blend_recent_tilt

Rationale: Tilt the IQ frontier blend toward the regularized recent-stable model while keeping the short-window and RF hedge.

Result: overall MAE 18.3165; SJ 24.0709; IQ 6.8077.

Selection score: SJ 21.1578; IQ 6.6663.

Decision: Not selected for the additive submission.

### a49_iq_lgbm_rf_frontier_blend_short_tilt

Rationale: Tilt the IQ frontier blend toward the all-year-strong short-window model without relying on it alone.

Result: overall MAE 18.9551; SJ 25.0397; IQ 6.7861.

Selection score: SJ 21.7393; IQ 6.6858.

Decision: Not selected for the additive submission.

### a50_sj_quantile_lgbm_frontier_blend_quantile_tilt

Rationale: Tilt the SJ frontier blend toward the simpler quantile seasonal model.

Result: overall MAE 17.3790; SJ 22.4651; IQ 7.2067.

Selection score: SJ 20.7218; IQ 7.0928.

Decision: Not selected for the additive submission.

### a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt

Rationale: Tilt the SJ frontier blend toward the regularized tree seasonal model.

Result: overall MAE 17.3694; SJ 22.4615; IQ 7.1851.

Selection score: SJ 20.7149; IQ 7.0950.

Decision: Not selected for the additive submission.

## Final additive selection

| City | Selected iteration | All-year MAE | Recent MAE | Selection MAE | Model type |
| --- | --- | ---: | ---: | ---: | --- |
| iq | a46_iq_lgbm_rf_frontier_blend | 6.7668 | 6.3510 | 6.6421 | blend |
| sj | a47_sj_quantile_lgbm_frontier_blend | 22.4495 | 16.6346 | 20.7050 | blend |

## Selection robustness snapshot

This reports how often each iteration is selected across the sensitivity grid in `selection_sensitivity.csv`.

| City | Selected iteration | Scenarios selected |
| --- | --- | ---: |
| iq | a46_iq_lgbm_rf_frontier_blend | 25 |
| sj | a47_sj_quantile_lgbm_frontier_blend | 23 |
| sj | a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | 2 |

## Best additive single iteration

a47_sj_quantile_lgbm_frontier_blend had the best additive overall MAE at 17.3630.
