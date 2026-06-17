# main_goal_additive.py changelog

Run timestamp UTC: 2026-06-17T14:20:50.652678Z

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
| a01_calendar_lgbm_l1_raw | lgbm_calendar | 18.2284 | 23.7356 | 7.2139 |
| a02_calendar_lgbm_l1_recent_weighted | lgbm_calendar_recent_weighted | 18.2676 | 23.8041 | 7.1947 |
| a03_calendar_lgbm_l1_with_year_index | lgbm_calendar_year_index | 24.1490 | 31.9111 | 8.6250 |
| a04_week_median_profile | week_profile | 18.4936 | 24.0601 | 7.3606 |
| a05_week_mean_profile | week_profile | 23.4367 | 31.5204 | 7.2692 |
| a06_week_median_profile_pm1 | week_profile | 18.3405 | 23.8942 | 7.2332 |
| a07_week_median_profile_pm2 | week_profile | 18.3069 | 23.8425 | 7.2356 |
| a08_recent5_week_median_profile | week_profile | 18.7212 | 24.3858 | 7.3918 |
| a10_calendar_lgbm_l1_recent_weighted_soft | lgbm_calendar_recent_weighted | 18.0625 | 23.4880 | 7.2115 |
| a11_calendar_lgbm_l1_recent_weighted_strong | lgbm_calendar_recent_weighted | 18.5409 | 24.1995 | 7.2236 |
| a12_calendar_lgbm_l1_recent_weighted_curved | lgbm_calendar_recent_weighted | 18.5248 | 24.1442 | 7.2861 |
| a13_recent8_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.9327 | 23.2632 | 7.2716 |
| a14_recent6_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.0641 | 23.5505 | 7.0913 |
| a15_recent7_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.6635 | 22.8678 | 7.2548 |
| a16_recent9_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.9207 | 23.2740 | 7.2139 |
| a17_recent10_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 17.8389 | 23.1514 | 7.2139 |
| a18_recent12_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.1514 | 23.6202 | 7.2139 |
| a19_calendar_lgbm_l1_harmonic2 | lgbm_calendar | 18.1691 | 23.6575 | 7.1923 |
| a20_recent8_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.8838 | 23.2007 | 7.2500 |
| a21_recent4_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.2788 | 23.9483 | 6.9399 |
| a22_recent5_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.3662 | 23.9483 | 7.2019 |
| a23_recent6_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.9824 | 23.4291 | 7.0889 |
| a24_recent7_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 17.5865 | 22.7656 | 7.2284 |
| a25_recent2_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 20.6386 | 27.5325 | 6.8510 |
| a26_recent3_calendar_lgbm_l1_raw | lgbm_calendar_recent_window | 18.6282 | 24.3173 | 7.2500 |
| a27_recent4_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.1987 | 23.8257 | 6.9447 |
| a28_recent7_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 17.5905 | 22.7668 | 7.2380 |
| a29_recent2_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 20.0449 | 26.6166 | 6.9014 |
| a30_recent3_calendar_lgbm_l1_harmonic2 | lgbm_calendar_recent_window | 18.5585 | 24.1923 | 7.2909 |
| a31_recent4_calendar_lgbm_l1_harmonic3 | lgbm_calendar_recent_window | 18.0921 | 23.6647 | 6.9471 |
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 17.5144 | 22.6647 | 7.2139 |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | lgbm_calendar_recent_window | 18.1170 | 23.7236 | 6.9038 |
| a34_recent7_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 17.7668 | 23.1130 | 7.0745 |
| a35_recent4_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 18.1787 | 23.8137 | 6.9087 |
| a36_recent2_calendar_rf_abs_harmonic2 | rf_calendar_recent_window | 22.2981 | 29.9579 | 6.9784 |
| a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized | lgbm_calendar_recent_window | 17.3966 | 22.4808 | 7.2284 |
| a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | lgbm_calendar_recent_window | 17.8974 | 23.3966 | 6.8990 |
| a39_recent7_calendar_quantile_harmonic2 | quantile_calendar_recent_window | 17.4062 | 22.5156 | 7.1875 |
| a40_recent4_calendar_quantile_harmonic2 | quantile_calendar_recent_window | 17.8862 | 23.3546 | 6.9495 |
| a41_recent7_calendar_poisson_glm_harmonic2 | glm_poisson_calendar_recent_window | 21.7540 | 29.0637 | 7.1346 |
| a42_recent4_calendar_poisson_glm_harmonic2 | glm_poisson_calendar_recent_window | 21.4014 | 28.4904 | 7.2236 |
| a43_recent7_calendar_negbin_glm_harmonic2 | glm_negbin_calendar_recent_window | 21.6987 | 28.9844 | 7.1274 |
| a44_recent4_calendar_negbin_glm_harmonic2 | glm_negbin_calendar_recent_window | 21.2492 | 28.2668 | 7.2139 |
| a09_calendar_lgbm_pm1_median_blend | blend | 18.2885 | 23.8041 | 7.2572 |
| a45_iq_lgbm_frontier_blend | blend | 18.7821 | 24.7656 | 6.8149 |
| a46_iq_lgbm_rf_frontier_blend | blend | 18.5080 | 24.3582 | 6.8077 |
| a47_sj_quantile_lgbm_frontier_blend | blend | 17.3413 | 22.4135 | 7.1971 |
| a48_iq_lgbm_rf_frontier_blend_recent_tilt | blend | 18.2941 | 24.0264 | 6.8293 |
| a49_iq_lgbm_rf_frontier_blend_short_tilt | blend | 18.8149 | 24.8149 | 6.8149 |
| a50_sj_quantile_lgbm_frontier_blend_quantile_tilt | blend | 17.3622 | 22.4399 | 7.2067 |
| a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt | blend | 17.3654 | 22.4435 | 7.2091 |

## Selection score summary

| Iteration | City | All-year MAE | Recent MAE | Selection MAE |
| --- | --- | ---: | ---: | ---: |
| a46_iq_lgbm_rf_frontier_blend | iq | 6.8077 | 6.4038 | 6.6865 |
| a48_iq_lgbm_rf_frontier_blend_recent_tilt | iq | 6.8293 | 6.3798 | 6.6945 |
| a49_iq_lgbm_rf_frontier_blend_short_tilt | iq | 6.8149 | 6.4471 | 6.7046 |
| a45_iq_lgbm_frontier_blend | iq | 6.8149 | 6.4663 | 6.7103 |
| a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | iq | 6.8990 | 6.3510 | 6.7346 |
| a35_recent4_calendar_rf_abs_harmonic2 | iq | 6.9087 | 6.3510 | 6.7413 |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | iq | 6.9038 | 6.3702 | 6.7438 |
| a21_recent4_calendar_lgbm_l1_raw | iq | 6.9399 | 6.3798 | 6.7719 |
| a40_recent4_calendar_quantile_harmonic2 | iq | 6.9495 | 6.3750 | 6.7772 |
| a27_recent4_calendar_lgbm_l1_harmonic2 | iq | 6.9447 | 6.4087 | 6.7839 |
| a31_recent4_calendar_lgbm_l1_harmonic3 | iq | 6.9471 | 6.4423 | 6.7957 |
| a25_recent2_calendar_lgbm_l1_raw | iq | 6.8510 | 6.6827 | 6.8005 |
| a29_recent2_calendar_lgbm_l1_harmonic2 | iq | 6.9014 | 6.6346 | 6.8214 |
| a36_recent2_calendar_rf_abs_harmonic2 | iq | 6.9784 | 6.7740 | 6.9171 |
| a14_recent6_calendar_lgbm_l1_raw | iq | 7.0913 | 6.6202 | 6.9500 |
| a23_recent6_calendar_lgbm_l1_harmonic2 | iq | 7.0889 | 6.6538 | 6.9584 |
| a34_recent7_calendar_rf_abs_harmonic2 | iq | 7.0745 | 6.7740 | 6.9844 |
| a43_recent7_calendar_negbin_glm_harmonic2 | iq | 7.1274 | 6.8413 | 7.0416 |
| a41_recent7_calendar_poisson_glm_harmonic2 | iq | 7.1346 | 6.8413 | 7.0466 |
| a26_recent3_calendar_lgbm_l1_raw | iq | 7.2500 | 6.5865 | 7.0510 |
| a39_recent7_calendar_quantile_harmonic2 | iq | 7.1875 | 6.7885 | 7.0678 |
| a50_sj_quantile_lgbm_frontier_blend_quantile_tilt | iq | 7.2067 | 6.8221 | 7.0913 |
| a30_recent3_calendar_lgbm_l1_harmonic2 | iq | 7.2909 | 6.6298 | 7.0925 |
| a19_calendar_lgbm_l1_harmonic2 | iq | 7.1923 | 6.8606 | 7.0928 |
| a02_calendar_lgbm_l1_recent_weighted | iq | 7.1947 | 6.8558 | 7.0930 |
| a44_recent4_calendar_negbin_glm_harmonic2 | iq | 7.2139 | 6.8125 | 7.0935 |
| a22_recent5_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8413 | 7.0938 |
| a47_sj_quantile_lgbm_frontier_blend | iq | 7.1971 | 6.8558 | 7.0947 |
| a11_calendar_lgbm_l1_recent_weighted_strong | iq | 7.2236 | 6.8029 | 7.0974 |
| a10_calendar_lgbm_l1_recent_weighted_soft | iq | 7.2115 | 6.8413 | 7.1005 |
| a42_recent4_calendar_poisson_glm_harmonic2 | iq | 7.2236 | 6.8365 | 7.1075 |
| a01_calendar_lgbm_l1_raw | iq | 7.2139 | 6.8654 | 7.1094 |
| a16_recent9_calendar_lgbm_l1_raw | iq | 7.2139 | 6.8654 | 7.1094 |
| a17_recent10_calendar_lgbm_l1_raw | iq | 7.2139 | 6.8654 | 7.1094 |
| a18_recent12_calendar_lgbm_l1_raw | iq | 7.2139 | 6.8654 | 7.1094 |
| a06_week_median_profile_pm1 | iq | 7.2332 | 6.8462 | 7.1171 |
| a07_week_median_profile_pm2 | iq | 7.2356 | 6.8413 | 7.1173 |
| a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt | iq | 7.2091 | 6.9087 | 7.1190 |
| a05_week_mean_profile | iq | 7.2692 | 6.7981 | 7.1279 |
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | iq | 7.2139 | 6.9375 | 7.1310 |
| a24_recent7_calendar_lgbm_l1_harmonic2 | iq | 7.2284 | 6.9327 | 7.1397 |
| a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized | iq | 7.2284 | 6.9423 | 7.1425 |
| a09_calendar_lgbm_pm1_median_blend | iq | 7.2572 | 6.8894 | 7.1469 |
| a12_calendar_lgbm_l1_recent_weighted_curved | iq | 7.2861 | 6.8269 | 7.1483 |
| a28_recent7_calendar_lgbm_l1_harmonic3 | iq | 7.2380 | 6.9471 | 7.1507 |
| a15_recent7_calendar_lgbm_l1_raw | iq | 7.2548 | 6.9471 | 7.1625 |
| a20_recent8_calendar_lgbm_l1_harmonic2 | iq | 7.2500 | 6.9760 | 7.1678 |
| a13_recent8_calendar_lgbm_l1_raw | iq | 7.2716 | 6.9808 | 7.1844 |
| a04_week_median_profile | iq | 7.3606 | 6.9327 | 7.2322 |
| a08_recent5_week_median_profile | iq | 7.3918 | 6.9952 | 7.2728 |
| a03_calendar_lgbm_l1_with_year_index | iq | 8.6250 | 7.5096 | 8.2904 |
| a47_sj_quantile_lgbm_frontier_blend | sj | 22.4135 | 16.6106 | 20.6726 |
| a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt | sj | 22.4435 | 16.6154 | 20.6951 |
| a50_sj_quantile_lgbm_frontier_blend_quantile_tilt | sj | 22.4399 | 16.6250 | 20.6954 |
| a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized | sj | 22.4808 | 16.6154 | 20.7212 |
| a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | sj | 23.3966 | 14.5433 | 20.7406 |
| a39_recent7_calendar_quantile_harmonic2 | sj | 22.5156 | 16.6442 | 20.7542 |
| a40_recent4_calendar_quantile_harmonic2 | sj | 23.3546 | 14.9327 | 20.8280 |
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | sj | 22.6647 | 16.6346 | 20.8556 |
| a24_recent7_calendar_lgbm_l1_harmonic2 | sj | 22.7656 | 16.6923 | 20.9436 |
| a28_recent7_calendar_lgbm_l1_harmonic3 | sj | 22.7668 | 16.7212 | 20.9531 |
| a31_recent4_calendar_lgbm_l1_harmonic3 | sj | 23.6647 | 14.6587 | 20.9629 |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | sj | 23.7236 | 14.5865 | 20.9825 |
| a35_recent4_calendar_rf_abs_harmonic2 | sj | 23.8137 | 14.5048 | 21.0210 |
| a15_recent7_calendar_lgbm_l1_raw | sj | 22.8678 | 16.7308 | 21.0267 |
| a27_recent4_calendar_lgbm_l1_harmonic2 | sj | 23.8257 | 14.6490 | 21.0727 |
| a48_iq_lgbm_rf_frontier_blend_recent_tilt | sj | 24.0264 | 14.3029 | 21.1094 |
| a17_recent10_calendar_lgbm_l1_raw | sj | 23.1514 | 16.4183 | 21.1315 |
| a21_recent4_calendar_lgbm_l1_raw | sj | 23.9483 | 14.6394 | 21.1556 |
| a23_recent6_calendar_lgbm_l1_harmonic2 | sj | 23.4291 | 15.9760 | 21.1931 |
| a20_recent8_calendar_lgbm_l1_harmonic2 | sj | 23.2007 | 16.5096 | 21.1934 |
| a34_recent7_calendar_rf_abs_harmonic2 | sj | 23.1130 | 16.7163 | 21.1940 |
| a16_recent9_calendar_lgbm_l1_raw | sj | 23.2740 | 16.5240 | 21.2490 |
| a13_recent8_calendar_lgbm_l1_raw | sj | 23.2632 | 16.5817 | 21.2588 |
| a14_recent6_calendar_lgbm_l1_raw | sj | 23.5505 | 15.9423 | 21.2680 |
| a46_iq_lgbm_rf_frontier_blend | sj | 24.3582 | 14.1298 | 21.2897 |
| a45_iq_lgbm_frontier_blend | sj | 24.7656 | 13.9856 | 21.5316 |
| a49_iq_lgbm_rf_frontier_blend_short_tilt | sj | 24.8149 | 13.9327 | 21.5502 |
| a10_calendar_lgbm_l1_recent_weighted_soft | sj | 23.4880 | 17.5288 | 21.7002 |
| a18_recent12_calendar_lgbm_l1_raw | sj | 23.6202 | 17.3365 | 21.7351 |
| a22_recent5_calendar_lgbm_l1_raw | sj | 23.9483 | 16.8221 | 21.8105 |
| a02_calendar_lgbm_l1_recent_weighted | sj | 23.8041 | 17.2212 | 21.8292 |
| a19_calendar_lgbm_l1_harmonic2 | sj | 23.6575 | 17.9038 | 21.9314 |
| a12_calendar_lgbm_l1_recent_weighted_curved | sj | 24.1442 | 16.9038 | 21.9721 |
| a01_calendar_lgbm_l1_raw | sj | 23.7356 | 17.9183 | 21.9904 |
| a11_calendar_lgbm_l1_recent_weighted_strong | sj | 24.1995 | 16.8750 | 22.0022 |
| a09_calendar_lgbm_pm1_median_blend | sj | 23.8041 | 17.8269 | 22.0109 |
| a07_week_median_profile_pm2 | sj | 23.8425 | 17.7885 | 22.0263 |
| a06_week_median_profile_pm1 | sj | 23.8942 | 17.8558 | 22.0827 |
| a30_recent3_calendar_lgbm_l1_harmonic2 | sj | 24.1923 | 17.1971 | 22.0938 |
| a08_recent5_week_median_profile | sj | 24.3858 | 17.0577 | 22.1874 |
| a26_recent3_calendar_lgbm_l1_raw | sj | 24.3173 | 17.2548 | 22.1986 |
| a04_week_median_profile | sj | 24.0601 | 17.9375 | 22.2233 |
| a29_recent2_calendar_lgbm_l1_harmonic2 | sj | 26.6166 | 13.6827 | 22.7364 |
| a25_recent2_calendar_lgbm_l1_raw | sj | 27.5325 | 13.7115 | 23.3862 |
| a44_recent4_calendar_negbin_glm_harmonic2 | sj | 28.2668 | 14.6106 | 24.1700 |
| a42_recent4_calendar_poisson_glm_harmonic2 | sj | 28.4904 | 14.6635 | 24.3423 |
| a36_recent2_calendar_rf_abs_harmonic2 | sj | 29.9579 | 13.9135 | 25.1446 |
| a43_recent7_calendar_negbin_glm_harmonic2 | sj | 28.9844 | 17.6827 | 25.5939 |
| a41_recent7_calendar_poisson_glm_harmonic2 | sj | 29.0637 | 17.6731 | 25.6465 |
| a03_calendar_lgbm_l1_with_year_index | sj | 31.9111 | 19.9135 | 28.3118 |
| a05_week_mean_profile | sj | 31.5204 | 23.7260 | 29.1821 |

## Iteration notes

### a01_calendar_lgbm_l1_raw

Rationale: Re-run the previous selected no-log calendar LightGBM as the additive baseline.

Result: overall MAE 18.2284; SJ 23.7356; IQ 7.2139.

Selection score: SJ 21.9904; IQ 7.1094.

Decision: Not selected for the additive submission.

### a02_calendar_lgbm_l1_recent_weighted

Rationale: Favor recent years modestly because the test set is a future holdout.

Result: overall MAE 18.2676; SJ 23.8041; IQ 7.1947.

Selection score: SJ 21.8292; IQ 7.0930.

Decision: Not selected for the additive submission.

### a03_calendar_lgbm_l1_with_year_index

Rationale: Check whether a simple non-target time trend helps without adding weather complexity.

Result: overall MAE 24.1490; SJ 31.9111; IQ 8.6250.

Selection score: SJ 28.3118; IQ 8.2904.

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

Result: overall MAE 18.0625; SJ 23.4880; IQ 7.2115.

Selection score: SJ 21.7002; IQ 7.1005.

Decision: Not selected for the additive submission.

### a11_calendar_lgbm_l1_recent_weighted_strong

Rationale: Use a stronger recency weight after SJ improved on the latest validation years.

Result: overall MAE 18.5409; SJ 24.1995; IQ 7.2236.

Selection score: SJ 22.0022; IQ 7.0974.

Decision: Not selected for the additive submission.

### a12_calendar_lgbm_l1_recent_weighted_curved

Rationale: Concentrate extra weight on the newest years without fully discarding older seasons.

Result: overall MAE 18.5248; SJ 24.1442; IQ 7.2861.

Selection score: SJ 21.9721; IQ 7.1483.

Decision: Not selected for the additive submission.

### a13_recent8_calendar_lgbm_l1_raw

Rationale: Fit the same calendar model using only the latest eight training years as a recency stress test.

Result: overall MAE 17.9327; SJ 23.2632; IQ 7.2716.

Selection score: SJ 21.2588; IQ 7.1844.

Decision: Not selected for the additive submission.

### a14_recent6_calendar_lgbm_l1_raw

Rationale: Check whether a shorter six-year calendar window improves future alignment without too little data.

Result: overall MAE 18.0641; SJ 23.5505; IQ 7.0913.

Selection score: SJ 21.2680; IQ 6.9500.

Decision: Not selected for the additive submission.

### a15_recent7_calendar_lgbm_l1_raw

Rationale: Test the neighboring seven-year window around the current eight-year winner.

Result: overall MAE 17.6635; SJ 22.8678; IQ 7.2548.

Selection score: SJ 21.0267; IQ 7.1625.

Decision: Not selected for the additive submission.

### a16_recent9_calendar_lgbm_l1_raw

Rationale: Test the neighboring nine-year window around the current eight-year winner.

Result: overall MAE 17.9207; SJ 23.2740; IQ 7.2139.

Selection score: SJ 21.2490; IQ 7.1094.

Decision: Not selected for the additive submission.

### a17_recent10_calendar_lgbm_l1_raw

Rationale: Check whether a ten-year window keeps the recency benefit with more seasonal history.

Result: overall MAE 17.8389; SJ 23.1514; IQ 7.2139.

Selection score: SJ 21.1315; IQ 7.1094.

Decision: Not selected for the additive submission.

### a18_recent12_calendar_lgbm_l1_raw

Rationale: Check whether a wider twelve-year window remains better than using all years.

Result: overall MAE 18.1514; SJ 23.6202; IQ 7.2139.

Selection score: SJ 21.7351; IQ 7.1094.

Decision: Not selected for the additive submission.

### a19_calendar_lgbm_l1_harmonic2

Rationale: Add second-harmonic seasonal terms to capture asymmetry while staying calendar-only.

Result: overall MAE 18.1691; SJ 23.6575; IQ 7.1923.

Selection score: SJ 21.9314; IQ 7.0928.

Decision: Not selected for the additive submission.

### a20_recent8_calendar_lgbm_l1_harmonic2

Rationale: Combine the best recent-window idea with second-harmonic seasonal terms.

Result: overall MAE 17.8838; SJ 23.2007; IQ 7.2500.

Selection score: SJ 21.1934; IQ 7.1678.

Decision: Not selected for the additive submission.

### a21_recent4_calendar_lgbm_l1_raw

Rationale: Test a shorter four-year recent window after IQ improved with six recent years.

Result: overall MAE 18.2788; SJ 23.9483; IQ 6.9399.

Selection score: SJ 21.1556; IQ 6.7719.

Decision: Not selected for the additive submission.

### a22_recent5_calendar_lgbm_l1_raw

Rationale: Test a five-year recent calendar model between the direct recent profile and six-year winner.

Result: overall MAE 18.3662; SJ 23.9483; IQ 7.2019.

Selection score: SJ 21.8105; IQ 7.0938.

Decision: Not selected for the additive submission.

### a23_recent6_calendar_lgbm_l1_harmonic2

Rationale: Check whether the IQ-favored six-year window benefits from a richer seasonal shape.

Result: overall MAE 17.9824; SJ 23.4291; IQ 7.0889.

Selection score: SJ 21.1931; IQ 6.9584.

Decision: Not selected for the additive submission.

### a24_recent7_calendar_lgbm_l1_harmonic2

Rationale: Check whether the SJ-favored seven-year window benefits from a richer seasonal shape.

Result: overall MAE 17.5865; SJ 22.7656; IQ 7.2284.

Selection score: SJ 20.9436; IQ 7.1397.

Decision: Not selected for the additive submission.

### a25_recent2_calendar_lgbm_l1_raw

Rationale: Test a very short two-year window to bound the IQ recency effect.

Result: overall MAE 20.6386; SJ 27.5325; IQ 6.8510.

Selection score: SJ 23.3862; IQ 6.8005.

Decision: Not selected for the additive submission.

### a26_recent3_calendar_lgbm_l1_raw

Rationale: Test a three-year window just below the current IQ four-year winner.

Result: overall MAE 18.6282; SJ 24.3173; IQ 7.2500.

Selection score: SJ 22.1986; IQ 7.0510.

Decision: Not selected for the additive submission.

### a27_recent4_calendar_lgbm_l1_harmonic2

Rationale: Check whether the IQ-favored four-year window benefits from second-harmonic seasonality.

Result: overall MAE 18.1987; SJ 23.8257; IQ 6.9447.

Selection score: SJ 21.0727; IQ 6.7839.

Decision: Not selected for the additive submission.

### a28_recent7_calendar_lgbm_l1_harmonic3

Rationale: Test one extra harmonic on the current SJ seven-year harmonic winner.

Result: overall MAE 17.5905; SJ 22.7668; IQ 7.2380.

Selection score: SJ 20.9531; IQ 7.1507.

Decision: Not selected for the additive submission.

### a29_recent2_calendar_lgbm_l1_harmonic2

Rationale: Check whether the IQ all-year-strong two-year window benefits from second-harmonic seasonality.

Result: overall MAE 20.0449; SJ 26.6166; IQ 6.9014.

Selection score: SJ 22.7364; IQ 6.8214.

Decision: Not selected for the additive submission.

### a30_recent3_calendar_lgbm_l1_harmonic2

Rationale: Check whether the three-year window gains enough shape from second-harmonic seasonality to compete with the four-year winner.

Result: overall MAE 18.5585; SJ 24.1923; IQ 7.2909.

Selection score: SJ 22.0938; IQ 7.0925.

Decision: Not selected for the additive submission.

### a31_recent4_calendar_lgbm_l1_harmonic3

Rationale: Test whether one extra harmonic improves the current IQ four-year harmonic winner or starts overfitting.

Result: overall MAE 18.0921; SJ 23.6647; IQ 6.9471.

Selection score: SJ 20.9629; IQ 6.7957.

Decision: Not selected for the additive submission.

### a32_recent7_calendar_lgbm_l1_harmonic2_regularized

Rationale: Retest the current SJ seven-year harmonic winner with a smaller, more regularized tree shape.

Result: overall MAE 17.5144; SJ 22.6647; IQ 7.2139.

Selection score: SJ 20.8556; IQ 7.1310.

Decision: Not selected for the additive submission.

### a33_recent4_calendar_lgbm_l1_harmonic2_regularized

Rationale: Retest the current IQ four-year harmonic winner with a smaller, more regularized tree shape.

Result: overall MAE 18.1170; SJ 23.7236; IQ 6.9038.

Selection score: SJ 20.9825; IQ 6.7438.

Decision: Not selected for the additive submission.

### a34_recent7_calendar_rf_abs_harmonic2

Rationale: Mirror the SJ seven-year harmonic setup with an absolute-error Random Forest to test whether bagging beats boosting here.

Result: overall MAE 17.7668; SJ 23.1130; IQ 7.0745.

Selection score: SJ 21.1940; IQ 6.9844.

Decision: Not selected for the additive submission.

### a35_recent4_calendar_rf_abs_harmonic2

Rationale: Mirror the IQ four-year harmonic setup with an absolute-error Random Forest to test model-family sensitivity.

Result: overall MAE 18.1787; SJ 23.8137; IQ 6.9087.

Selection score: SJ 21.0210; IQ 6.7413.

Decision: Not selected for the additive submission.

### a36_recent2_calendar_rf_abs_harmonic2

Rationale: Test whether Random Forest stabilizes the very short IQ-favored two-year harmonic window.

Result: overall MAE 22.2981; SJ 29.9579; IQ 6.9784.

Selection score: SJ 25.1446; IQ 6.9171.

Decision: Not selected for the additive submission.

### a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized

Rationale: Check whether the SJ seven-year harmonic winner improves with one more step of shrinkage.

Result: overall MAE 17.3966; SJ 22.4808; IQ 7.2284.

Selection score: SJ 20.7212; IQ 7.1425.

Decision: Not selected for the additive submission.

### a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized

Rationale: Check whether the IQ four-year harmonic winner improves with one more step of shrinkage.

Result: overall MAE 17.8974; SJ 23.3966; IQ 6.8990.

Selection score: SJ 20.7406; IQ 6.7346.

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

Result: overall MAE 18.2885; SJ 23.8041; IQ 7.2572.

Selection score: SJ 22.0109; IQ 7.1469.

Decision: Not selected for the additive submission.

### a45_iq_lgbm_frontier_blend

Rationale: Blend the two IQ models that split the selection-sensitivity grid to reduce selection-rule brittleness.

Result: overall MAE 18.7821; SJ 24.7656; IQ 6.8149.

Selection score: SJ 21.5316; IQ 6.7103.

Decision: Not selected for the additive submission.

### a46_iq_lgbm_rf_frontier_blend

Rationale: Add the close IQ Random Forest check to the frontier blend as a small model-family hedge.

Result: overall MAE 18.5080; SJ 24.3582; IQ 6.8077.

Selection score: SJ 21.2897; IQ 6.6865.

Decision: Selected for final additive submission city/cities: iq

### a47_sj_quantile_lgbm_frontier_blend

Rationale: Blend the two strongest broad-stability SJ seasonal models to test whether averaging improves the selected quantile model.

Result: overall MAE 17.3413; SJ 22.4135; IQ 7.1971.

Selection score: SJ 20.6726; IQ 7.0947.

Decision: Selected for final additive submission city/cities: sj

### a48_iq_lgbm_rf_frontier_blend_recent_tilt

Rationale: Tilt the IQ frontier blend toward the regularized recent-stable model while keeping the short-window and RF hedge.

Result: overall MAE 18.2941; SJ 24.0264; IQ 6.8293.

Selection score: SJ 21.1094; IQ 6.6945.

Decision: Not selected for the additive submission.

### a49_iq_lgbm_rf_frontier_blend_short_tilt

Rationale: Tilt the IQ frontier blend toward the all-year-strong short-window model without relying on it alone.

Result: overall MAE 18.8149; SJ 24.8149; IQ 6.8149.

Selection score: SJ 21.5502; IQ 6.7046.

Decision: Not selected for the additive submission.

### a50_sj_quantile_lgbm_frontier_blend_quantile_tilt

Rationale: Tilt the SJ frontier blend toward the simpler quantile seasonal model.

Result: overall MAE 17.3622; SJ 22.4399; IQ 7.2067.

Selection score: SJ 20.6954; IQ 7.0913.

Decision: Not selected for the additive submission.

### a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt

Rationale: Tilt the SJ frontier blend toward the regularized tree seasonal model.

Result: overall MAE 17.3654; SJ 22.4435; IQ 7.2091.

Selection score: SJ 20.6951; IQ 7.1190.

Decision: Not selected for the additive submission.

## Final additive selection

| City | Selected iteration | All-year MAE | Recent MAE | Selection MAE | Model type |
| --- | --- | ---: | ---: | ---: | --- |
| iq | a46_iq_lgbm_rf_frontier_blend | 6.8077 | 6.4038 | 6.6865 | blend |
| sj | a47_sj_quantile_lgbm_frontier_blend | 22.4135 | 16.6106 | 20.6726 | blend |

## Selection robustness snapshot

This reports how often each iteration is selected across the sensitivity grid in `selection_sensitivity.csv`.

| City | Selected iteration | Scenarios selected |
| --- | --- | ---: |
| iq | a46_iq_lgbm_rf_frontier_blend | 21 |
| iq | a48_iq_lgbm_rf_frontier_blend_recent_tilt | 4 |
| sj | a47_sj_quantile_lgbm_frontier_blend | 23 |
| sj | a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized | 2 |

## Best additive single iteration

a47_sj_quantile_lgbm_frontier_blend had the best additive overall MAE at 17.3413.
