# main_goal_additive.py changelog

Run timestamp UTC: 2026-06-17T13:57:09.826605Z

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
| a09_calendar_lgbm_pm1_median_blend | blend | 18.2348 | 23.7296 | 7.2452 |

## Selection score summary

| Iteration | City | All-year MAE | Recent MAE | Selection MAE |
| --- | --- | ---: | ---: | ---: |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | iq | 6.9087 | 6.3029 | 6.7269 |
| a27_recent4_calendar_lgbm_l1_harmonic2 | iq | 6.9591 | 6.3510 | 6.7767 |
| a29_recent2_calendar_lgbm_l1_harmonic2 | iq | 6.8269 | 6.6635 | 6.7779 |
| a25_recent2_calendar_lgbm_l1_raw | iq | 6.8317 | 6.6875 | 6.7885 |
| a31_recent4_calendar_lgbm_l1_harmonic3 | iq | 6.9567 | 6.4087 | 6.7923 |
| a21_recent4_calendar_lgbm_l1_raw | iq | 6.9663 | 6.4135 | 6.8005 |
| a23_recent6_calendar_lgbm_l1_harmonic2 | iq | 7.1058 | 6.6394 | 6.9659 |
| a14_recent6_calendar_lgbm_l1_raw | iq | 7.1178 | 6.6683 | 6.9829 |
| a26_recent3_calendar_lgbm_l1_raw | iq | 7.2500 | 6.5769 | 7.0481 |
| a10_calendar_lgbm_l1_recent_weighted_soft | iq | 7.1875 | 6.8221 | 7.0779 |
| a11_calendar_lgbm_l1_recent_weighted_strong | iq | 7.2115 | 6.7837 | 7.0832 |
| a01_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a16_recent9_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a17_recent10_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a18_recent12_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a19_calendar_lgbm_l1_harmonic2 | iq | 7.2091 | 6.8462 | 7.1002 |
| a30_recent3_calendar_lgbm_l1_harmonic2 | iq | 7.3053 | 6.6250 | 7.1012 |
| a06_week_median_profile_pm1 | iq | 7.2332 | 6.8462 | 7.1171 |
| a22_recent5_calendar_lgbm_l1_raw | iq | 7.2212 | 6.8750 | 7.1173 |
| a07_week_median_profile_pm2 | iq | 7.2356 | 6.8413 | 7.1173 |
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
| a32_recent7_calendar_lgbm_l1_harmonic2_regularized | sj | 22.6274 | 16.5817 | 20.8137 |
| a33_recent4_calendar_lgbm_l1_harmonic2_regularized | sj | 23.5625 | 14.5625 | 20.8625 |
| a31_recent4_calendar_lgbm_l1_harmonic3 | sj | 23.6058 | 14.6394 | 20.9159 |
| a24_recent7_calendar_lgbm_l1_harmonic2 | sj | 22.7296 | 16.7548 | 20.9371 |
| a28_recent7_calendar_lgbm_l1_harmonic3 | sj | 22.7440 | 16.7452 | 20.9444 |
| a15_recent7_calendar_lgbm_l1_raw | sj | 22.7644 | 16.7212 | 20.9514 |
| a27_recent4_calendar_lgbm_l1_harmonic2 | sj | 23.7404 | 14.6442 | 21.0115 |
| a21_recent4_calendar_lgbm_l1_raw | sj | 23.7764 | 14.6731 | 21.0454 |
| a17_recent10_calendar_lgbm_l1_raw | sj | 23.0565 | 16.4856 | 21.0852 |
| a20_recent8_calendar_lgbm_l1_harmonic2 | sj | 23.1130 | 16.5481 | 21.1435 |
| a23_recent6_calendar_lgbm_l1_harmonic2 | sj | 23.3281 | 16.0673 | 21.1499 |
| a13_recent8_calendar_lgbm_l1_raw | sj | 23.1659 | 16.5721 | 21.1877 |
| a16_recent9_calendar_lgbm_l1_raw | sj | 23.1971 | 16.5721 | 21.2096 |
| a14_recent6_calendar_lgbm_l1_raw | sj | 23.4519 | 15.9856 | 21.2120 |
| a18_recent12_calendar_lgbm_l1_raw | sj | 23.4904 | 17.3269 | 21.6413 |
| a10_calendar_lgbm_l1_recent_weighted_soft | sj | 23.4135 | 17.5288 | 21.6481 |
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

Decision: Selected for final additive submission city/cities: sj

### a33_recent4_calendar_lgbm_l1_harmonic2_regularized

Rationale: Retest the current IQ four-year harmonic winner with a smaller, more regularized tree shape.

Result: overall MAE 18.0112; SJ 23.5625; IQ 6.9087.

Selection score: SJ 20.8625; IQ 6.7269.

Decision: Selected for final additive submission city/cities: iq

### a09_calendar_lgbm_pm1_median_blend

Rationale: Average the best flexible calendar learner with the smoothed median profile.

Result: overall MAE 18.2348; SJ 23.7296; IQ 7.2452.

Selection score: SJ 21.9588; IQ 7.1327.

Decision: Not selected for the additive submission.

## Final additive selection

| City | Selected iteration | All-year MAE | Recent MAE | Selection MAE | Model type |
| --- | --- | ---: | ---: | ---: | --- |
| iq | a33_recent4_calendar_lgbm_l1_harmonic2_regularized | 6.9087 | 6.3029 | 6.7269 | lgbm_calendar_recent_window |
| sj | a32_recent7_calendar_lgbm_l1_harmonic2_regularized | 22.6274 | 16.5817 | 20.8137 | lgbm_calendar_recent_window |

## Best additive single iteration

a32_recent7_calendar_lgbm_l1_harmonic2_regularized had the best additive overall MAE at 17.4920.
