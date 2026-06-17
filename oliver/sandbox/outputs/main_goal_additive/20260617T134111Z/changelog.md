# main_goal_additive.py changelog

Run timestamp UTC: 2026-06-17T13:41:11.638947Z

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
| a09_calendar_lgbm_pm1_median_blend | blend | 18.2348 | 23.7296 | 7.2452 |

## Selection score summary

| Iteration | City | All-year MAE | Recent MAE | Selection MAE |
| --- | --- | ---: | ---: | ---: |
| a01_calendar_lgbm_l1_raw | iq | 7.2019 | 6.8365 | 7.0923 |
| a06_week_median_profile_pm1 | iq | 7.2332 | 6.8462 | 7.1171 |
| a07_week_median_profile_pm2 | iq | 7.2356 | 6.8413 | 7.1173 |
| a02_calendar_lgbm_l1_recent_weighted | iq | 7.2356 | 6.8606 | 7.1231 |
| a05_week_mean_profile | iq | 7.2692 | 6.7981 | 7.1279 |
| a09_calendar_lgbm_pm1_median_blend | iq | 7.2452 | 6.8702 | 7.1327 |
| a04_week_median_profile | iq | 7.3606 | 6.9327 | 7.2322 |
| a08_recent5_week_median_profile | iq | 7.3918 | 6.9952 | 7.2728 |
| a03_calendar_lgbm_l1_with_year_index | iq | 8.6346 | 7.5240 | 8.3014 |
| a02_calendar_lgbm_l1_recent_weighted | sj | 23.7079 | 17.1971 | 21.7547 |
| a01_calendar_lgbm_l1_raw | sj | 23.6178 | 17.9231 | 21.9094 |
| a09_calendar_lgbm_pm1_median_blend | sj | 23.7296 | 17.8269 | 21.9588 |
| a07_week_median_profile_pm2 | sj | 23.8425 | 17.7885 | 22.0263 |
| a06_week_median_profile_pm1 | sj | 23.8942 | 17.8558 | 22.0827 |
| a08_recent5_week_median_profile | sj | 24.3858 | 17.0577 | 22.1874 |
| a04_week_median_profile | sj | 24.0601 | 17.9375 | 22.2233 |
| a03_calendar_lgbm_l1_with_year_index | sj | 32.2188 | 19.6635 | 28.4522 |
| a05_week_mean_profile | sj | 31.5204 | 23.7260 | 29.1821 |

## Iteration notes

### a01_calendar_lgbm_l1_raw

Rationale: Re-run the previous selected no-log calendar LightGBM as the additive baseline.

Result: overall MAE 18.1458; SJ 23.6178; IQ 7.2019.

Selection score: SJ 21.9094; IQ 7.0923.

Decision: Selected for final additive submission city/cities: iq

### a02_calendar_lgbm_l1_recent_weighted

Rationale: Favor recent years modestly because the test set is a future holdout.

Result: overall MAE 18.2171; SJ 23.7079; IQ 7.2356.

Selection score: SJ 21.7547; IQ 7.1231.

Decision: Selected for final additive submission city/cities: sj

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

### a09_calendar_lgbm_pm1_median_blend

Rationale: Average the best flexible calendar learner with the smoothed median profile.

Result: overall MAE 18.2348; SJ 23.7296; IQ 7.2452.

Selection score: SJ 21.9588; IQ 7.1327.

Decision: Not selected for the additive submission.

## Final additive selection

| City | Selected iteration | All-year MAE | Recent MAE | Selection MAE | Model type |
| --- | --- | ---: | ---: | ---: | --- |
| iq | a01_calendar_lgbm_l1_raw | 7.2019 | 6.8365 | 7.0923 | lgbm_calendar |
| sj | a02_calendar_lgbm_l1_recent_weighted | 23.7079 | 17.1971 | 21.7547 | lgbm_calendar_recent_weighted |

## Best additive single iteration

a01_calendar_lgbm_l1_raw had the best additive overall MAE at 18.1458.
