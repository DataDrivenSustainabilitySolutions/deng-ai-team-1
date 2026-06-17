# Best-Practice DengAI Approach

This run encodes a risk-aware approach after the forward-validation and adversarial-validation results.

## Guardrails

- San Juan and Iquitos are modeled separately.
- Validation is expanding full-year forward chaining.
- `total_cases` is used only as the supervised label and validation ground truth.
- No case-derived or target-derived columns are allowed in model features.
- Raw `year`, `week_start_date`, and raw `weekofyear` are not model inputs; only cyclic week features are used.
- Fold preprocessing uses only the fold training history for imputation anchors.
- Weather features are accepted for final selection only if they clear a shift-risk MAE margin.

## Candidate Results

| Candidate | City | Uses weather | MAE | Selection MAE |
| --- | --- | ---: | ---: | ---: |
| broad_weather_l1_log | all | True | 18.9712 | 19.2212 |
| broad_weather_l1_log | iq | True | 7.3966 | 7.6466 |
| broad_weather_l1_log | sj | True | 24.7584 | 25.0084 |
| calendar_broad_log_blend | all | True | 18.0601 | 18.3101 |
| calendar_broad_log_blend | iq | True | 7.1058 | 7.3558 |
| calendar_broad_log_blend | sj | True | 23.5373 | 23.7873 |
| calendar_l1_raw | all | False | 18.1458 | 18.1458 |
| calendar_l1_raw | iq | False | 7.2019 | 7.2019 |
| calendar_l1_raw | sj | False | 23.6178 | 23.6178 |

## Risk-Aware Selection

| City | Selected | Validation MAE | Selection MAE | Pure validation best |
| --- | --- | ---: | ---: | --- |
| sj | calendar_l1_raw | 23.6178 | 23.6178 | calendar_broad_log_blend (23.5373) |
| iq | calendar_l1_raw | 7.2019 | 7.2019 | calendar_broad_log_blend (7.1058) |

Risk-aware selected overall validation MAE: 18.1458.

## Adversarial Context

| City | Scenario | Train/test AUC | Interpretation |
| --- | --- | ---: | --- |
| iq | weather_only | 0.739 | meaningful covariate shift |
| iq | weather_plus_seasonality | 0.741 | meaningful covariate shift |
| sj | weather_only | 0.869 | strong covariate shift |
| sj | weather_plus_seasonality | 0.872 | strong covariate shift |

## Interpretation

The pure validation-best candidate is useful evidence, but it is not automatically the safest final choice because adversarial validation shows non-IID train/test weather covariates. With the default margin, the small weather/blend gains do not clear the risk threshold, so the final submission uses the stable calendar-only L1 model for each city.

Generated files:
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/best_practice/approach_summary.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/best_practice/selected_approach_by_city.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/best_practice/fold_scores.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/best_practice/validation_predictions.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/best_practice/submission.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/best_practice/submission_validation_best.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/best_practice/feature_importance.csv`
- `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/outputs/diagnostic/best_practice/test_component_predictions.csv`
