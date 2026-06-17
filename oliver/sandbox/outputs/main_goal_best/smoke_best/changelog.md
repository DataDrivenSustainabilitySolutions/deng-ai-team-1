# Current Best Standalone Run

Run label: `smoke_best`

This run implements only the selected no-log sandbox solution from the additive search.
It uses raw `total_cases` labels for supervised training, but no target-derived input features.
The only model inputs are week-of-year harmonic calendar features.

## City models

| City | Selected iteration | Members |
| --- | --- | --- |
| sj | `a47_sj_quantile_lgbm_frontier_blend` | 0.5 x `a39_recent7_calendar_quantile_harmonic2`, 0.5 x `a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized` |
| iq | `a46_iq_lgbm_rf_frontier_blend` | 0.3333 x `a29_recent2_calendar_lgbm_l1_harmonic2`, 0.3333 x `a33_recent4_calendar_lgbm_l1_harmonic2_regularized`, 0.3333 x `a35_recent4_calendar_rf_abs_harmonic2` |

## Reference comparison

Reference: `/Users/oliverhennhoefer/Code/Github/deng-ai-team-1/oliver/sandbox/outputs/main_goal_additive/20260617T141841Z/submission.csv`
Exact match: `True`
Changed rows: `0`
Max absolute difference: `0`
Mean absolute difference on changed rows: `0.0`
