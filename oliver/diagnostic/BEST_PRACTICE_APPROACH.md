# Best-Practice Approach

This folder now contains two complementary diagnostics:

- `adversarial_validation.py`: checks whether train and test covariates are distinguishable.
- `best_practice_approach.py`: runs a risk-aware, calendar-first modeling approach informed by the validation and adversarial results.

Run from the repository root:

```bash
python3 oliver/diagnostic/best_practice_approach.py
```

The approach is deliberately conservative:

- Model `sj` and `iq` separately.
- Use expanding full-year validation.
- Use `total_cases` only as the supervised label and validation ground truth.
- Keep all case-derived features out of model inputs.
- Keep raw `year`, `week_start_date`, and raw `weekofyear` out of model inputs; use only cyclic week features.
- Use fold-safe forward-fill plus training-fold medians for weather-feature imputation.
- Treat weather features as useful but risky because adversarial validation found train/test covariate shift.
- Select the stable calendar model unless a weather-using candidate clears a configurable MAE risk margin.

Generated outputs are written to `oliver/outputs/diagnostic/best_practice/`.
