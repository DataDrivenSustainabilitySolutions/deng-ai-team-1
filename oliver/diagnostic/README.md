# Diagnostic Scripts

`adversarial_validation.py` checks whether the raw DengAI train and test feature
distributions are distinguishable.

`best_practice_approach.py` runs a risk-aware, calendar-first modeling approach
based on the current validation and adversarial-diagnostic evidence.

Run from the repository root:

```bash
python3 oliver/diagnostic/adversarial_validation.py
python3 oliver/diagnostic/best_practice_approach.py
```

Defaults:

- train features: `data/dengue_features_train.csv`
- test features: `data/dengue_features_test.csv`
- outputs: `oliver/outputs/diagnostic`
- folds: `5`
- random state: `42`

The script runs city-specific domain classifiers with `train = 1` and `test = 0`.
It excludes absolute time fields from classifier inputs and never loads labels or
uses `total_cases`.

The best-practice approach writes its generated outputs under
`oliver/outputs/diagnostic/best_practice/`.
