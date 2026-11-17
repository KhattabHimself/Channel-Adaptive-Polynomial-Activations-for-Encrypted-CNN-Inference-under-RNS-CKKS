# interval formula check

**Question:** The interval formula fixed in advance was slightly off. Does the exact one change any conclusion?

Recomputes spread, rotation and the time saving with the exact formula. It reuses the timing file of `grouping_time`, so run that first. Its second verdict file says CONTROL MISMATCH. That is expected: it reuses a check from `grouping_time` with different degrees on purpose.

## Run

From the `code/` folder:

    python -m experiments.interval_formula_check.interval_formula_check

Time: 1 minute.

## Output

Written to `results/interval_formula_check/`:

- `exp4b_interval_scale.csv`, `exp4b_time_saving.csv`, `exp4b_bucket_degrees_bnout.csv`

## Where it is used

Section 6.8 (called exp4b).

## Checked?

Rerun 16 Nov 2026: all files identical.

Old name: `exp4b_interval_scale.py`.
