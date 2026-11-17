# level saving search

**Question:** If channel demand did differ between activations, could some grouping save levels?

Tries every possible grouping of 4 to 9 made-up channels and scores each by levels. 1,488 small cases. Made-up degrees, not from a network.

## Run

From the `code/` folder:

    python -m experiments.level_saving_search.level_saving_search

Time: 4 seconds.

## Output

Written to `results/level_saving_search/`:

- `level_objective_probe.csv`: one row per case

## Where it is used

Section 6.4.

## Checked?

Rerun 16 Nov 2026: output identical.

Old name: `phase5/level_objective_probe.py`.
