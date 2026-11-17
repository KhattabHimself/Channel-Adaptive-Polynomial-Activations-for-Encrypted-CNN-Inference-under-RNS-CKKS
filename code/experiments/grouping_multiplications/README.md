# grouping multiplications

**Question:** How many multiplications does grouping channels by degree save?

Counts only, no encryption. The solver itself is `code/src/grouping_solver.py`.

## Run

From the `code/` folder:

    python -m src.grouping_solver --validate-synthetic
    python -m experiments.grouping_multiplications.multiplications_real_networks
    python -m experiments.grouping_multiplications.synthetic_grouped_networks --trials 20

Time: 1 second, 30 seconds, 3 minutes.

## Output

Written to `results/grouping_multiplications/`:

- `synthetic_validation.csv`: solver against exhaustive search, 720 cases
- `real_degrees_placement.csv`: ResNet-20 and MobileNetV2 with degrees from the real weights
- `grouped_wrn_synthetic.csv`: made-up degrees on wider networks. Supporting evidence only

## Where it is used

Section 6.13.

## Checked?

Rerun 16 Nov 2026: all three files identical. Note `--trials 20`: the default is 40 and gives twice the rows.

Old name: `phase5/dp_solver.py`, `phase5/real_degrees.py`, `phase5/synthetic_grouped_wrn.py`.
