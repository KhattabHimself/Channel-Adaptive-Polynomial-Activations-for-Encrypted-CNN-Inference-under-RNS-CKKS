# activation accuracy

**Question:** How deep does a polynomial activation have to be for ResNet-20 to keep its accuracy?

Plain PyTorch, no encryption. `per_channel_degree_accuracy.py` uses one polynomial per channel. `composite_sequences.py` uses compositions of small sign polynomials (57 sequences).

## Run

From the `code/` folder:

    python -m experiments.activation_accuracy.per_channel_degree_accuracy --budgets 0.01 0.003 --intervals 3 minmax
    python -m experiments.activation_accuracy.per_channel_degree_accuracy --approximant composite --compositions 1 2 3 4 5
    python -m experiments.activation_accuracy.composite_sequences

Time: 5 minutes, 1 minute, 39 minutes on CPU.

## Output

Written to `results/activation_accuracy/`:

- `accuracy_sweep.csv`, `accuracy_composite.csv`, `oa_composite.csv`, `oa_composite_frontier.csv`
- `check_train_calibration/`: reruns with training-image calibration, made during verification

## Where it is used

Section 6.10.

## Checked?

Rerun 16 Nov 2026 with the ranges taken from 2,000 **training** images (now the default; `--calib-split test` gives the old behaviour). The files in this folder are those reruns and are what the thesis reports: 30 of 57 sequences within one point, 7-15-15 the cheapest (0.66 points lost), best single-polynomial setting 2.89 points lost. The August runs calibrated on the first 2,000 test images, which were also scored; those files are kept in `superseded_test_calibration/` and are not used. No conclusion changed between the two. See VERIFICATION_REPORT.md, problem 2.

Old name: `phase3/accuracy_sweep.py`, `phase5/oa_composite.py`.
