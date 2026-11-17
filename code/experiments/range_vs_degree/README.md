# range vs degree

**Question:** For a polynomial activation, which matters more: the interval it is fitted on, or its degree?

Plain PyTorch, no encryption. Seven ways of choosing the interval against four degrees, on three networks. Ranges come from 2,000 training images, accuracy from all 10,000 test images. `input_tails.py` measures how far the activation inputs are from a bell curve, which explains the result.

## Run

From the `code/` folder:

    python -m experiments.range_vs_degree.range_vs_degree --calib-split train --data-root data
    python -m experiments.range_vs_degree.input_tails --data-root data --targets cifar10_resnet20 cifar10_resnet56

Time: about 3 hours on this laptop's CPU for all three networks (30 minutes on a GPU); 1 minute.

## Output

Written to `results/range_vs_degree/`:

- `exp3_range_grid_train.csv`: **the numbers the thesis uses**
- `exp3_range_grid.csv`: the first run, calibrated on test images. Not used
- `exp3_range_grid_smoke.csv`: a 1,000-image trial. Not a result
- `exp3b_tails*.csv`: the tail measurements
- `check_cpu_rerun_2026-11-16/`: the full rerun on the laptop, made during verification

## Where it is used

Sections 6.11 and 6.12 (called exp3 and exp3b).

## Checked?

Rerun 16 Nov 2026 on CPU, all three networks: the numbers that carry the conclusion agree with the saved GPU run within 0.2 points. The two percentile rules use random sampling with no fixed seed and can move by a few points at degree 7. They are at chance from degree 15 up either way.

Old name: `exp3_range_selection.py`, `exp3b_tails.py`.
