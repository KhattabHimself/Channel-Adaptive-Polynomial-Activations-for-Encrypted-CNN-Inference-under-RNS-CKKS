# sensitivity proxy

**Question:** Is the cheap estimate of a channel's importance (the size of the weights that read it) any good?

On ResNet-20, nudges one channel at a time over 128 images and measures how much the output moves. Reads `invariants_resnet20.csv`, which `python -m src.channel_degrees` writes.

## Run

From the `code/` folder:

    python -m src.channel_degrees
    python -m experiments.sensitivity_proxy.sensitivity_proxy --images 128

Time: 30 seconds.

## Output

Written to `results/sensitivity_proxy/`:

- `proxy_validation_resnet20.csv`: one row per channel
- `invariants_resnet20.csv`: the input table
- `check_fresh_input/`: the same run with a correct input table, made during verification

## Where it is used

Section 6.9.

## Checked?

Rerun 16 Nov 2026 on a freshly generated input table, so all 784 channels are covered (mean correlation within a layer +0.20, positive on 17 of 21 layers). The August result covered 736 channels because its input table came from an older version of the code; that version is in git history and in `check_fresh_input/` you find the first corrected run. Conclusion unchanged: the estimate is weak. See VERIFICATION_REPORT.md, problem 5.

Old name: `phase3/proxy_validation.py`.
