# level realignment

**Question:** When two ciphertexts at different levels are added, are the spare levels of the richer one thrown away?

Eight made-up channels are split into two groups in two ways. Each group's ciphertext is squared as many times as its activation would use levels, then the two are added in OpenFHE and the level of the sum is read. It does not evaluate a polynomial or a convolution, and it sets no security level. That is all the proof in thesis section 4.8 needs.

## Run

From the `code/` folder:

    docker run --rm -v "<project folder>:/work" -w /work/code thesis-openfhe python -m experiments.level_realignment.level_realignment
    docker run --rm -v "<project folder>:/work" -w /work/code thesis-openfhe python -m experiments.level_realignment.level_realignment --logN 16 --depth 30

Time: under a minute. Needs Docker (see `code/scripts/openfhe_docker`).

## Output

Written to `results/level_realignment/`:

- `exp1_realignment.json`, `exp1_realignment_N16.json`: levels before and after the addition
- `VERDICT_*.txt`: one line each

## Where it is used

Section 6.3 (called exp1).

## Checked?

Rerun 16 Nov 2026: same levels (both groupings end at level 5, both ring sizes).

Old name: `exp1_realignment.py`.
