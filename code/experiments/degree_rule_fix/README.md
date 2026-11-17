# degree rule fix

**Question:** Is there a degree rule that keeps a MobileNetV2 accurate, and what does grouping save under it?

The thesis rule sends the CIFAR-10 MobileNetV2 to chance (see `grouping_accuracy_proxy`). This looks for a rule that works, picks it on images that are not test images, and measures on the full test set what grouping saves. Plain PyTorch for accuracy; activation time from the five OpenFHE timing runs in `results/grouping_time/repeat_runs`.

## Run

From the `code/` folder:

    python -m experiments.degree_rule_fix.degree_rule_fix sweep  --data-root data
    python -m experiments.degree_rule_fix.degree_rule_fix test   --data-root data
    python -m experiments.degree_rule_fix.degree_rule_fix retime --data-root data

Time: about 1.5 hours, 2 hours and 40 minutes on this laptop's CPU.

Splits: calibration = training images 0 to 1,999; validation = training images 40,000 to 41,999; test = the 10,000 test images. The selection rule is in the module docstring and was written before the sweep ran.

## Output

Written to `results/degree_rule_fix/`:

- `sweep_validation.csv`: 25 rules, each as layerwise and grouped, on validation
- `test_accuracy.csv`: the chosen rules on the test set
- `test_saving_same_rule.csv`: **the main table.** Grouping against the layerwise version of the same rule, as packed and after pruning dead channels, with the saving per timing run
- `test_saving_best_layerwise.csv`: grouping against the cheapest layerwise setting that passed the same validation bar

## What it found

The chosen rule (sensitivity normalised by its median within each site, budget 0.02, cap 31, 25% margin; channels whose interval excludes zero run as exact straight lines):

| | layerwise | grouped |
|---|---|---|
| top-1 on the test set (unmodified 94.05%) | 93.32% | 93.37% |
| activation time, as packed | 60.5 s | 44.6 s, **25.6 +/- 3.7% less** |
| after pruning 6,908 dead channels: top-1 | 93.32% | 93.16% |
| activation time | 47.8 s | 43.4 s, **8.5 +/- 4.1% less** |

## Things to know

- The validation images are training images. The network scores 99.95% on them, so losses there are smaller than on the test set.
- `retime` exists because the first version counted the layerwise baseline with 62 ciphertexts and the grouped setting with 67. It recomputes every time with one count for both. The selection did not change. Accuracies were never affected.
- Activation time only. No level is saved by any of this.
- This is the CIFAR-10 MobileNetV2, not the ImageNet network of the time experiment.

## Where it is used

Thesis Appendix D, section D.1.
