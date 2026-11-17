# grouping accuracy proxy

**Question:** Do the polynomial degrees behind the measured time saving give a network that still classifies?

The time saving (`grouping_time`) was measured on the ImageNet MobileNetV2, and there are no ImageNet images on this machine, so that network cannot be scored. This scores the closest network that can be: the CIFAR-10 MobileNetV2, with the same degree rule, the same seven budgets and the same grouping code. Plain PyTorch, no encryption.

**It is a proxy.** Different network, different dataset, 46% dead channels, and only 18 of its 35 activation sites need more than one ciphertext. It says what the degree rule does on one depthwise network. It does not say what the timed network would score.

## Run

From the `code/` folder:

    python -m experiments.grouping_accuracy_proxy.grouping_accuracy_proxy --data-root data --images 2000
    python -m experiments.grouping_accuracy_proxy.grouping_accuracy_proxy --data-root data --cap 31 --floor 15 --margin 0.25 --budgets 0.1 0.03 0.01

Then, from the project folder:

    python code/scripts/time_saving_with_degree_limits.py

Options: `--cap` and `--floor` limit the degrees (defaults 63 and 1, the setting of the time experiment); `--margin` widens the observed range (default 0.05, the setting of the thesis); `--budgets` picks budgets; leave out `--images` for the full test set.

Time: about 15 minutes for the first (2,000 test images) and 70 minutes for the second (full test set, three budgets) on this laptop's CPU.

## Output

Written to `results/grouping_accuracy_proxy/`. The file name carries the settings.

- `..._cap63_first2000.csv`: **the setting of the time experiment.** Chance accuracy at all seven budgets in all three settings.
- `..._cap31_floor15_margin0.25.csv`: **the what-if used in the thesis table**, on all 10,000 test images. Degrees limited to 15 to 31, 25% margin: 90 to 92%.
- `..._cap31_floor15_margin0.25_first2000.csv`: the same on the first 2,000 test images, all seven budgets. It shows that from B = 0.3 up every channel is at degree 15.
- `..._cap31_first2000.csv` and `..._cap31_margin0.25_first2000.csv`: intermediate what-ifs (a cap without a floor). Not quoted in the thesis except through the uniform-degree rows.

Every file starts with rows named `uniform_<d>`: the same degree at every channel, as a reference for what the network can do.

`results/grouping_time/repeat_runs/summary_time_saving_degree_limits.csv` is the time saving of the timed network recomputed under the same degree limits, from the saved groups and the five timing runs. Its first block (floor 1, cap 63) is a control and reproduces `summary_time_saving.csv`.

## What it found

1. With the degrees of the time experiment the network is at chance (8.6 to 10.9%) at every budget, for layerwise, grouped and per-channel degrees alike.
2. With one degree everywhere the same network scores 88.0% at degree 15 (2,000 images, 5% margin), and with a 25% margin 90.0% at degree 15 and 93.3% at degree 31 on the full test set (unmodified: 94.05%). So the degree rule fails, not the network.
3. The rule gives the first activation of each block a much lower degree than the second (largest degree 6 to 12 against 63 at the tightest budget). The ImageNet MobileNetV2 gets the same pattern.
4. With degrees limited to 15 to 31 the network keeps 90 to 92% (full test set). Grouping is up to 0.6 points below the layerwise baseline. The time saving under those limits is 0.6 +/- 1.4% at the tightest budget and at most 11.1 +/- 9.2%.

## Things to know

- The runs with the unlimited degrees are on 2,000 test images. Everything there is at chance, so the sample size does not matter. The what-if is on the full test set.
- The limits 15 and 31 and the 25% margin were chosen after seeing result 1. They are a what-if, not a setting fixed in advance.
- `configure_exact` replaces the fitted line by the exact one for channels whose inputs never cross zero. `PolyAct` limits the kink position to [-1.5, 1.5], which gives a shifted line for always-positive channels beyond that. The CIFAR ResNets have no such channel (`experiments/range_vs_degree/kink_clamp_check.py`); this network does.
- The grouping is checked against `grouping_time.dp_place_shared`: the script stops if its groups do not have the same cost.

## Where it is used

Section 6.14.3 and Table 6.15; discussed in Sections 7.1, 7.2 and 7.5.

## Checked?

New in November 2026. The what-if was run twice, on 2,000 images and on the full test set, and the two agree within 0.7 points at every entry. Nothing else was rerun.
