# dead channels

**Question:** Why does the CIFAR-10 MobileNetV2 show a spread of 75,033?

Almost half its channels are switched off. `dead_channels_from_weights.py` counts near-zero channels from the weights. `dead_channels_on_images.py` runs the 10,000 CIFAR-10 test images, finds the channels whose output never changes, replaces them by a constant and checks accuracy, with random channels as the control.

## Run

From the `code/` folder:

    python -m experiments.dead_channels.dead_channels_from_weights
    python -m experiments.dead_channels.dead_channels_on_images --data-root data

Time: 5 seconds; 13 minutes on CPU.

## Output

Written to `results/dead_channels/`:

- `exp2b_*.csv`: counts from the weights
- `exp2c_channels.csv`, `exp2c_ablation.csv`, `exp2c_mobilenet_data.json`: the run on images

## Where it is used

Section 6.7 (called exp2b and exp2c).

## Checked?

Rerun 16 Nov 2026: all files identical.

Old name: `exp2b_dead_channels.py`, `exp2c_mobilenet_data.py`.
