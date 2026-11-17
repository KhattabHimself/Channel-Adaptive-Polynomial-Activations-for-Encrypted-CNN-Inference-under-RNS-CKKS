# spread and rotation

**Question:** How different are the channels of a layer, and do the expensive channels change between the two activations of a block?

Reads only the saved weights of six pretrained networks. `spread_and_rotation_six_networks.py` is the October version (six networks). `spread_first_networks.py` is the August version (ResNet-20 and MobileNetV2 only); it needs the per-channel table first.

## Run

From the `code/` folder:

    python -m experiments.spread_and_rotation.spread_and_rotation_six_networks
    python -m src.channel_degrees
    python -m experiments.spread_and_rotation.spread_first_networks
    python -m experiments.spread_and_rotation.spread_first_networks --net mobilenetv2

Time: 20 seconds; 10 seconds each for the August scripts.

## Output

Written to `results/spread_and_rotation/`:

- `exp2_spread.csv`, `exp2_rotation.csv`, `exp2_second_network.json`: the six-network run
- `dispersion_*.csv`, `tau_sweep_*.csv`, `conjunct3_mobilenetv2.csv`: the August run

## Where it is used

Sections 6.5 and 6.6 (the October run is called exp2).

## Checked?

Rerun 16 Nov 2026: rotation file identical; spread file equal to 7 digits (cluster against laptop); August files identical. Careful: the weights-only spread for the CIFAR MobileNetV2 reads 75,033. That number is meaningless, see `dead_channels`.

Old name: `exp2_second_network.py`, `phase3/gate_b.py`.
