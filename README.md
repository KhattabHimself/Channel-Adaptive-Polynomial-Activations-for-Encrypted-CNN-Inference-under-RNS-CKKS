# Channel-Adaptive Polynomial Activations for Encrypted CNN Inference under RNS-CKKS

MSc thesis, Jodrey School of Computer Science, Acadia University
CILS Lab

**Author:** Mohamed Khattab
**Supervisor:** Dr. Lydia Bouzar-Benlabiod
**Status:** in progress, defense planned for April 2027

> This repository holds the experiment code and the result files.
> The thesis text is not in this repository. A few files are held back and
> will be added later.

## What is in this repository

    code/
      src/           shared code
      experiments/   one folder per experiment, each with a README
      scripts/       figures, timing summaries, statistics, cost model, OpenFHE container
      tests/         automated tests
      reproduce.py   one command per result, and a check of the saved results
      environment.yml
    results/         one folder per experiment, same names as code/experiments/

From the `code/` folder:

    python reproduce.py list      what can be reproduced, and how long it takes
    python reproduce.py check     compare the saved result files with the expected numbers
    python reproduce.py run <name>

`check` reports one missing file, for the entry `window`: its code and result
file are among those held back. Every other entry passes.

Python 3.12 with PyTorch 2.4.1 on CPU (`environment.yml`). The encrypted
experiments use OpenFHE 1.6.1 in a Linux container
(`code/scripts/openfhe_docker`). Pretrained models and datasets download
themselves on first use. In the result files, local paths and the machine
name have been replaced by placeholders such as `<project>`.

## What this is about

Running a neural network on encrypted data (homomorphic encryption) lets a
server classify an image without ever seeing it. The catch is cost. Under the
CKKS scheme every multiplication uses up a "level", and when the levels run
out the ciphertext has to be refreshed with bootstrapping, which is by far the
most expensive step.

ReLU cannot be computed directly under encryption, so it is replaced with a
polynomial. Higher degree means better accuracy but more levels. Most work
uses one polynomial for the whole network.

This thesis asks a simple question: **can we save levels by giving each
channel its own polynomial degree**, so cheap channels stop paying for
expensive ones?

## What the thesis looks at

- **Level accounting:** how many levels a CNN block really uses under
  RNS-CKKS, and where they go.
- **Channel grouping:** whether grouping channels by the degree they need can
  lower the level cost, and what in the packing, the weights and the next
  layer decides that.
- **Range vs degree:** for the polynomial that replaces ReLU, how much the
  choice of input range matters compared to the choice of degree.

The failure conditions were written down before the experiments were run, so
a negative answer is reported as a result, not hidden.

## Setup used

- Encryption: RNS-CKKS with [OpenFHE](https://github.com/openfheorg/openfhe-development)
- Networks: pretrained ResNet-20, ResNet-56, MobileNetV2 and MNASNet
- Data: CIFAR-10, CIFAR-100, ImageNet weights
- Compute: Digital Research Alliance of Canada (Nibi cluster, H100 GPUs)

## Related work by the author

- A systematic literature review of acceleration techniques for
  homomorphic-encryption CNN inference (in preparation).

## Contact

Mohamed Khattab, Acadia University

## Acknowledgements

This research was enabled in part by support provided by the Digital Research
Alliance of Canada (alliancecan.ca).
