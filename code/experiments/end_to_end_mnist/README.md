# end to end on MNIST

**Question:** Does a network run completely under CKKS behave as the thesis says: same accuracy as plain arithmetic, ciphertexts realigned by the next layer, and failure with a statistical interval?

A small CNN (one convolution, one hidden layer) on MNIST, trained here with ReLU, then run under CKKS in OpenFHE with polynomial activations: encrypt, infer, decrypt, compare. 128-bit security, 14 levels, ring dimension 2^15, no bootstrapping. All 10,000 test images in one run.

## Run

From the `code/` folder, then in the container:

    python -m experiments.end_to_end_mnist.train_and_plan --data-root data
    docker run --rm -v "<project folder>:/work" -w /work/code thesis-openfhe python -m experiments.end_to_end_mnist.encrypted_run

Add `--images 64` for a trial, `--tag _repeat2` to keep a repeat run in its own file.

Time: 1 minute, then about 5 minutes for the four configurations (longer if the laptop is busy).

## Output

Written to `results/end_to_end_mnist/`:

- `plaintext_accuracy.csv`: every setting considered on validation, and the four that are run encrypted
- `plan.npz`, `plan.json`, `mnist_small_cnn.pt`: the weights, intervals, degrees and test inputs the encrypted run reads
- `encrypted_run.csv`: **the result.** `encrypted_run_repeat2.csv` and `_repeat3.csv` are timing repeats

## What it found

| | A uniform 15 | B per-channel | C layer maximum of B | D three std |
|---|---|---|---|---|
| decrypted accuracy | 96.87% | 96.94% | 96.93% | decryption refused |
| same network, plain arithmetic | 96.87% | 96.94% | 96.93% | 61.49% |
| levels after first activation | 6 | 5 to 6 | 6 | 6 |
| levels after the next layer | 7 | 7 | 7 | 7 |
| levels at the output | 13 | 13 | 13 | 13 |

The ReLU network scores 97.09%. Per-channel degrees save no level. The linear layer after per-channel activations was 16 to 37% slower in three runs.

## Things to know

- One ciphertext per value of the network, images in the slots. No large network would be packed this way. It is the packing most favourable to per-channel degrees.
- Timings vary by up to 30% between runs on this laptop.
- `train_and_plan.py` does its matrix products through torch because numpy's own product aborts on this Windows install (two OpenMP runtimes).

## Where it is used

Thesis Appendix D, section D.2.
