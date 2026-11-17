"""
End-to-end encrypted inference on MNIST, step 1 of 2: train a small CNN and
write the plan for the encrypted runs (extension).

The thesis has no network run completely under encryption. This is the
smallest honest one that fits on a laptop: no bootstrapping, a network small
enough for one modulus chain, the whole MNIST test set, decrypted accuracy.

Network (trained here, in plain PyTorch, with ReLU):
    28x28 image -> 2x2 average pool (done by the client before encryption) -> 14x14
    conv 3x3, stride 2, 4 channels  -> batch norm -> ReLU        144 values
    fully connected 144 -> 32       -> batch norm -> ReLU         32 values
    fully connected 32 -> 10                                       logits

Splits, fixed here:
    training      MNIST training images 0 to 49,999
    calibration   training images 0 to 1,999 (intervals of the activations)
    validation    training images 50,000 to 59,999 (never trained on; used to
                  choose the margin, the uniform degree and the degree rule)
    test          the 10,000 test images (reported once, plain and encrypted)

After training, the ReLUs are replaced by Chebyshev interpolants without any
retraining, exactly as in the thesis: one interval per channel of the
convolution and one per neuron of the hidden layer.

Packing for the encrypted run: one ciphertext per value of the network
(per pixel, per neuron), with the images of the batch in the slots. The
linear layers are then sums of ciphertexts times plaintext numbers and need
no rotation. Every channel has its own ciphertexts, so every channel can have
its own degree without any grouping: this is the most favourable packing for
per-channel degrees (the "one ciphertext per channel" end of research
question 3).

What is chosen on validation, by rules fixed here before anything was run:
    margin          the margin in {0.05, 0.1, 0.25, 0.5} with the best
                    validation accuracy at a uniform degree of 15
    uniform degree  the smallest of 3, 7, 15, 31 whose validation accuracy is
                    within 0.5 points of the ReLU network
    per-channel     the site-normalised rule (see degree_rule_fix): among the
                    settings within 0.5 points on validation, the one with
                    the least activation time by the per-degree timing table

Writes results/end_to_end_mnist/plan.npz and plan.json for encrypted_run.py,
and plaintext_accuracy.csv.

run (from code/):
    python -m experiments.end_to_end_mnist.train_and_plan --data-root data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # before numpy: on this Windows install the other order loads two OpenMP runtimes
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.run_stamp as _stamp  # noqa: E402
from src.network_tools import cheb_coeffs, required_degree  # noqa: E402
from src.run_stamp import banner, write_table  # noqa: E402

BAR = 0.5
MARGINS = (0.05, 0.1, 0.25, 0.5)
UNIFORM = (3, 7, 15, 31)
TIMING = _stamp.PROJECT_RESULTS / "grouping_time" / "repeat_runs" / "summary_per_degree.csv"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(144, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x)))
        x = F.relu(self.bn2(self.fc1(x.flatten(1))))
        return self.fc2(x)


def load_mnist(root: Path):
    import torchvision
    tr = torchvision.datasets.MNIST(root=root, train=True, download=True)
    te = torchvision.datasets.MNIST(root=root, train=False, download=True)

    def prep(d):
        x = d.data.float().div(255).unsqueeze(1)
        x = F.avg_pool2d(x, 2)
        return (x - 0.1307) / 0.3081, d.targets.clone()
    return prep(tr), prep(te)


def train(xtr, ytr, epochs: int = 8) -> Net:
    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
    net.eval()
    return net


def as_matrices(net: Net):
    """the network as three affine maps on flat vectors, batch norm folded in"""
    with torch.no_grad():
        eye = torch.eye(196).view(196, 1, 14, 14)
        zero = torch.zeros(1, 1, 14, 14)
        f1 = lambda x: net.bn1(net.conv(x)).flatten(1)  # noqa: E731
        c1 = f1(zero)[0]
        A1 = (f1(eye) - c1).T                       # 144 x 196
        f2 = lambda h: net.bn2(net.fc1(h))          # noqa: E731
        c2 = f2(torch.zeros(1, 144))[0]
        A2 = (f2(torch.eye(144)) - c2).T            # 32 x 144
        A3, c3 = net.fc2.weight.clone(), net.fc2.bias.clone()
    return [m.double().numpy() for m in (A1, c1, A2, c2, A3, c3)]


def mm(a, b):
    """matrix product through torch: numpy's own product loads a second
    OpenMP runtime on this Windows install and aborts"""
    return (torch.from_numpy(np.ascontiguousarray(a)) @ torch.from_numpy(np.ascontiguousarray(b))).numpy()


def norm(v, axis=None):
    return np.sqrt((np.asarray(v) ** 2).sum(axis=axis))


def relu_forward(M, X):
    A1, c1, A2, c2, A3, c3 = M
    z1 = mm(X, A1.T) + c1
    z2 = mm(np.maximum(z1, 0), A2.T) + c2
    return z1, z2, mm(np.maximum(z2, 0), A3.T) + c3


def intervals(z1, z2, rule: str, margin: float):
    """(center, half) per value of each layer. The 144 values of the conv
    layer share one interval per channel (36 values each)."""
    out = []
    for z, groups in ((z1, np.repeat(np.arange(4), 36)), (z2, np.arange(32))):
        m, h = np.zeros(z.shape[1]), np.zeros(z.shape[1])
        for g in np.unique(groups):
            v = z[:, groups == g]
            if rule == "observed":
                lo, hi = v.min(), v.max()
                pad = margin * (hi - lo)
                lo, hi = lo - pad, hi + pad
            else:                                   # mean +/- 3 standard deviations
                lo, hi = v.mean() - 3 * v.std(), v.mean() + 3 * v.std()
            m[groups == g], h[groups == g] = (hi + lo) / 2, max((hi - lo) / 2, 1e-6)
        out.append((m, h, groups))
    return out


def layer_spec(m, h, degrees):
    """per value: kind (0 polynomial, 1 identity, -1 zero), degree, Chebyshev
    coefficients of max(0, t - t0) on [-1, 1]"""
    spec = []
    for mi, hi, d in zip(m, h, degrees):
        t0 = -mi / hi
        if t0 <= -1:
            spec.append({"kind": 1, "degree": 1, "t0": float(t0), "coef": [float(-t0), 1.0]})
        elif t0 >= 1:
            spec.append({"kind": -1, "degree": 0, "t0": float(t0), "coef": [0.0]})
        else:
            cc = cheb_coeffs(int(d), int(round(t0 * 1000)))
            spec.append({"kind": 0, "degree": int(d), "t0": float(t0), "coef": [float(c) for c in cc]})
    return spec


def poly_forward(M, iv, specs, X):
    """the polynomial network in ordinary arithmetic, written the way the
    encrypted run computes it: u = (z - m)/h, g(u), then h folded forward"""
    A1, c1, A2, c2, A3, c3 = M
    (m1, h1, _), (m2, h2, _) = iv
    cheb = np.polynomial.chebyshev.chebval
    u1 = mm(X, (A1 / h1[:, None]).T) + (c1 - m1) / h1
    g1 = np.stack([cheb(u1[:, j], s["coef"]) for j, s in enumerate(specs[0])], 1)
    u2 = mm(g1, (A2 * h1[None, :] / h2[:, None]).T) + (c2 - m2) / h2
    g2 = np.stack([cheb(u2[:, j], s["coef"]) for j, s in enumerate(specs[1])], 1)
    return mm(g2, (A3 * h2[None, :]).T) + c3


def acc(logits, y):
    ok = np.isfinite(logits).all(1) & (logits.argmax(1) == y)
    return 100.0 * ok.mean()


def rule_degrees(M, iv, B, floor, cap):
    """site-normalised degree rule, per channel (conv) and per neuron (fc)"""
    A2, A3 = M[2], M[4]
    out = []
    for (m, h, groups), W in zip(iv, (A2, A3)):
        col = norm(W, axis=0)
        gs = np.unique(groups)
        vn = np.array([norm(col[groups == g]) for g in gs])
        hh = np.array([h[groups == g][0] for g in gs])
        t0 = np.array([-m[groups == g][0] / h[groups == g][0] for g in gs])
        cross = (t0 > -1) & (t0 < 1)
        sens = hh * vn
        med = np.median(sens[cross]) if cross.any() else 1.0
        d = np.ones(len(gs), dtype=int)
        for i in np.flatnonzero(cross):
            d[i] = min(max(required_degree(sens[i] / med, t0[i], B, max_degree=cap), floor), cap)
        out.append(d[np.searchsorted(gs, groups)])
    return out


def main() -> None:
    _stamp.use_results("end_to_end_mnist")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    banner(["END TO END ON MNIST, step 1: train and plan"])
    (xtr, ytr), (xte, yte) = load_mnist(Path(args.data_root))
    net = train(xtr[:50000], ytr[:50000])
    M = as_matrices(net)
    flat = lambda x: x.flatten(1).double().numpy()  # noqa: E731
    Xc, Xv, Xt = flat(xtr[:2000]), flat(xtr[50000:]), flat(xte)
    yv, yt = ytr[50000:].numpy(), yte.numpy()
    with torch.no_grad():
        ref = net(xte).double().numpy()
    assert np.abs(relu_forward(M, Xt)[2] - ref).max() < 1e-4, "matrix form differs from the network"
    base_v, base_t = acc(relu_forward(M, Xv)[2], yv), acc(relu_forward(M, Xt)[2], yt)
    print(f"ReLU network: validation {base_v:.2f}%, test {base_t:.2f}%", flush=True)
    z1c, z2c, _ = relu_forward(M, Xc)
    tdeg = pd.read_csv(TIMING, comment="#").set_index("degree").seconds_mean
    cost = lambda degs: float(sum(tdeg.get(int(d), 0.0) for d in degs if d >= 1))  # noqa: E731

    rows, plans = [], {}

    def consider(name, rule, margin, degs, keep=False, note=""):
        iv = intervals(z1c, z2c, rule, margin)
        specs = [layer_spec(m, h, d) for (m, h, _), d in zip(iv, degs)]
        eff = [[s["degree"] for s in sp] for sp in specs]
        r = {"config": name, "interval": rule, "margin": margin,
             "validation_top1": acc(poly_forward(M, iv, specs, Xv), yv),
             "test_top1": acc(poly_forward(M, iv, specs, Xt), yt),
             "relu_validation_top1": base_v, "relu_test_top1": base_t,
             "max_degree_layer1": max(eff[0]), "max_degree_layer2": max(eff[1]),
             "activation_time_weight_s": cost(eff[0]) + cost(eff[1]),
             "polynomial_values": int(sum(s["kind"] == 0 for sp in specs for s in sp)),
             "identity_values": int(sum(s["kind"] == 1 for sp in specs for s in sp)),
             "zero_values": int(sum(s["kind"] == -1 for sp in specs for s in sp)),
             "note": note}
        r["validation_points_lost"] = base_v - r["validation_top1"]
        r["test_points_lost"] = base_t - r["test_top1"]
        rows.append(r)
        if keep:
            plans[name] = (iv, specs)
        return r

    best_margin = max(MARGINS, key=lambda m: consider(
        f"uniform 15, margin {m}", "observed", m, [[15] * 144, [15] * 32])["validation_top1"])
    print(f"margin chosen on validation: {best_margin}", flush=True)
    uni = {d: consider(f"uniform {d}", "observed", best_margin, [[d] * 144, [d] * 32])
           for d in UNIFORM}
    ok = [d for d in UNIFORM if uni[d]["validation_points_lost"] <= BAR]
    d_star = min(ok) if ok else max(UNIFORM)
    print(f"uniform degree chosen on validation: {d_star}", flush=True)
    cands = []
    iv = intervals(z1c, z2c, "observed", best_margin)
    for B in (0.3, 0.15, 0.08, 0.04, 0.02, 0.01):
        for floor in (1, 3, 7):
            degs = rule_degrees(M, iv, B, floor, 31)
            r = consider(f"site_norm B={B} floor={floor}", "observed", best_margin, degs)
            cands.append((r, degs))
    okc = [(r, d) for r, d in cands if r["validation_points_lost"] <= BAR]
    chosen = min(okc, key=lambda rd: rd[0]["activation_time_weight_s"]) if okc else None

    # the configurations that are run under encryption
    consider("A uniform", "observed", best_margin, [[d_star] * 144, [d_star] * 32], keep=True,
             note=f"uniform degree {d_star}, chosen on validation")
    if chosen:
        r = consider("B per-channel", "observed", best_margin, chosen[1], keep=True,
                     note="per-channel degrees: " + chosen[0]["config"])
        # the same channels, every one raised to the largest degree of its layer
        top = [[max(d)] * len(d) for d in chosen[1]]
        consider("C per-channel raised to layer maximum", "observed", best_margin, top, keep=True,
                 note="layerwise version of B")
    consider("D uniform, three standard deviations", "three_std", 0.0,
             [[d_star] * 144, [d_star] * 32], keep=True,
             note=f"uniform degree {d_star} on the interval rule that fails in the thesis")
    out = pd.DataFrame(rows)
    A1, c1, A2, c2, A3, c3 = M
    _stamp.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_stamp.RESULTS_DIR / "plan.npz", X=Xt, y=yt, A1=A1, c1=c1, A2=A2, c2=c2,
                        A3=A3, c3=c3,
                        **{f"logits::{k}": poly_forward(M, iv_, sp, Xt) for k, (iv_, sp) in plans.items()})
    plan = {k: {"m1": iv_[0][0].tolist(), "h1": iv_[0][1].tolist(),
                "m2": iv_[1][0].tolist(), "h2": iv_[1][1].tolist(),
                "layer1": sp[0], "layer2": sp[1]} for k, (iv_, sp) in plans.items()}
    (_stamp.RESULTS_DIR / "plan.json").write_text(json.dumps(plan), encoding="utf8")
    torch.save(net.state_dict(), _stamp.RESULTS_DIR / "mnist_small_cnn.pt")
    kept = out[out.config.str.match(r"^[A-D] ")]
    verdict = (f"ReLU network {base_t:.2f}% on the test set. Margin {best_margin}, uniform degree "
               f"{d_star}. In ordinary arithmetic: " + "; ".join(
                   f"{r.config}: {r.test_top1:.2f}%" for r in kept.itertuples()))
    write_table("plaintext_accuracy", out, {
        "experiment": "end to end on MNIST, plaintext planning",
        "bar_points": BAR, "margin": best_margin, "uniform_degree": d_star}, verdict)
    banner([verdict])


if __name__ == "__main__":
    main()
