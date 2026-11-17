"""
Why does mean +- 3 std fail so badly? (Lydia, 2026-10-15: "who used the
statistical standard and lost 83 points")

A Gaussian puts 0.27% of its mass outside mean +- 3 std. If ReLU inputs were
Gaussian, a 3-sigma interval would be safe almost everywhere. This measures
how far from Gaussian they are: per channel, the share of values outside
k std, the largest |x - mu| / std seen, and the kurtosis. Same calibration
as exp3's train run (2,000 train images), then measured on 2,000 test images.

Then the arithmetic: a Chebyshev series fitted on [-1, 1] and evaluated at
|t| > 1 grows like (|t| + sqrt(t^2 - 1))^d. That is the blow-up exp3 logged
as non-finite outputs.

run:  python exp3b_tails.py --data-root <cifar folder>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.network_tools import install_cifar_resnet, load_model  # noqa: E402
from experiments.range_vs_degree.range_vs_degree import TARGETS, evaluate, loader  # noqa: E402
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_table  # noqa: E402

KS = [3, 4, 6]
GAUSS_OUT = {k: math.erfc(k / math.sqrt(2)) for k in KS}


class Tail(torch.nn.Module):
    """records |x - mu| / std per channel against fixed calibration stats"""

    def __init__(self, mu, std):
        super().__init__()
        self.mu, self.std = mu.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
        C = mu.numel()
        self.n = 0
        self.out = {k: torch.zeros(C) for k in KS}
        self.zmax = torch.zeros(C)
        self.m2 = torch.zeros(C)
        self.m4 = torch.zeros(C)

    def forward(self, x):
        z = ((x - self.mu) / self.std).transpose(0, 1).reshape(x.shape[1], -1)
        self.n += z.shape[1]
        for k in KS:
            self.out[k] += (z.abs() > k).sum(1).float()
        self.zmax = torch.maximum(self.zmax, z.abs().max(1).values)
        self.m2 += (z ** 2).sum(1)
        self.m4 += (z ** 4).sum(1)
        return torch.relu(x)


def main() -> None:
    _stamp.use_results("range_vs_degree")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--images", type=int, default=2000)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    args = ap.parse_args()
    root = Path(args.data_root)
    banner(["EXP 3b - how heavy are the tails of ReLU inputs?",
            "Gaussian share outside k std: " + ", ".join(
                f"{k}: {100 * GAUSS_OUT[k]:.3g}%" for k in KS)])

    rows = []
    for name in args.targets:
        dataset, _ = TARGETS[name]
        model = load_model(name)
        acts = install_cifar_resnet(model)
        for a in acts.values():
            a.reset_calibration()
            a.mode = "calibrate"
        evaluate(model, loader(dataset, "train", args.images, 500, root), "cpu")
        tails = {}
        for site, a in acts.items():
            mu, std, _, _ = a.stats()
            tails[site] = Tail(mu, std)
            a.mode = "tail"
            a.forward = tails[site].forward
        evaluate(model, loader(dataset, "test", args.images, 500, root), "cpu")
        for site, t in tails.items():
            for c in range(t.zmax.numel()):
                row = {"net": name, "site": site, "channel": c,
                       "zmax": float(t.zmax[c]),
                       "kurtosis": float(t.m4[c] / t.n) /
                       max(float(t.m2[c] / t.n) ** 2, 1e-12)}
                for k in KS:
                    row[f"out_{k}sd_pct"] = 100 * float(t.out[k][c]) / t.n
                rows.append(row)
        print(f"  {name}: done", flush=True)

    df = pd.DataFrame(rows)
    summ = df.groupby("net").agg(
        channels=("channel", "count"),
        median_out_3sd_pct=("out_3sd_pct", "median"),
        median_out_6sd_pct=("out_6sd_pct", "median"),
        share_channels_beyond_3sd=("zmax", lambda z: float((z > 3).mean())),
        share_channels_beyond_6sd=("zmax", lambda z: float((z > 6).mean())),
        median_zmax=("zmax", "median"),
        median_kurtosis=("kurtosis", "median"))
    print(summ.round(4).to_string())
    r = summ.loc["cifar10_resnet20"]
    verdict = (f"ResNet-20: {100 * r.share_channels_beyond_3sd:.0f}% of channels "
               f"see test values beyond 3 std (median {r.median_zmax:.1f} std, "
               f"kurtosis {r.median_kurtosis:.1f} vs 3 for a Gaussian); median "
               f"share outside 3 std {r.median_out_3sd_pct:.2f}% vs 0.27%")
    write_table("exp3b_tails", df, {"experiment": "EXP 3b tails",
                                    "images": args.images}, verdict)
    write_table("exp3b_tails_summary", summ.reset_index(),
                {"experiment": "EXP 3b summary"})
    banner([verdict])


if __name__ == "__main__":
    main()
