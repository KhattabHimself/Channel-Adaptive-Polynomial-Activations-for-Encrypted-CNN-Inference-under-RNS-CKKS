"""
Check of one implementation detail of the range experiment (added 17 Nov 2026).

PolyAct fits the polynomial to max(0, t - t0) on [-1, 1], where t0 is the
position of the ReLU kink in the channel's interval, and it clamps t0 to
[-1.5, 1.5]. If the true t0 is below -1.5, every input of the channel is
positive, ReLU is the identity there, and the clamped fit is the wrong
straight line: it returns the input shifted by a constant. (Above +1.5 the
clamp is harmless: the function is zero either way.)

This counts how many channels are affected under the observed-range interval
and measures what the clamp costs in accuracy, by scoring the same setting
twice: as the range experiment does it, and with those channels made exact.

run (from code/):
    python -m experiments.range_vs_degree.kink_clamp_check --data-root data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.run_stamp as _stamp  # noqa: E402
from src.network_tools import install_cifar_resnet, load_model  # noqa: E402
from src.run_stamp import banner, write_table  # noqa: E402
from experiments.range_vs_degree.range_vs_degree import (  # noqa: E402
    TARGETS, evaluate, interval, loader)


def make_exact(act, center, half) -> None:
    """overwrite the coefficients of channels whose kink is outside the
    interval with the exact straight line (identity, or zero)"""
    t0 = -center / half.clamp_min(1e-6)
    coef = act.coef
    pos, neg = t0 <= -1.0, t0 >= 1.0
    coef[pos | neg] = 0.0
    coef[pos, 1] = 1.0
    coef[pos, 0] = -t0[pos]
    act.coef = coef


def main() -> None:
    _stamp.use_results("range_vs_degree")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--targets", nargs="+", default=["cifar10_resnet20", "cifar10_resnet56"],
                    choices=list(TARGETS))
    ap.add_argument("--degrees", type=int, nargs="+", default=[15, 31])
    ap.add_argument("--images", type=int, default=None)
    args = ap.parse_args()
    banner(["KINK CLAMP CHECK for the range experiment"])
    root = Path(args.data_root)
    rows = []
    for name in args.targets:
        dataset, _ = TARGETS[name]
        model = load_model(name)
        acts = install_cifar_resnet(model)
        dl = loader(dataset, "test", args.images, 500, root)
        base, _ = evaluate(model, dl, "cpu")
        for a in acts.values():
            a.reset_calibration()
            a.mode = "calibrate"
        evaluate(model, loader(dataset, "train", 2000, 500, root), "cpu")
        for a in acts.values():
            a.mode = "relu"
        n = wrong = linear_pos = linear_neg = 0
        for a in acts.values():
            center, half = interval("minmax", a)
            t0 = -center / half.clamp_min(1e-6)
            n += len(t0)
            wrong += int((t0 < -1.5).sum())
            linear_pos += int((t0 <= -1.0).sum())
            linear_neg += int((t0 >= 1.0).sum())
        print(f"{name}: {n} channels; kink below -1.5 (clamp gives the wrong line): {wrong}; "
              f"always positive: {linear_pos}; always negative: {linear_neg}", flush=True)
        for d in args.degrees:
            for exact in (False, True):
                for a in acts.values():
                    center, half = interval("minmax", a)
                    a.configure(center, half, [d] * len(center))
                    if exact:
                        make_exact(a, center, half)
                acc, nonfinite = evaluate(model, dl, "cpu")
                for a in acts.values():
                    a.mode = "relu"
                rows.append({"target": name, "degree": d,
                             "handling": "exact" if exact else "as in the range experiment",
                             "top1": acc, "baseline_top1": base,
                             "delta_points": acc - base,
                             "nonfinite_logit_frac": nonfinite, "channels": n,
                             "channels_kink_below_-1.5": wrong,
                             "channels_always_positive": linear_pos,
                             "channels_always_negative": linear_neg})
                print(f"  degree {d:<2} {rows[-1]['handling']:<27} top-1 {acc:6.2f}%  "
                      f"nonfinite {nonfinite:.4f}", flush=True)
    out = pd.DataFrame(rows)
    p = out.pivot_table(index=["target", "degree"], columns="handling", values="top1")
    diff = (p["exact"] - p["as in the range experiment"])
    verdict = ("Kink clamp: " + "; ".join(
        f"{t} d={d}: {v:+.2f} points when made exact" for (t, d), v in diff.items())
        + ". Affected channels: " + ", ".join(
            f"{t} {int(g['channels_kink_below_-1.5'].iloc[0])} of {int(g.channels.iloc[0])}"
            for t, g in out.groupby("target")))
    write_table("kink_clamp_check", out, {
        "experiment": "kink clamp check, observed-range interval, train calibration"}, verdict)
    banner([verdict])


if __name__ == "__main__":
    main()
