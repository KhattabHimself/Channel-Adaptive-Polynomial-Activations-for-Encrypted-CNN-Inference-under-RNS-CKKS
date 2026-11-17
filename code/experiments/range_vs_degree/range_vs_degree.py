"""
Range vs degree, work in progress.

Swap every ReLU for a chebyshev polynomial, then try different ways of picking
each channel's range (minmax, mean +- k std, percentiles) against different
degrees, on resnet20 / resnet56 on cifar10 and resnet20 on cifar100.
Should also give the 6 sigma and degree 13 rows that aren't actually in
accuracy_sweep.csv.

run:  python exp3_range_selection.py --targets cifar10_resnet20 --images 1000 --tag smoke
      python exp3_range_selection.py

TODO
- smoke run first
- cifar100 normalisation numbers need checking (baseline check should catch it)
- probably calibrate on train not test (--calib-split train)
- the minmax row isn't the same setup as the old one, don't compare directly
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.network_tools import install_cifar_resnet, load_model  # noqa: E402
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_result, write_table  # noqa: E402

RULES = ["minmax", "sigma2", "sigma3", "sigma4", "sigma6", "pct1", "pct0.1"]
DEGREES = [7, 15, 31, 63]

TARGETS = {
    "cifar10_resnet20": ("cifar10", 92.60),
    "cifar10_resnet56": ("cifar10", 94.37),
    "cifar100_resnet20": ("cifar100", 68.83),
}

# normalisation used by chenyaofo/pytorch-cifar-models
# cifar10 values are from the old run, cifar100 ones not checked yet
# (the baseline check should fail if they're wrong)
NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

# old values, not the same setup, just to eyeball
ARCHIVED_CIFAR10_RESNET20 = {"minmax": -2.17, "sigma3": -82.60}


def loader(dataset: str, split: str, n: int | None, batch: int, root: Path):
    import torchvision
    import torchvision.transforms as T
    mean, std = NORM[dataset]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = {"cifar10": torchvision.datasets.CIFAR10,
           "cifar100": torchvision.datasets.CIFAR100}[dataset]
    ds = cls(root=root, train=(split == "train"), download=True, transform=tf)
    if n:
        ds = torch.utils.data.Subset(ds, range(n))
    return torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=2)


@torch.no_grad()
def evaluate(model, dl, device) -> tuple[float, float]:
    """(top-1 %, fraction of images with any non-finite logit)."""
    ok = tot = bad = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        bad += (~torch.isfinite(out)).any(1).sum().item()
        ok += (out.argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * ok / tot, bad / tot


def interval(rule: str, act) -> tuple[torch.Tensor, torch.Tensor]:
    """(center, half-width) per channel for one rule."""
    mu, std, lo, hi = act.stats()
    if rule == "minmax":
        pad = 0.05 * (hi - lo)
        return (hi + lo) / 2, (hi - lo) / 2 + pad
    if rule.startswith("sigma"):
        return mu, float(rule[len("sigma"):]) * std
    if rule.startswith("pct"):
        q = float(rule[len("pct"):]) / 100.0
        qlo, qhi = act.quantiles(q)
        return (qhi + qlo) / 2, (qhi - qlo) / 2
    raise ValueError(f"unknown rule {rule!r}")


def run_target(name: str, args, device) -> tuple[list[dict], dict]:
    dataset, published = TARGETS[name]
    model = load_model(name).to(device)
    acts = install_cifar_resnet(model)
    root = Path(args.data_root)

    dl = loader(dataset, "test", args.images, args.batch, root)
    base, _ = evaluate(model, dl, device)
    print(f"  baseline top-1 {base:.2f}%  (published {published:.2f}%)",
          flush=True)
    info = {"dataset": dataset, "baseline_top1": base,
            "published_top1": published}
    if args.images is None and abs(base - published) > args.baseline_tol:
        info["error"] = (f"BASELINE MISMATCH: {base:.2f} vs published "
                         f"{published:.2f}; preprocessing or checkpoint wrong")
        print(f"  {info['error']} - grid NOT run for {name}", flush=True)
        return [], info

    for a in acts.values():
        a.reset_calibration()
        a.mode = "calibrate"
    evaluate(model, loader(dataset, args.calib_split, args.calib_images,
                           args.batch, root), device)
    for a in acts.values():
        a.mode = "relu"

    rows = []
    for rule in RULES:
        for d in DEGREES:
            t = time.time()
            for a in acts.values():
                center, half = interval(rule, a)
                a.configure(center, half, [d] * len(center))
            acc, nonfinite = evaluate(model, dl, device)
            for a in acts.values():
                a.mode = "relu"
            rows.append({
                "target": name, "dataset": dataset, "rule": rule,
                "degree": d, "top1": acc, "baseline_top1": base,
                "delta_points": acc - base,
                "nonfinite_logit_frac": nonfinite,
                "calib_split": args.calib_split,
                "n_images": args.images or len(dl.dataset),
            })
            print(f"  {rule:<7} d={d:<3} top-1 {acc:6.2f}%  "
                  f"delta {acc - base:+7.2f}  nonfinite {nonfinite:.3f}  "
                  f"({time.time() - t:.0f}s)", flush=True)

    g = pd.DataFrame(rows)
    at31 = g[g.degree == 31].delta_points
    under_mm = g[g.rule == "minmax"].delta_points
    info["range_effect_d31"] = float(at31.max() - at31.min())
    info["degree_effect_minmax"] = float(under_mm.max() - under_mm.min())
    info["range_dominates"] = bool(info["range_effect_d31"]
                                   > info["degree_effect_minmax"])
    if name == "cifar10_resnet20":
        info["archived_for_orientation"] = ARCHIVED_CIFAR10_RESNET20
    return rows, info


def main() -> None:
    _stamp.use_results("range_vs_degree")
    ap = argparse.ArgumentParser(description="EXP 3: range vs degree grid")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS),
                    choices=list(TARGETS))
    ap.add_argument("--images", type=int, default=None,
                    help="evaluate on the first N test images (smoke runs); "
                         "the baseline guard is skipped when set")
    ap.add_argument("--calib-images", type=int, default=2000)
    ap.add_argument("--calib-split", choices=["test", "train"], default="test")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--baseline-tol", type=float, default=1.5)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner(["EXP 3 - APPROXIMATION RANGE vs DEGREE, one grid",
            f"rules {RULES}", f"degrees {DEGREES}",
            f"device {device}"
            + (f" ({torch.cuda.get_device_name(0)})"
               if device.type == "cuda" else " - expect this to be slow"),
            "range dominates on a target if effect(rules @ d31) > "
            "effect(degrees @ minmax)"])

    all_rows, per_target = [], {}
    for name in args.targets:
        print(f"\n--- {name} ---", flush=True)
        rows, info = run_target(name, args, device)
        all_rows.extend(rows)
        per_target[name] = info

    mismatched = [n for n, i in per_target.items() if "error" in i]
    scored = {n: i for n, i in per_target.items() if "range_dominates" in i}
    new = [n for n in scored if n != "cifar10_resnet20"]
    dom_new = [n for n in new if scored[n]["range_dominates"]]

    parts = [f"{n}: range {i['range_effect_d31']:.1f} vs degree "
             f"{i['degree_effect_minmax']:.1f} pts"
             for n, i in scored.items()]
    if args.images is not None:
        verdict = "SMOKE RUN ONLY (subset of images) - not a result. " \
                  + "; ".join(parts)
    elif mismatched:
        verdict = (f"BASELINE MISMATCH on {', '.join(mismatched)} - fix "
                   f"preprocessing before reading anything. "
                   + "; ".join(parts))
    else:
        verdict = (f"Range dominates degree on {len(dom_new)}/{len(new)} new "
                   f"targets. " + "; ".join(parts))

    tag = f"_{args.tag}" if args.tag else ""
    payload = {
        "experiment": "EXP 3 approximation range vs degree",
        "rules": RULES, "degrees": DEGREES,
        "calib_split": args.calib_split,
        "calib_images": args.calib_images,
        "device": str(device),
        "per_target": per_target,
    }
    if all_rows:
        write_table(f"exp3_range_grid{tag}", pd.DataFrame(all_rows),
                    {"experiment": payload["experiment"],
                     "calib_split": args.calib_split})
    out = write_result(f"exp3_range_selection{tag}", payload, verdict)
    banner([f"VERDICT: {verdict}"])
    print(f"wrote {out}", flush=True)
    raise SystemExit(1 if mismatched and args.images is None else 0)


if __name__ == "__main__":
    main()
