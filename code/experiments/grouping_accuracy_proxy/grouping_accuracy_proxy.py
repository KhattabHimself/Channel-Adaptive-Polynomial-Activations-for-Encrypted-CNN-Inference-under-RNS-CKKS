"""
Accuracy proxy for the grouping time result (thesis section 6.14).

The time saving is measured on the ImageNet MobileNetV2, and no ImageNet
images are on this machine, so its accuracy with polynomial activations
cannot be measured here. This runs the closest thing that can: the CIFAR-10
MobileNetV2 checkpoint, a different network of the same architecture family,
with the SAME degree rule, the SAME seven budgets and the SAME grouping
procedure, and scores it on the CIFAR-10 test set.

For every budget B it replaces every activation by a per-channel Chebyshev
polynomial and scores three ways of setting the degrees:

  per_channel   every channel gets the degree the rule gives it (what the
                rule asks for; not realisable when channels share a ciphertext)
  grouped       channels are sorted and grouped as in the time experiment, and
                every channel runs at the largest degree in its group
  layerwise     every channel runs at the largest degree of its layer (the
                baseline the time saving is measured against)

Each is scored with the interval taken from the observed range on 2,000
training images (plus 5%), which is the only interval rule that works
(section 6.11). Two extra rows score the layerwise setting with the
weights-only three-standard-deviation interval the degrees were computed for.

Plain PyTorch, no encryption. ReLU6 is treated as ReLU; on this network no
activation input exceeds 6 on any test image (dead_channels experiment), and
the script checks that the unmodified accuracy still comes out at 94.05%.

What this is NOT: it is not the ImageNet MobileNetV2, the number of
ciphertexts per layer is different (CIFAR images are small, so most layers
fit in one or two ciphertexts), and 46% of this network's channels are dead.
It says whether the degree rule gives a usable MobileNetV2-style network at
each budget on CIFAR-10. It does not say what the ImageNet network would do.

run (from code/):
    python -m experiments.grouping_accuracy_proxy.grouping_accuracy_proxy --data-root data
    add --images 2000 to score on the first 2,000 test images only
    add --cap 31 or --margin 0.25 for the what-if runs (see README.md here)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.run_stamp as _stamp  # noqa: E402
from src.network_tools import (PolyAct, channel_table, degrees_at,  # noqa: E402
                               depthwise_regions, load_model)
from src.run_stamp import banner, write_table  # noqa: E402
from experiments.grouping_time.grouping_time import (  # noqa: E402
    BUDGETS, SLOTS, dp_place_shared, kappa_sqrt2d, shared_order)
from experiments.range_vs_degree.range_vs_degree import evaluate, loader  # noqa: E402

NET = "cifar10_mobilenetv2_x1_0"
PUBLISHED = 93.79
BASE_GRID = 32 * 32            # CIFAR input grid, as for the CIFAR ResNets


def install(model) -> dict[str, PolyAct]:
    """replace every ReLU6 that follows a BatchNorm by a PolyAct, keyed by
    the name of that BatchNorm"""
    leaves = [(n, m) for n, m in model.named_modules() if not list(m.children())]
    acts = {}
    for (bn_name, bn), (act_name, act) in zip(leaves, leaves[1:]):
        if isinstance(bn, nn.BatchNorm2d) and isinstance(act, nn.ReLU6):
            parent = model
            *path, last = act_name.split(".")
            for part in path:
                parent = getattr(parent, part)
            pa = PolyAct(bn_name)
            setattr(parent, last, pa)
            acts[bn_name] = pa
    return acts


def configure_exact(act: PolyAct, center, half, degrees) -> int:
    """PolyAct.configure, then make the channels whose kink lies outside the
    interval exact.

    PolyAct clamps the kink position t0 = -center/half to [-1.5, 1.5]. When
    the true kink is further out, the channel's inputs never cross zero on the
    interval and ReLU is exactly linear there: the identity if every input is
    positive (t0 <= -1), zero if every input is negative (t0 >= 1). The clamp
    then gives the wrong straight line for t0 < -1.5 (it returns x shifted by
    a constant). In the CIFAR ResNets almost no channel is like that; in this
    MobileNetV2 about a tenth are. Here those channels get the exact line:
    coefficients (-t0, 1) for the identity, all zero for the dead side.
    Returns the number of channels that the clamp would have got wrong.
    """
    act.configure(center, half, degrees)
    h = half.clamp_min(1e-6)
    t0 = (-center / h)
    coef = act.coef
    pos = t0 <= -1.0                     # always on: ReLU(x) = x = h * (t - t0)
    neg = t0 >= 1.0                      # always off: ReLU(x) = 0
    coef[pos | neg] = 0.0
    if coef.shape[1] > 1:
        coef[pos, 1] = 1.0
    coef[pos, 0] = -t0[pos]
    act.coef = coef
    return int((t0 < -1.5).sum())


def group_assignment(sites: list[list[int]], n_groups: int, cap: int):
    """The grouping of dp_place_shared, returned per channel.

    Same order and same cut rule as experiments.grouping_time; the cost is
    checked against that function so the two cannot drift apart. Returns, for
    each site, the degree every channel runs at (the largest in its group).
    """
    n = len(sites[0])
    order = shared_order(sites)
    seg = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        running = [0] * len(sites)
        for k in range(i + 1, n + 1):
            c = order[k - 1]
            for si, s in enumerate(sites):
                running[si] = max(running[si], s[c])
            seg[i][k] = sum(kappa_sqrt2d(m) for m in running)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(n_groups + 1)]
    back = [[-1] * (n + 1) for _ in range(n_groups + 1)]
    dp[0][0] = 0.0
    for j in range(1, n_groups + 1):
        for i in range(j, n - (n_groups - j) + 1):
            for ip in range(max(j - 1, i - cap), i):
                if dp[j - 1][ip] == INF:
                    continue
                cand = dp[j - 1][ip] + seg[ip][i]
                if cand < dp[j][i]:
                    dp[j][i], back[j][i] = cand, ip
    ref_cost, _ = dp_place_shared(sites, n_groups, cap, kappa_sqrt2d)
    assert abs(dp[n_groups][n] - ref_cost) < 1e-9, "grouping differs from grouping_time"
    sizes, i = [], n
    for j in range(n_groups, 0, -1):
        sizes.append(i - back[j][i])
        i = back[j][i]
    sizes = sizes[::-1]
    out = [list(s) for s in sites]
    pos = 0
    for sz in sizes:
        block = order[pos:pos + sz]
        for si, s in enumerate(sites):
            m = max(s[c] for c in block)
            for c in block:
                out[si][c] = m
        pos += sz
    return out


def main() -> None:
    _stamp.use_results("grouping_accuracy_proxy")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--images", type=int, default=None,
                    help="score on the first N test images (quick look only)")
    ap.add_argument("--calib-images", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--cap", type=int, default=63,
                    help="largest degree allowed. 63 is the setting of the time "
                         "experiment. A lower cap is a what-if: every degree "
                         "above it is lowered to it")
    ap.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS),
                    help="budgets to score (default: the seven of the time experiment)")
    ap.add_argument("--floor", type=int, default=1,
                    help="smallest degree allowed. 1 is the setting of the time "
                         "experiment. A higher floor is a what-if: every degree "
                         "below it is raised to it")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="widening of the observed range at each end, as a share "
                         "of the range. 0.05 is the setting of the thesis")
    args = ap.parse_args()
    banner(["ACCURACY PROXY FOR THE GROUPING TIME RESULT",
            f"{NET} on CIFAR-10, degrees from the weights-only rule"])
    root = Path(args.data_root)
    torch.manual_seed(0)

    model = load_model(NET)
    table = channel_table(model, NET)
    regions = depthwise_regions(model)

    # true tensor shape at every BatchNorm, for the number of ciphertexts
    shapes, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            hooks.append(mod.register_forward_hook(
                lambda _m, _i, out, name=name: shapes.__setitem__(name, tuple(out.shape[1:]))))
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for h in hooks:
        h.remove()

    def site_G(name: str) -> int:
        C, H, W = shapes[name]
        return math.ceil(BASE_GRID * math.ceil(C * H * W / BASE_GRID) / SLOTS)

    acts = install(model)
    dl = loader("cifar10", "test", args.images, args.batch, root)
    base, _ = evaluate(model, dl, "cpu")
    print(f"unmodified top-1 {base:.2f}% (published {PUBLISHED:.2f}%), "
          f"{len(acts)} activation sites", flush=True)
    if args.images is None and abs(base - 94.05) > 0.01:
        raise SystemExit("unmodified accuracy is not 94.05%: ReLU6 is not ReLU here, stopping")

    for a in acts.values():
        a.reset_calibration()
        a.mode = "calibrate"
    evaluate(model, loader("cifar10", "train", args.calib_images, args.batch, root), "cpu")
    for a in acts.values():
        a.mode = "relu"
    observed = {}
    for k, a in acts.items():
        _mu, _std, lo, hi = a.stats()
        pad = args.margin * (hi - lo)
        observed[k] = ((hi + lo) / 2, (hi - lo) / 2 + pad)

    # regions: the two activations of a block share one grouping; the rest stand alone
    in_region = {s for pair in regions.values() for s in pair}
    units = [list(pair) for pair in regions.values()]
    units += [[s] for s in acts if s not in in_region]
    units = [[s for s in u if s in acts] for u in units]
    units = [u for u in units if u]

    rows = []
    for d in (3, 7, 15, 31, 63):
        if d > args.cap:
            continue
        t = time.time()
        for s, a in acts.items():
            center, half = observed[s]
            configure_exact(a, center, half, [d] * len(center))
        acc, nonfinite = evaluate(model, dl, "cpu")
        for a in acts.values():
            a.mode = "relu"
        rows.append({"budget_B": float("nan"), "degrees": f"uniform_{d}",
                     "interval": "observed", "degree_cap": args.cap, "degree_floor": args.floor,
                     "margin": args.margin,
                     "top1": acc, "baseline_top1": base, "delta_points": acc - base,
                     "nonfinite_logit_frac": nonfinite, "degree_median": float(d),
                     "degree_mean": float(d), "degree_max": d,
                     "sites_with_2plus_ciphertexts": 0,
                     "channels_always_positive_beyond_clamp": 0,
                     "n_images": args.images or len(dl.dataset)})
        print(f"  uniform degree {d:<2} observed  top-1 {acc:6.2f}%  nonfinite {nonfinite:.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)
    for B in args.budgets:
        df = degrees_at(table, B)
        deg = {s: [max(min(int(d), args.cap), args.floor) for d in g.sort_values("channel_idx").degree]
               for s, g in df.groupby("site") if s in acts}
        ab = {s: g.sort_values("channel_idx")[["a_c", "b_c"]].to_numpy()
              for s, g in df.groupby("site") if s in acts}
        assign = {"per_channel": dict(deg), "layerwise": {}, "grouped": {}}
        n_multi = 0
        for u in units:
            sites = [deg[s] for s in u]
            for s, d in zip(u, sites):
                assign["layerwise"][s] = [max(d)] * len(d)
            G = max(site_G(s) for s in u)
            C = len(sites[0])
            if G < 2 or len({len(x) for x in sites}) != 1:
                for s, d in zip(u, sites):       # one ciphertext: the layer maximum
                    assign["grouped"][s] = [max(d)] * len(d)
                continue
            n_multi += len(u)
            grouped = group_assignment(sites, G, math.ceil(C / G))
            for s, d in zip(u, grouped):
                assign["grouped"][s] = d

        def score(setting: str, interval: str) -> None:
            t = time.time()
            n_fixed = [0]
            for s, a in acts.items():
                if interval == "observed":
                    center, half = observed[s]
                else:                       # the interval the degrees were computed for
                    center = torch.tensor(ab[s][:, 1], dtype=torch.float32)
                    half = 3.0 * torch.tensor(np.abs(ab[s][:, 0]), dtype=torch.float32)
                n_fixed[0] += configure_exact(a, center, half, assign[setting][s])
            acc, nonfinite = evaluate(model, dl, "cpu")
            for a in acts.values():
                a.mode = "relu"
            alld = np.concatenate([np.asarray(assign[setting][s]) for s in acts])
            rows.append({
                "budget_B": B, "degrees": setting, "interval": interval,
                "degree_cap": args.cap, "degree_floor": args.floor,
                "margin": args.margin,
                "top1": acc, "baseline_top1": base, "delta_points": acc - base,
                "nonfinite_logit_frac": nonfinite,
                "degree_median": float(np.median(alld)),
                "degree_mean": float(alld.mean()), "degree_max": int(alld.max()),
                "sites_with_2plus_ciphertexts": n_multi,
                "channels_always_positive_beyond_clamp": n_fixed[0],
                "n_images": args.images or len(dl.dataset)})
            print(f"  B={B:<5} {setting:<11} {interval:<9} top-1 {acc:6.2f}%  "
                  f"delta {acc - base:+7.2f}  nonfinite {nonfinite:.3f}  "
                  f"median degree {rows[-1]['degree_median']:.0f}  ({time.time() - t:.0f}s)",
                  flush=True)

        for setting in ("layerwise", "grouped", "per_channel"):
            score(setting, "observed")
        if B in (0.01, 0.3) and args.floor == 1 and args.cap == 63:
            score("layerwise", "three_std_from_weights")

    out = pd.DataFrame(rows)
    obs = out[(out.interval == "observed") & out.budget_B.notna()]
    best = obs.loc[obs.top1.idxmax()]
    uni = out[out.budget_B.isna()]
    tight = obs[obs.budget_B == min(args.budgets)].set_index("degrees").top1
    verdict = (f"{NET}, CIFAR-10, unmodified {base:.2f}%. Best polynomial setting: "
               f"{best.top1:.2f}% (B={best.budget_B:g}, {best.degrees}). At the tightest "
               f"budget B={min(args.budgets):g}: layerwise {tight['layerwise']:.2f}%, grouped "
               f"{tight['grouped']:.2f}%, per-channel {tight['per_channel']:.2f}%. "
               "Uniform degrees for reference: "
               + ", ".join(f"d={int(r.degree_max)} {r.top1:.2f}%" for r in uni.itertuples()) + ".")
    tag = f"_cap{args.cap}" + ("" if args.floor == 1 else f"_floor{args.floor}") + ("" if args.margin == 0.05 else f"_margin{args.margin:g}") + ("" if args.images is None else f"_first{args.images}")
    write_table("grouping_accuracy_proxy" + tag, out, {
        "experiment": "accuracy proxy for the grouping time result",
        "network": NET, "calib_images": args.calib_images, "calib_split": "train", "margin": args.margin, "degree_floor": args.floor,
        "note": "proxy: CIFAR-10 MobileNetV2, not the ImageNet MobileNetV2 of the time experiment"},
        verdict)
    banner([verdict])


if __name__ == "__main__":
    main()
