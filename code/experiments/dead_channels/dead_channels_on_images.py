"""
The CIFAR MobileNetV2 spread of 75,033, checked on real images.

exp2b looked at the weights only and found channels whose sensitivity
s_c = |a_c| * ||V[:,c]|| is ~0. Weights alone cannot say whether those
channels are really switched off. This runs CIFAR-10 images through the
network and asks:

1. Is the checkpoint loaded right? Top-1 on all 10,000 test images
   (published 93.79).
2. At every BN -> ReLU6 site, per channel, on the test images: how often the
   input is above 0 (ReLU on), how often above 6 (ReLU6 capped), and how much
   the output varies.
3. Do the weight-dead channels (exp2b's rule) match the channels whose output
   is constant on data?
4. Why are they dead: small gamma, small next-layer weights, or both?
5. Mean ablation: replace a set of channels' outputs by their mean on 2,000
   train images and score all 10,000 test images. If a set is really dead,
   accuracy does not move. Control: the same number of random other channels
   at each site.
6. Rotation (exp2's rule) with the dead channels taken out.

run:  python exp2c_mobilenet_data.py --data-root ../data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.network_tools import (channel_table, degrees_at, depthwise_regions,  # noqa: E402
                  load_model, rotation, spread)
from experiments.dead_channels.dead_channels_from_weights import SWEEP  # noqa: E402
from experiments.spread_and_rotation.spread_and_rotation_six_networks import BUDGETS  # noqa: E402
from experiments.range_vs_degree.range_vs_degree import evaluate, loader  # noqa: E402
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_result, write_table  # noqa: E402

NET = "cifar10_mobilenetv2_x1_0"
PUBLISHED = 93.79
CONST_REL = 1e-3      # output std below this share of the site median: constant
SEEDS = [0, 1]


def act_modules(model) -> dict[str, nn.Module]:
    """BN name -> the ReLU6 right after it"""
    leaves = [(n, m) for n, m in model.named_modules() if not list(m.children())]
    return {n: nxt for (n, m), (_, nxt) in zip(leaves, leaves[1:])
            if isinstance(m, nn.BatchNorm2d) and isinstance(nxt, nn.ReLU6)}


class Stats:
    """per-channel counts of a ReLU6 input, accumulated over batches"""

    def __init__(self, C: int):
        z = lambda: torch.zeros(C, dtype=torch.float64)  # noqa: E731
        self.n = 0
        self.on, self.cap, self.sy, self.syy = z(), z(), z(), z()
        self.lo = torch.full((C,), np.inf, dtype=torch.float64)
        self.hi = torch.full((C,), -np.inf, dtype=torch.float64)

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().transpose(0, 1).reshape(x.shape[1], -1).double()
        y = x.clamp(0, 6)
        self.n += x.shape[1]
        self.on += (x > 0).sum(1)
        self.cap += (x > 6).sum(1)
        self.sy += y.sum(1)
        self.syy += (y * y).sum(1)
        self.lo = torch.minimum(self.lo, x.min(1).values)
        self.hi = torch.maximum(self.hi, x.max(1).values)

    def mean(self) -> torch.Tensor:
        return self.sy / self.n

    def std(self) -> torch.Tensor:
        return (self.syy / self.n - self.mean() ** 2).clamp(min=0).sqrt()


def record(model, acts, dl) -> tuple[dict[str, Stats], float]:
    stats = {s: Stats(m_bn.num_features) for s, m_bn in
             ((s, dict(model.named_modules())[s]) for s in acts)}
    hooks = [m.register_forward_pre_hook(
        lambda mod, inp, s=s: stats[s].add(inp[0])) for s, m in acts.items()]
    top1, _ = evaluate(model, dl, "cpu")
    for h in hooks:
        h.remove()
    return stats, top1


def ablate(model, acts, means, sets: dict[str, np.ndarray], dl) -> float:
    """top-1 with the channels in sets[site] replaced by their mean"""
    hooks = []
    for s, idx in sets.items():
        if len(idx) == 0:
            continue
        i = torch.as_tensor(idx, dtype=torch.long)
        val = means[s][i].float().view(1, -1, 1, 1)

        def hook(mod, inp, out, i=i, val=val):
            out = out.clone()
            out[:, i] = val.expand(out.shape[0], -1, out.shape[2], out.shape[3])
            return out
        hooks.append(acts[s].register_forward_hook(hook))
    top1, _ = evaluate(model, dl, "cpu")
    for h in hooks:
        h.remove()
    return top1


def random_like(sets: dict[str, np.ndarray], sizes: dict[str, int],
                seed: int) -> dict[str, np.ndarray]:
    """same count per site, drawn from the channels NOT in the set"""
    rng = np.random.default_rng(seed)
    out = {}
    for s, idx in sets.items():
        rest = np.setdiff1d(np.arange(sizes[s]), idx)
        out[s] = rng.choice(rest, size=min(len(idx), len(rest)), replace=False)
    return out


def max_rotation(table: pd.DataFrame, regions, drop: set) -> float:
    """exp2's rotation rate, max over (delta model, G), after dropping
    every channel that is dead at either site of its region"""
    keep = table.copy()
    for s1, s2 in regions.values():
        dead_c = {c for (s, c) in drop if s in (s1, s2)}
        keep = keep[~(keep.site.isin([s1, s2]) & keep.channel_idx.isin(dead_c))]
    rot = pd.concat([rotation(degrees_at(keep, B), regions, B) for B in BUDGETS],
                    ignore_index=True)
    return float(rot.groupby(["delta_model", "G"]).rotates.mean().max())


def main() -> None:
    _stamp.use_results("dead_channels")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--calib-images", type=int, default=2000)
    args = ap.parse_args()
    root = Path(args.data_root)
    torch.set_grad_enabled(False)
    banner(["EXP 2c - CIFAR MobileNetV2 dead channels, on real images"])

    model = load_model(NET)
    acts = act_modules(model)
    mods = dict(model.named_modules())
    sizes = {s: mods[s].num_features for s in acts}
    test = loader("cifar10", "test", None, 500, root)

    # 1 + 2: baseline and per-channel statistics on all test images
    st, base = record(model, acts, test)
    print(f"  baseline top-1 {base:.2f}% (published {PUBLISHED})", flush=True)
    calib, _ = record(model, acts,
                      loader("cifar10", "train", args.calib_images, 500, root))
    means = {s: c.mean() for s, c in calib.items()}

    # weights side, as exp2b
    table = channel_table(model, NET)
    rows = []
    for s in acts:
        g = table[table.site == s].sort_values("channel_idx")
        bn = mods[s]
        gamma = bn.weight.abs().numpy()
        rstd = np.sqrt(bn.running_var.numpy() + bn.eps)
        sc, v = g.s_c.to_numpy(), g.v_norm.to_numpy()
        t = st[s]
        sd = t.std().numpy()
        for c in range(sizes[s]):
            rows.append({
                "site": s, "channel": c, "s_c_rel": sc[c] / np.median(sc),
                "gamma_rel": gamma[c] / np.median(gamma),
                "v_rel": v[c] / np.median(v),
                "running_std_rel": rstd[c] / np.median(rstd),
                "beta": float(bn.bias[c]),
                "on_frac": float(t.on[c] / t.n), "cap_frac": float(t.cap[c] / t.n),
                "in_min": float(t.lo[c]), "in_max": float(t.hi[c]),
                "out_std_rel": sd[c] / max(np.median(sd), 1e-30),
            })
    df = pd.DataFrame(rows)
    df["off"] = df.on_frac == 0
    df["constant"] = df.out_std_rel < CONST_REL
    df["linear"] = (df.in_min > 0) & (df.in_max < 6)
    df["capped"] = df.cap_frac > 0
    for cut in SWEEP:
        df[f"weight_dead_{cut:g}"] = df.s_c_rel < cut

    wd = df["weight_dead_0.001"]
    agree = {
        "weight_dead": int(wd.sum()),
        "data_constant": int(df.constant.sum()),
        "data_off": int(df.off.sum()),
        "weight_dead_and_constant": int((wd & df.constant).sum()),
        "weight_dead_share_constant": float(df.constant[wd].mean()),
        "constant_share_weight_dead": float(wd[df.constant].mean()),
        "weight_dead_median_gamma_rel": float(df.gamma_rel[wd].median()),
        "weight_dead_median_v_rel": float(df.v_rel[wd].median()),
        "weight_dead_median_running_std_rel": float(df.running_std_rel[wd].median()),
        "weight_dead_share_beta_le_0": float((df.beta[wd] <= 0).mean()),
        "constant_share_off": float(df.off[df.constant].mean()),
        "channels": len(df), "sites": len(acts),
        "share_linear": float(df.linear.mean()),
        "share_capped_channels": float(df.capped.mean()),
        "share_values_capped": float((df.cap_frac).mean()),
    }
    live = df[~df.constant]
    agree["median_site_spread_live_data"] = float(
        pd.Series([spread(pd.Series(g.s_c_rel.to_numpy()))
                   for _, g in live.groupby("site")]).median())
    print(pd.Series(agree).to_string(), flush=True)

    # 5: mean ablation
    def sets_for(col):
        return {s: df[(df.site == s) & df[col]].channel.to_numpy() for s in acts}

    abl = [{"set": "none", "kind": "baseline", "channels": 0, "top1": base}]
    for col in ["constant", "weight_dead_0.001", "weight_dead_0.01",
                "weight_dead_0.1"]:
        sets = sets_for(col)
        n = int(sum(len(v) for v in sets.values()))
        acc = ablate(model, acts, means, sets, test)
        abl.append({"set": col, "kind": "this set", "channels": n, "top1": acc})
        print(f"  ablate {col:18s} {n:5d} ch -> {acc:.2f}%", flush=True)
        for seed in SEEDS:
            acc = ablate(model, acts, means, random_like(sets, sizes, seed), test)
            abl.append({"set": col, "kind": f"random seed {seed}",
                        "channels": n, "top1": acc})
            print(f"     random, seed {seed}           -> {acc:.2f}%", flush=True)
    abl = pd.DataFrame(abl)

    # 6: rotation without the dead channels
    regions = depthwise_regions(model)
    rot = {"all channels": max_rotation(table, regions, set())}
    for col in ["constant", "weight_dead_0.001"]:
        drop = set(zip(df.site[df[col]], df.channel[df[col]]))
        rot[f"without {col}"] = max_rotation(table, regions, drop)
    print("  max rotation:", {k: round(v, 3) for k, v in rot.items()}, flush=True)

    verdict = (f"{NET}: top-1 {base:.2f} (published {PUBLISHED}). "
               f"{agree['data_constant']} of {agree['channels']} channels at "
               f"activation sites have constant output on test images; "
               f"{100 * agree['weight_dead_share_constant']:.0f}% of exp2b's "
               f"weight-dead channels are among them. Ablating the constant ones: "
               f"{abl.top1[abl.set.eq('constant') & abl.kind.eq('this set')].iloc[0]:.2f}%. "
               f"Max rotation {rot['all channels']:.3f} -> "
               f"{rot['without constant']:.3f} without them.")
    write_table("exp2c_channels", df, {"experiment": "EXP 2c per channel",
                                       "net": NET}, None)
    write_table("exp2c_ablation", abl, {"experiment": "EXP 2c mean ablation",
                                        "calib_images": args.calib_images})
    write_result("exp2c_mobilenet_data",
                 {"agreement": agree, "rotation": rot, "baseline_top1": base,
                  "published_top1": PUBLISHED, "const_rel": CONST_REL},
                 verdict)
    banner([verdict[i:i + 100] for i in range(0, len(verdict), 100)])


if __name__ == "__main__":
    main()
