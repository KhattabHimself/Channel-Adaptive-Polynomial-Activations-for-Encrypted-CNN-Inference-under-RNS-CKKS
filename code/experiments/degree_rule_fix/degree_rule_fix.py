"""
A degree rule that keeps a MobileNetV2 accurate, and what grouping saves under it.

The accuracy check of thesis section 6.14.3 showed that the weights-only degree
rule leaves the CIFAR-10 MobileNetV2 at chance: it gives the first activation
of every block a degree far too low. This experiment looks for a rule that
works, picks it on a validation split, and then measures on the test set what
grouping saves under it.

Splits (CIFAR-10), fixed here and never mixed:
  calibration   training images 0 to 1,999        intervals and channel types
  validation    training images 40,000 to 41,999  choosing the rule
  test          the 10,000 test images            the reported accuracy, once

Every rule uses each channel's observed range on the calibration images,
widened by a margin, as its interval. A channel whose widened range does not
contain zero is exactly linear there (the identity, or zero) and needs no
polynomial: it runs at degree 1. The rules differ in the degree they give the
channels that do cross zero:

  uniform      the same degree d for every crossing channel
  thesis       the degree of the thesis rule (weights only, section 4.4)
  site_norm    the thesis rule with the sensitivity divided by its median
               over the crossing channels of the same site, so that every
               site gets the same budget for its typical channel; then a
               floor and a cap on the degree

Three ways of running a rule are scored:
  layerwise    every ciphertext of a layer at the largest degree in the layer
  grouped      channels sorted and grouped as in the time experiment; every
               ciphertext at the largest degree in its group
  per_channel  every channel at its own degree (not realisable; a reference)

Activation time of a setting = sum over its ciphertexts of the measured time
of one activation at that degree (results/grouping_time/repeat_runs, the five
OpenFHE timing runs, 12-level chain). Activations only.

Selection rule, fixed before the sweep was run: among the rules whose GROUPED
validation accuracy is within the bar (1.0 and 2.0 points) of the unmodified
network's validation accuracy, take the one with the least grouped activation
time. The same is done for LAYERWISE on its own, so that grouping is compared
with the best layerwise setting at the same accuracy bar and not only with
the layerwise version of its own rule.

run (from code/):
    python -m experiments.degree_rule_fix.degree_rule_fix sweep --data-root data
    python -m experiments.degree_rule_fix.degree_rule_fix test  --data-root data
    python -m experiments.degree_rule_fix.degree_rule_fix retime --data-root data

The third stage was added after the first two had run. The first version
counted the layerwise baseline with fewer ciphertexts than the grouped
setting at the stride-2 blocks (62 against 67), which understated the saving.
`retime` recomputes every time from the saved rules with one count for both,
checks that the selection does not change, and adds the comparison after the
dead channels are pruned. Accuracies are not affected by the count.
"""

from __future__ import annotations

import argparse
import json
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
from src.network_tools import (channel_table, cheb_coeffs, degrees_at,  # noqa: E402
                               depthwise_regions, load_model, required_degree)
from src.run_stamp import banner, write_table  # noqa: E402
from experiments.grouping_accuracy_proxy.grouping_accuracy_proxy import install  # noqa: E402
from experiments.grouping_time.grouping_time import (  # noqa: E402
    SLOTS, dp_place_shared, kappa_sqrt2d, shared_order)
from experiments.range_vs_degree.range_vs_degree import NORM  # noqa: E402

NET = "cifar10_mobilenetv2_x1_0"
BASE_GRID = 32 * 32
CALIB = (0, 2000)
VALID = (40000, 42000)
BARS = (1.0, 2.0)
TIMING = _stamp.PROJECT_RESULTS / "grouping_time" / "repeat_runs"


# ----------------------------------------------------------------- data
def tensors(root: Path, train: bool, lo: int | None = None, hi: int | None = None):
    import torchvision
    import torchvision.transforms as T
    mean, std = NORM["cifar10"]
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True,
                                      transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    idx = range(lo or 0, hi if hi is not None else len(ds))
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx])
    return xs, ys


@torch.no_grad()
def score(model, xs, ys, batch: int = 500) -> tuple[float, float]:
    ok = bad = 0
    for i in range(0, len(xs), batch):
        out = model(xs[i:i + batch])
        bad += (~torch.isfinite(out)).any(1).sum().item()
        ok += (out.argmax(1) == ys[i:i + batch]).sum().item()
    return 100.0 * ok / len(xs), bad / len(xs)


# ------------------------------------------------------------- the rules
class Sites:
    """everything about the activation sites that does not depend on the rule"""

    def __init__(self, model, calib_x):
        self.table = channel_table(model, NET)
        regions = depthwise_regions(model)
        shapes, hooks = {}, []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                hooks.append(mod.register_forward_hook(
                    lambda _m, _i, out, name=name: shapes.__setitem__(name, tuple(out.shape[1:]))))
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
        for h in hooks:
            h.remove()
        self.acts = install(model)
        self.G = {s: math.ceil(BASE_GRID * math.ceil(np.prod(shapes[s]) / BASE_GRID) / SLOTS)
                  for s in self.acts}
        self.hw = {s: shapes[s][1] * shapes[s][2] for s in self.acts}
        in_region = {s for pair in regions.values() for s in pair}
        units = [list(pair) for pair in regions.values()]
        units += [[s] for s in self.acts if s not in in_region]
        self.units = [u for u in ([s for s in u if s in self.acts] for u in units) if u]
        self.vnorm = {s: torch.tensor(g.sort_values("channel_idx").v_norm.to_numpy(),
                                      dtype=torch.float32)
                      for s, g in self.table.groupby("site") if s in self.acts}
        for a in self.acts.values():
            a.reset_calibration()
            a.mode = "calibrate"
        score(model, calib_x, torch.zeros(len(calib_x), dtype=torch.long))
        self.lohi = {}
        for s, a in self.acts.items():
            _mu, _sd, lo, hi = a.stats()
            self.lohi[s] = (lo.clone(), hi.clone())
            a.mode = "relu"
        self._cache: dict = {}

    def interval(self, s: str, margin: float):
        """center, half-width, and the kind of every channel: 0 crosses zero,
        +1 always positive (identity), -1 never positive (zero)"""
        lo, hi = self.lohi[s]
        pad = margin * (hi - lo)
        lo_p, hi_p = lo - pad, hi + pad
        center, half = (hi_p + lo_p) / 2, ((hi_p - lo_p) / 2).clamp_min(1e-6)
        kind = torch.zeros(len(lo), dtype=torch.long)
        kind[lo_p >= 0] = 1
        kind[hi_p <= 0] = -1
        return center, half, kind

    def degrees(self, rule: dict) -> dict[str, list[int]]:
        """per-channel degree at every site under a rule"""
        key = json.dumps(rule, sort_keys=True)
        if key in self._cache:
            return self._cache[key]
        out = {}
        thesis = None
        if rule["rule"] == "thesis":
            df = degrees_at(self.table, rule["B"])
            thesis = {s: g.sort_values("channel_idx").degree.to_numpy()
                      for s, g in df.groupby("site") if s in self.acts}
        for s in self.acts:
            center, half, kind = self.interval(s, rule["margin"])
            cross = kind == 0
            d = np.ones(len(kind), dtype=int)
            if rule["rule"] == "uniform":
                d[cross.numpy()] = rule["d"]
            elif rule["rule"] == "thesis":
                d[cross.numpy()] = thesis[s][cross.numpy()]
            else:
                sens = half * self.vnorm[s]
                med = float(sens[cross].median()) if cross.any() else 1.0
                t0 = (-center / half)
                for c in np.flatnonzero(cross.numpy()):
                    need = required_degree(float(sens[c]) / max(med, 1e-12), float(t0[c]),
                                           rule["B"], max_degree=rule["cap"])
                    d[c] = min(max(need, rule["floor"]), rule["cap"])
            out[s] = d.tolist()
        self._cache[key] = out
        return out


def blocks_for(sites: list[list[int]], n_groups: int, cap: int, cost=kappa_sqrt2d):
    """the groups of dp_place_shared, as lists of channel indices"""
    n = len(sites[0])
    order = shared_order(sites)
    seg = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        running = [0] * len(sites)
        for k in range(i + 1, n + 1):
            c = order[k - 1]
            for si, s in enumerate(sites):
                if s[c] > running[si]:
                    running[si] = s[c]
            seg[i][k] = sum(cost(m) for m in running)
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
    ref_cost, _ = dp_place_shared(sites, n_groups, cap, cost)
    assert abs(dp[n_groups][n] - ref_cost) < 1e-9, "grouping differs from grouping_time"
    sizes, i = [], n
    for j in range(n_groups, 0, -1):
        sizes.append(i - back[j][i])
        i = back[j][i]
    out, pos = [], 0
    for sz in sizes[::-1]:
        out.append(list(order[pos:pos + sz]))
        pos += sz
    return out


def settings(st: Sites, deg: dict[str, list[int]], cost=kappa_sqrt2d):
    """per-channel degrees for the three settings, and the degree of every
    ciphertext (for the time) under layerwise and grouped"""
    assign = {"per_channel": deg, "layerwise": {}, "grouped": {}}
    cts = {"layerwise": [], "grouped": []}
    for u in st.units:
        sites = [deg[s] for s in u]
        G = max(st.G[s] for s in u)
        # Both settings are counted with G ciphertexts at every site of the
        # unit, G being the larger of its sites. This is the convention of the
        # time experiment (thesis section 5.5): it overstates the count at the
        # second site of the stride-2 blocks, by the same amount for both.
        for s, d in zip(u, sites):
            assign["layerwise"][s] = [max(d)] * len(d)
            cts["layerwise"] += [max(d)] * G
        if G < 2 or len({len(x) for x in sites}) != 1:
            for s, d in zip(u, sites):
                assign["grouped"][s] = [max(d)] * len(d)
                cts["grouped"] += [max(d)] * G
            continue
        blocks = blocks_for(sites, G, math.ceil(len(sites[0]) / G), cost)
        for s, d in zip(u, sites):
            a = list(d)
            for b in blocks:
                m = max(d[c] for c in b)
                for c in b:
                    a[c] = m
                cts["grouped"].append(m)
            assign["grouped"][s] = a
    return assign, cts


def settings_pruned(st: Sites, deg: dict[str, list[int]], margin: float, cost=kappa_sqrt2d):
    """the same, after removing the channels that need no activation at all.

    A channel whose activation output is zero on every calibration image can
    be deleted, with its slots. In a two-site unit the channel is deleted if
    it is zero at either site: zero at the second site means nothing reads it,
    and zero at the first means the second site sees a constant, which folds
    into the next layer's bias. Returns the per-channel degrees (deleted
    channels keep degree 1; they are exactly zero or constant either way) and
    the ciphertext degrees."""
    assign = {"layerwise": {}, "grouped": {}}
    cts = {"layerwise": [], "grouped": []}
    removed = total = 0
    for u in st.units:
        kinds = [st.interval(s, margin)[2].numpy() for s in u]
        dead = np.zeros(len(kinds[0]), dtype=bool)
        if len({len(k) for k in kinds}) == 1:
            for k in kinds:
                dead |= k == -1
        else:
            dead = None
        for si, s in enumerate(u):
            d = np.asarray(deg[s])
            gone = dead if dead is not None else (kinds[si] == -1)
            keep = np.flatnonzero(~gone)
            removed += int(gone.sum())
            total += len(d)
            G = math.ceil(BASE_GRID * math.ceil(len(keep) * st.hw[s] / BASE_GRID) / SLOTS) if len(keep) else 0
            top = int(d[keep].max()) if len(keep) else 1
            a = np.ones(len(d), dtype=int)
            a[keep] = top
            assign["layerwise"][s] = a.tolist()
            cts["layerwise"] += [top] * G
            st._pruned_G = getattr(st, "_pruned_G", {})
            st._pruned_G[s] = (G, keep)
        Gu = max(st._pruned_G[s][0] for s in u)
        same = dead is not None
        if Gu < 2 or not same:
            for s in u:
                G, keep = st._pruned_G[s]
                assign["grouped"][s] = assign["layerwise"][s]
                cts["grouped"] += [int(np.asarray(deg[s])[keep].max()) if len(keep) else 1] * G
            continue
        keep = st._pruned_G[u[0]][1]
        # as above, both settings with Gu ciphertexts at every site of the unit
        extra = sum(Gu - st._pruned_G[s][0] for s in u)
        sites = [[deg[s][c] for c in keep] for s in u]
        blocks = blocks_for(sites, Gu, math.ceil(len(keep) / Gu), cost)
        for s, d in zip(u, sites):
            a = np.ones(len(deg[s]), dtype=int)
            for b in blocks:
                m = max(d[c] for c in b)
                a[keep[b]] = m
                cts["grouped"].append(m)
            assign["grouped"][s] = a.tolist()
            cts["layerwise"] += [max(d)] * (Gu - st._pruned_G[s][0])
        del extra
    return assign, cts, removed, total


def configure(st: Sites, assign: dict[str, list[int]], margin: float) -> None:
    for s, act in st.acts.items():
        center, half, kind = st.interval(s, margin)
        d = assign[s]
        t0 = -center / half
        coef = torch.zeros(len(d), int(max(d)) + 1)
        for c in range(len(d)):
            if kind[c] == 1:                      # identity: h * (t - t0) = x
                coef[c, 0], coef[c, 1] = -t0[c], 1.0
            elif kind[c] == 0:
                cc = cheb_coeffs(int(d[c]), int(round(float(t0[c]) * 1000)))
                coef[c, :len(cc)] = torch.tensor(cc, dtype=torch.float32)
        act.coef = coef
        act.center, act.half = center.view(1, -1, 1, 1), half.view(1, -1, 1, 1)
        act.mode = "poly"


def relu_mode(st: Sites) -> None:
    for a in st.acts.values():
        a.mode = "relu"


# ----------------------------------------------------------------- time
def timing_runs() -> list[dict[int, float]]:
    runs = []
    for r in sorted(TIMING.glob("run_*/"), key=lambda p: int(p.name.split("_")[1])):
        t = pd.read_csv(r / "exp4_degree_timing.csv", comment="#")
        t = t[t.chain_depth == t.chain_depth.min()]
        runs.append(dict(zip(t.degree.astype(int), t.seconds_median)))
    assert len(runs) == 5, f"expected five timing runs, found {len(runs)}"
    return runs


def seconds(cts: list[int], runs) -> np.ndarray:
    return np.array([sum(r[d] for d in cts) for r in runs])


def mean_time_cost(runs):
    mean = {d: float(np.mean([r[d] for r in runs])) for d in runs[0]}
    return lambda d: mean[int(d)]


# ---------------------------------------------------------------- rules
def candidate_rules() -> list[dict]:
    rules = []
    for margin in (0.1, 0.25, 0.5):
        for d in (15, 31):
            rules.append({"rule": "uniform", "d": d, "margin": margin})
    rules.append({"rule": "uniform", "d": 7, "margin": 0.25})
    rules.append({"rule": "uniform", "d": 63, "margin": 0.25})
    for B in (0.01, 0.1):
        rules.append({"rule": "thesis", "B": B, "margin": 0.25})
    for B in (0.08, 0.04, 0.02, 0.01):
        for floor in (1, 7, 15):
            rules.append({"rule": "site_norm", "B": B, "floor": floor, "cap": 31, "margin": 0.25})
    for B in (0.02, 0.01, 0.005):
        rules.append({"rule": "site_norm", "B": B, "floor": 7, "cap": 63, "margin": 0.25})
    return rules


def label(rule: dict) -> str:
    if rule["rule"] == "uniform":
        return f"uniform d={rule['d']} margin={rule['margin']}"
    if rule["rule"] == "thesis":
        return f"thesis B={rule['B']} margin={rule['margin']}"
    return (f"site_norm B={rule['B']} floor={rule['floor']} cap={rule['cap']} "
            f"margin={rule['margin']}")


def row_for(st, rule, setting, acc, nonfinite, base, cts, runs, n_images, split, deg):
    alld = np.concatenate([np.asarray(deg[s]) for s in st.acts])
    kinds = torch.cat([st.interval(s, rule["margin"])[2] for s in st.acts])
    out = {"rule": label(rule), "rule_json": json.dumps(rule, sort_keys=True),
           "setting": setting, "split": split, "n_images": n_images,
           "top1": acc, "unmodified_top1": base, "points_lost": base - acc,
           "nonfinite_frac": nonfinite,
           "crossing_channels": int((kinds == 0).sum()),
           "identity_channels": int((kinds == 1).sum()),
           "zero_channels": int((kinds == -1).sum()),
           "median_degree_crossing": float(np.median(alld[(kinds == 0).numpy()])),
           "ciphertexts": len(cts) if cts is not None else np.nan}
    if cts is not None:
        sec = seconds(cts, runs)
        out["activation_s_mean"], out["activation_s_std"] = sec.mean(), sec.std(ddof=1)
    return out


# ---------------------------------------------------------------- stages
def stage_sweep(args) -> None:
    root = Path(args.data_root)
    runs = timing_runs()
    model = load_model(NET)
    cx, _ = tensors(root, True, *CALIB)
    st = Sites(model, cx)
    vx, vy = tensors(root, True, *VALID)
    base, _ = score(model, vx, vy)
    print(f"unmodified, validation ({len(vx)} images): {base:.2f}%", flush=True)
    rows = []
    for rule in candidate_rules():
        deg = st.degrees(rule)
        assign, cts = settings(st, deg)
        for setting in ("layerwise", "grouped"):
            t = time.time()
            configure(st, assign[setting], rule["margin"])
            acc, nf = score(model, vx, vy)
            relu_mode(st)
            rows.append(row_for(st, rule, setting, acc, nf, base, cts[setting], runs,
                                len(vx), "validation", deg))
            print(f"  {label(rule):<48} {setting:<10} {acc:6.2f}%  lost {base - acc:6.2f}  "
                  f"time {rows[-1]['activation_s_mean']:7.2f}s  ({time.time() - t:.0f}s)",
                  flush=True)
            if rule["rule"] == "uniform":
                # layerwise and grouped differ only in where the linear channels sit
                pass
    out = pd.DataFrame(rows)
    picks = select(out)
    verdict = "Chosen on validation: " + "; ".join(
        f"bar {bar} {setting}: {p['rule']} ({p['top1']:.2f}%, {p['activation_s_mean']:.1f}s)"
        if p else f"bar {bar} {setting}: none"
        for (bar, setting), p in picks.items())
    write_table("sweep_validation", out, {
        "experiment": "degree rule fix, sweep on the validation split",
        "network": NET, "calibration_images": f"train {CALIB}", "validation_images": f"train {VALID}"},
        verdict)
    banner([verdict])


def select(sweep: pd.DataFrame) -> dict:
    """the selection rule of the module docstring"""
    picks = {}
    for bar in BARS:
        for setting in ("grouped", "layerwise"):
            ok = sweep[(sweep.setting == setting) & (sweep.points_lost <= bar)]
            picks[(bar, setting)] = (ok.loc[ok.activation_s_mean.idxmin()].to_dict()
                                     if len(ok) else None)
    return picks


def stage_test(args) -> None:
    root = Path(args.data_root)
    runs = timing_runs()
    sweep = pd.read_csv(_stamp.RESULTS_DIR / "sweep_validation.csv", comment="#")
    picks = select(sweep)
    rules = []
    for p in picks.values():
        if p and p["rule_json"] not in [json.dumps(r, sort_keys=True) for r in rules]:
            rules.append(json.loads(p["rule_json"]))
    model = load_model(NET)
    cx, _ = tensors(root, True, *CALIB)
    st = Sites(model, cx)
    tx, ty = tensors(root, False)
    base, _ = score(model, tx, ty)
    print(f"unmodified, test ({len(tx)} images): {base:.2f}%", flush=True)
    rows, per_run = [], []
    for rule in rules:
        deg = st.degrees(rule)
        assign, cts = settings(st, deg)
        assign_t, cts_t = settings(st, deg, cost=mean_time_cost(runs))
        assign["grouped_by_time"], cts["grouped_by_time"] = assign_t["grouped"], cts_t["grouped"]
        cts["per_channel"] = None
        for setting in ("layerwise", "grouped", "grouped_by_time", "per_channel"):
            t = time.time()
            configure(st, assign[setting], rule["margin"])
            acc, nf = score(model, tx, ty)
            relu_mode(st)
            rows.append(row_for(st, rule, setting, acc, nf, base, cts[setting], runs,
                                len(tx), "test", deg))
            print(f"  {label(rule):<48} {setting:<15} {acc:6.2f}%  lost {base - acc:6.2f}  "
                  f"({time.time() - t:.0f}s)", flush=True)
        lw = seconds(cts["layerwise"], runs)
        for name in ("grouped", "grouped_by_time"):
            g = seconds(cts[name], runs)
            sv = 100 * (1 - g / lw)
            per_run.append({"rule": label(rule), "grouping": name,
                            "layerwise_s_mean": lw.mean(), "grouped_s_mean": g.mean(),
                            "saving_pct_mean": sv.mean(), "saving_pct_std": sv.std(ddof=1),
                            **{f"saving_pct_run_{i + 1}": v for i, v in enumerate(sv)}})
    out = pd.DataFrame(rows)
    sv = pd.DataFrame(per_run)
    # grouping against the best layerwise setting at the same bar
    cross = []
    for bar in BARS:
        g, l = picks[(bar, "grouped")], picks[(bar, "layerwise")]
        if not g or not l:
            continue
        tg = out[(out.rule == g["rule"]) & (out.setting == "grouped")].iloc[0]
        tl = out[(out.rule == l["rule"]) & (out.setting == "layerwise")].iloc[0]
        deg_g, deg_l = st.degrees(json.loads(g["rule_json"])), st.degrees(json.loads(l["rule_json"]))
        sg = seconds(settings(st, deg_g)[1]["grouped"], runs)
        sl = seconds(settings(st, deg_l)[1]["layerwise"], runs)
        s = 100 * (1 - sg / sl)
        cross.append({"bar_points": bar, "grouped_rule": g["rule"], "layerwise_rule": l["rule"],
                      "grouped_test_top1": tg.top1, "layerwise_test_top1": tl.top1,
                      "grouped_s_mean": sg.mean(), "layerwise_s_mean": sl.mean(),
                      "saving_pct_mean": s.mean(), "saving_pct_std": s.std(ddof=1)})
    cr = pd.DataFrame(cross)
    verdict = (f"Test set, unmodified {base:.2f}%. " + " ".join(
        f"Bar {r.bar_points:g}: grouped [{r.grouped_rule}] {r.grouped_test_top1:.2f}% in "
        f"{r.grouped_s_mean:.1f}s against best layerwise [{r.layerwise_rule}] "
        f"{r.layerwise_test_top1:.2f}% in {r.layerwise_s_mean:.1f}s: saving "
        f"{r.saving_pct_mean:.1f} +/- {r.saving_pct_std:.1f}%." for r in cr.itertuples()))
    meta = {"experiment": "degree rule fix, test set", "network": NET}
    write_table("test_accuracy", out, meta, verdict)
    write_table("test_saving_same_rule", sv, meta,
                "Saving of grouping against the layerwise version of the same rule: " + "; ".join(
                    f"[{r.rule}] {r.grouping}: {r.saving_pct_mean:.1f} +/- {r.saving_pct_std:.1f}%"
                    for r in sv.itertuples()))
    write_table("test_saving_best_layerwise", cr, meta, verdict)
    banner([verdict])


def stage_retime(args) -> None:
    """Recompute every time from the saved rules with the corrected ciphertext
    count, check that the selection is unchanged, add the accounting after
    pruning, and score the grouping after pruning for the chosen rules."""
    root = Path(args.data_root)
    runs = timing_runs()
    sweep = pd.read_csv(_stamp.RESULTS_DIR / "sweep_validation.csv", comment="#")
    test = pd.read_csv(_stamp.RESULTS_DIR / "test_accuracy.csv", comment="#")
    before = {k: (v["rule"] if v else None) for k, v in select(sweep).items()}
    model = load_model(NET)
    cx, _ = tensors(root, True, *CALIB)
    st = Sites(model, cx)

    def retime(df):
        df = df.copy()
        for i, r in df.iterrows():
            rule = json.loads(r.rule_json)
            _, cts = settings(st, st.degrees(rule))
            key = "grouped" if r.setting.startswith("grouped") else r.setting
            if key in cts:
                sec = seconds(cts[key], runs)
                df.loc[i, ["activation_s_mean", "activation_s_std", "ciphertexts"]] = (
                    sec.mean(), sec.std(ddof=1), len(cts[key]))
        return df

    sweep, test = retime(sweep), retime(test)
    picks = select(sweep)
    after = {k: (v["rule"] if v else None) for k, v in picks.items()}
    print("selection before the correction:", before, flush=True)
    print("selection after the correction: ", after, flush=True)
    tested = set(test.rule)
    assert all(v in tested for v in after.values() if v), "a newly selected rule was not tested"
    tx, ty = tensors(root, False)
    base = float(test.unmodified_top1.iloc[0])
    rows, same, cross = [], [], []
    for rj in dict.fromkeys(test.rule_json):
        rule = json.loads(rj)
        deg = st.degrees(rule)
        _, cts = settings(st, deg)
        lw, g = seconds(cts["layerwise"], runs), seconds(cts["grouped"], runs)
        asg_p, cts_p, removed, total = settings_pruned(st, deg, rule["margin"])
        lwp, gp = seconds(cts_p["layerwise"], runs), seconds(cts_p["grouped"], runs)
        acc_p = {}
        if label(rule) in set(v for v in after.values() if v):
            for setting in ("layerwise", "grouped"):
                configure(st, asg_p[setting], rule["margin"])
                acc_p[setting], _ = score(model, tx, ty)
                relu_mode(st)
                print(f"  after pruning, {label(rule)} {setting}: {acc_p[setting]:.2f}%", flush=True)
        t = test[test.rule == label(rule)].set_index("setting").top1
        for name, a, b, n_a, n_b, acc in (
                ("as packed", lw, g, len(cts["layerwise"]), len(cts["grouped"]),
                 (t["layerwise"], t["grouped"])),
                ("dead channels pruned first", lwp, gp, len(cts_p["layerwise"]),
                 len(cts_p["grouped"]), (acc_p.get("layerwise"), acc_p.get("grouped")))):
            sv = 100 * (1 - b / a)
            same.append({"rule": label(rule), "accounting": name,
                         "channels_removed": removed if "pruned" in name else 0,
                         "channels": total, "ciphertexts_layerwise": n_a, "ciphertexts_grouped": n_b,
                         "layerwise_test_top1": acc[0], "grouped_test_top1": acc[1],
                         "unmodified_test_top1": base,
                         "layerwise_s_mean": a.mean(), "grouped_s_mean": b.mean(),
                         "saving_pct_mean": sv.mean(), "saving_pct_std": sv.std(ddof=1),
                         **{f"saving_pct_run_{i + 1}": v for i, v in enumerate(sv)}})
    for bar in BARS:
        g, l = picks[(bar, "grouped")], picks[(bar, "layerwise")]
        if not g or not l:
            continue
        cg = settings(st, st.degrees(json.loads(g["rule_json"])))[1]["grouped"]
        cl = settings(st, st.degrees(json.loads(l["rule_json"])))[1]["layerwise"]
        sg, sl = seconds(cg, runs), seconds(cl, runs)
        sv = 100 * (1 - sg / sl)
        tg = test[(test.rule == g["rule"]) & (test.setting == "grouped")].iloc[0]
        tl = test[(test.rule == l["rule"]) & (test.setting == "layerwise")].iloc[0]
        cross.append({"bar_points": bar, "grouped_rule": g["rule"], "layerwise_rule": l["rule"],
                      "grouped_validation_lost": g["points_lost"],
                      "layerwise_validation_lost": l["points_lost"],
                      "grouped_test_top1": tg.top1, "layerwise_test_top1": tl.top1,
                      "grouped_test_lost": base - tg.top1, "layerwise_test_lost": base - tl.top1,
                      "grouped_s_mean": sg.mean(), "layerwise_s_mean": sl.mean(),
                      "saving_pct_mean": sv.mean(), "saving_pct_std": sv.std(ddof=1)})
    sm, cr = pd.DataFrame(same), pd.DataFrame(cross)
    meta = {"experiment": "degree rule fix, times recomputed with one ciphertext count for both settings",
            "network": NET}
    v_same = "Grouping against the layerwise version of the same rule: " + "; ".join(
        f"[{r.rule}] {r.accounting}: {r.saving_pct_mean:.1f} +/- {r.saving_pct_std:.1f}%"
        for r in sm.itertuples())
    v_cross = f"Test set, unmodified {base:.2f}%. " + " ".join(
        f"Bar {r.bar_points:g}: grouped [{r.grouped_rule}] {r.grouped_test_top1:.2f}% in "
        f"{r.grouped_s_mean:.1f}s against the cheapest layerwise setting [{r.layerwise_rule}] "
        f"{r.layerwise_test_top1:.2f}% in {r.layerwise_s_mean:.1f}s: saving "
        f"{r.saving_pct_mean:.1f} +/- {r.saving_pct_std:.1f}%." for r in cr.itertuples())
    write_table("sweep_validation", sweep, meta, "Chosen on validation: " + "; ".join(
        f"bar {b} {s_}: {v}" for (b, s_), v in after.items()))
    write_table("test_accuracy", test, meta, v_cross)
    write_table("test_saving_same_rule", sm, meta, v_same)
    write_table("test_saving_best_layerwise", cr, meta, v_cross)
    banner([v_same, v_cross])


def main() -> None:
    _stamp.use_results("degree_rule_fix")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["sweep", "test", "retime"])
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    torch.manual_seed(0)
    banner([f"DEGREE RULE FIX, stage {args.stage}", NET])
    {"sweep": stage_sweep, "test": stage_test, "retime": stage_retime}[args.stage](args)


if __name__ == "__main__":
    main()
