"""
Shared code for the experiments. Copied from the thesis repo (src/phase1 and
src/phase3) so this folder runs on colab without the repo.
"""

from __future__ import annotations

import math
import os
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# level cost, copied from src/phase1/level_checker.py
def delta_A(d: int) -> int:
    """Paterson-Stockmeyer LOWER bound on polynomial depth: ceil(log2(d+1))."""
    if d < 1:
        raise ValueError(f"degree must be >= 1, got {d}")
    return math.ceil(math.log2(d + 1))


def delta_B(d: int) -> int:
    """Practical UPPER bound on polynomial depth: ceil(log2 d) + 1."""
    if d < 1:
        raise ValueError(f"degree must be >= 1, got {d}")
    return math.ceil(math.log2(d)) + 1


#: The two thesis models. Every result is reported under both (PROTOCOL rule 6).
DELTA_MODELS = {"A": delta_A, "B": delta_B}


# degree under budget, copied from src/phase3/invariants.py
SIGMA = 3.0
MAX_DEGREE = 63


@lru_cache(maxsize=None)
def eps_rel(degree: int, t0_quantised: int) -> float:
    """sup|g - p_d| on [-1,1] for g(t) = max(0, t - t0), t0 on a 1e-3 grid."""
    t0 = t0_quantised / 1000.0
    if t0 <= -1.0 or t0 >= 1.0:
        return 0.0
    g = lambda t: np.maximum(0.0, t - t0)  # noqa: E731
    series = np.polynomial.chebyshev.Chebyshev.interpolate(g, degree,
                                                           domain=[-1, 1])
    grid = np.linspace(-1.0, 1.0, 4001)
    return float(np.max(np.abs(g(grid) - series(grid))))


def required_degree(s_c: float, t0: float, budget: float,
                    max_degree: int = MAX_DEGREE) -> int:
    """Smallest degree meeting s_c * eps_rel(d) <= budget."""
    q = int(round(t0 * 1000))
    for d in range(1, max_degree + 1):
        if s_c * eps_rel(d, q) <= budget:
            return d
    return max_degree


# checkpoints
#: Aliases so the archived runs can be reproduced under their original names.
ALIASES = {"resnet20": "cifar10_resnet20", "mobilenetv2": "imagenet_mobilenetv2"}


def _cached_hub_repo(repo: str) -> str:
    """the local copy of a torch.hub repo that prefetch.py already downloaded"""
    found = sorted(Path(torch.hub.get_dir()).glob(repo.replace("/", "_") + "_*"))
    if not found:
        raise ValueError(f"{repo} is not in the cache at {torch.hub.get_dir()}. "
                         f"Run prefetch.py on the login node first.")
    return str(found[0])


def load_model(name: str) -> nn.Module:
    """load a pretrained checkpoint by name

    With THESIS_OFFLINE=1 (the DRAC job scripts set it) the hub repo is read
    from the local cache, since compute nodes have no internet. The weights
    come from the same cache either way.
    """
    name = ALIASES.get(name, name)
    if name.startswith(("cifar10_", "cifar100_")):
        repo = "chenyaofo/pytorch-cifar-models"
        if os.environ.get("THESIS_OFFLINE") == "1":
            m = torch.hub.load(_cached_hub_repo(repo), name, source="local",
                               pretrained=True)
            m.eval()
            return m
        try:
            m = torch.hub.load(repo, name, pretrained=True, trust_repo=True)
        except Exception as e:  # noqa: BLE001  re-raised with the valid names
            try:
                valid = torch.hub.list(repo, trust_repo=True)
            except Exception:  # noqa: BLE001
                valid = ["(could not list hub entries)"]
            raise ValueError(f"cannot load {name!r} from {repo}: {e}\n"
                             f"available: {valid}") from e
    elif name == "imagenet_mobilenetv2":
        import torchvision
        m = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif name == "imagenet_mnasnet1_0":
        import torchvision
        m = torchvision.models.mnasnet1_0(
            weights=torchvision.models.MNASNet1_0_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"unknown network {name!r}")
    m.eval()
    return m


# invariants, copied from src/phase3/invariants.py
def _consumer_norms(consumer: nn.Module, n_channels: int) -> np.ndarray:
    """||V[:,c]||_2 for each channel c consumed by `consumer`."""
    W = consumer.weight.detach().cpu()
    if isinstance(consumer, nn.Linear):
        return W.norm(dim=0).numpy()
    groups = getattr(consumer, "groups", 1)
    if groups == consumer.in_channels and groups > 1:
        return W.reshape(W.shape[0], -1).norm(dim=1).numpy()
    if groups > 1:
        per = consumer.in_channels // groups
        out = np.zeros(n_channels)
        for c in range(min(n_channels, consumer.in_channels)):
            out[c] = W[:, c % per].norm().item()
        return out
    return W.transpose(0, 1).reshape(W.shape[1], -1).norm(dim=1).numpy()


def _in_channels(mod: nn.Module) -> int:
    return getattr(mod, "in_channels", getattr(mod, "in_features", -1))


def bn_sites(model: nn.Module) -> list[tuple[str, nn.BatchNorm2d, nn.Module]]:
    """(name, bn, consuming weight-bearing module) in module order"""
    mods = [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm2d, nn.Conv2d, nn.Linear))]
    sites = []
    for i, (name, mod) in enumerate(mods):
        if not isinstance(mod, nn.BatchNorm2d):
            continue
        later = [m for _, m in mods[i + 1:]
                 if isinstance(m, (nn.Conv2d, nn.Linear))]
        consumer = next((m for m in later
                         if _in_channels(m) == mod.num_features), None)
        if consumer is None:
            consumer = later[0] if later else None
        if consumer is not None:
            sites.append((name, mod, consumer))
    return sites


def channel_table(model: nn.Module, net: str,
                  sigma: float = SIGMA) -> pd.DataFrame:
    """budget-independent part of invariants(): one row per (site, channel)"""
    rows = []
    for site_index, (name, bn, consumer) in enumerate(bn_sites(model)):
        gamma = bn.weight.detach().cpu()
        beta = bn.bias.detach().cpu()
        mean = bn.running_mean.detach().cpu()
        var = bn.running_var.detach().cpu()
        vnorms = _consumer_norms(consumer, bn.num_features)
        for c in range(bn.num_features):
            denom = math.sqrt(var[c].item() + bn.eps)
            a_c = gamma[c].item() / denom
            b_c = beta[c].item() - gamma[c].item() * mean[c].item() / denom
            half = sigma * abs(a_c)
            t0 = math.inf if half <= 0 else -b_c / half
            v = float(vnorms[c]) if c < len(vnorms) else float("nan")
            rows.append({
                "net": net, "site": name, "site_index": site_index,
                "channel_idx": c, "a_c": a_c, "b_c": b_c,
                "v_norm": v, "s_c": abs(a_c) * v,
                "t0": max(-1.5, min(1.5, t0)),
            })
    return pd.DataFrame(rows)


def degrees_at(table: pd.DataFrame, budget: float) -> pd.DataFrame:
    """The table with a `degree` column for one network-wide budget B."""
    out = table.copy()
    out["degree"] = [required_degree(s, t, budget)
                     for s, t in zip(out.s_c, out.t0)]
    return out


# dispersion, copied from src/phase3/gate_b.py
def spread(series: pd.Series) -> float:
    """p90/p10 ratio, the dispersion statistic Condition 1 is stated in."""
    lo = np.percentile(series, 10)
    hi = np.percentile(series, 90)
    return float(hi / lo) if lo > 0 else float("inf")


def dispersion(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    rows = []
    for (site_index, site), g in df.groupby(["site_index", "site"]):
        row = {
            "net": g.net.iloc[0], "budget_B": budget,
            "site_index": site_index, "site": site, "n_channels": len(g),
            "s_c_spread_p90_p10": spread(g.s_c),
            "degree_min": int(g.degree.min()),
            "degree_median": float(g.degree.median()),
            "degree_max": int(g.degree.max()),
        }
        for name, model in DELTA_MODELS.items():
            classes = g.degree.map(model)
            row[f"plateau_spread_{name}"] = int(classes.max() - classes.min())
            row[f"n_plateau_classes_{name}"] = int(classes.nunique())
        rows.append(row)
    return pd.DataFrame(rows)


# conjunct (3): rotation
_ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU, nn.GELU)
BUCKET_COUNTS = [2, 4, 8]


def _is_depthwise(m: nn.Module) -> bool:
    return (isinstance(m, nn.Conv2d) and m.groups > 1
            and m.groups == m.in_channels)


def depthwise_regions(model: nn.Module) -> dict[str, list[str]]:
    """mixing-free regions of length 2, found from module STRUCTURE"""
    dw = [n for n, m in model.named_modules() if _is_depthwise(m)]

    def n_dw(prefix: str) -> int:
        if prefix == "":
            return len(dw)
        return sum(1 for d in dw if d == prefix or d.startswith(prefix + "."))

    regions: dict[str, list[str]] = {}
    for name, mod in model.named_modules():
        if name == "" or n_dw(name) != 1:
            continue
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        if n_dw(parent) == 1:
            continue                                    # not maximal
        leaves = [(f"{name}.{sub}" if sub else name, m)
                  for sub, m in mod.named_modules(remove_duplicate=False)
                  if not list(m.children())
                  and isinstance(m, (nn.Conv2d, nn.BatchNorm2d, *_ACTIVATIONS))]
        k = next((i for i, (_, m) in enumerate(leaves) if _is_depthwise(m)),
                 None)
        if k is None:
            continue
        pairs = [i for i in range(len(leaves) - 1)
                 if isinstance(leaves[i][1], nn.BatchNorm2d)
                 and isinstance(leaves[i + 1][1], _ACTIVATIONS)]
        before = [i for i in pairs if i < k]
        after = [i for i in pairs if i > k]
        if not before or not after:
            continue
        s1, s2 = leaves[before[-1]], leaves[after[0]]
        if s1[1].num_features != s2[1].num_features:
            continue
        regions[name] = [s1[0], s2[0]]
    return regions


def rotation(df: pd.DataFrame, regions: dict[str, list[str]],
             budget: float) -> pd.DataFrame:
    """does the argmax bucket rotate between a region's two sites?"""
    rows = []
    for block, (site1, site2) in regions.items():
        a = df[df.site == site1].sort_values("channel_idx")
        b = df[df.site == site2].sort_values("channel_idx")
        if len(a) != len(b) or len(a) == 0:
            continue
        d1, d2 = a.degree.to_numpy(), b.degree.to_numpy()
        order = np.argsort(-np.maximum(d1, d2), kind="stable")
        for G in BUCKET_COUNTS:
            if len(order) < G:
                continue
            buckets = np.array_split(order, G)
            for name, model in DELTA_MODELS.items():
                f1 = [max(model(int(x)) for x in d1[bk]) for bk in buckets]
                f2 = [max(model(int(x)) for x in d2[bk]) for bk in buckets]
                arg1, arg2 = int(np.argmax(f1)), int(np.argmax(f2))
                rows.append({
                    "net": df.net.iloc[0], "budget_B": budget,
                    "block": block, "G": G, "delta_model": name,
                    "site_1": site1, "site_2": site2, "n_channels": len(d1),
                    "argmax_bucket_site1": arg1,
                    "argmax_bucket_site2": arg2,
                    "rotates": bool(arg1 != arg2),
                })
    return pd.DataFrame(rows)


# Chebyshev substitution, copied from src/phase3/accuracy_sweep.py
@lru_cache(maxsize=None)
def cheb_coeffs(degree: int, t0_q: int) -> tuple:
    """Chebyshev coefficients of g(t) = max(0, t - t0) on [-1,1]."""
    t0 = t0_q / 1000.0
    g = lambda t: np.maximum(0.0, t - t0)  # noqa: E731
    return tuple(np.polynomial.chebyshev.Chebyshev.interpolate(
        g, degree, domain=[-1, 1]).coef)


class PolyAct(nn.Module):
    """ReLU, or its per-channel Chebyshev surrogate"""

    SAMPLES_PER_IMAGE = 4

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.mode = "relu"
        self.reset_calibration()
        self.coef = self.center = self.half = None

    def reset_calibration(self):
        self.sum_ = self.sumsq = self.count = None
        self.lo = self.hi = None
        self.samples: list[torch.Tensor] = []

    def forward(self, x):
        if self.mode == "calibrate":
            flat = x.transpose(0, 1).reshape(x.shape[1], -1)
            s, sq, n = flat.sum(1), (flat ** 2).sum(1), flat.shape[1]
            self.sum_ = s if self.sum_ is None else self.sum_ + s
            self.sumsq = sq if self.sumsq is None else self.sumsq + sq
            self.count = n if self.count is None else self.count + n
            lo, hi = flat.min(1).values, flat.max(1).values
            self.lo = lo if self.lo is None else torch.minimum(self.lo, lo)
            self.hi = hi if self.hi is None else torch.maximum(self.hi, hi)
            N, C = x.shape[0], x.shape[1]
            idx = torch.randint(x.shape[2] * x.shape[3],
                                (N, self.SAMPLES_PER_IMAGE), device=x.device)
            picked = x.reshape(N, C, -1).gather(
                2, idx.unsqueeze(1).expand(N, C, -1))
            self.samples.append(picked.permute(1, 0, 2).reshape(C, -1).cpu())
            return torch.relu(x)
        if self.mode == "relu":
            return torch.relu(x)
        c = self.coef.to(x.device)
        ctr, h = self.center.to(x.device), self.half.to(x.device)
        t = (x - ctr) / h
        b1 = torch.zeros_like(x)
        b2 = torch.zeros_like(x)
        for k in range(c.shape[1] - 1, 0, -1):
            b1, b2 = 2 * t * b1 - b2 + c[:, k].view(1, -1, 1, 1), b1
        return h * (t * b1 - b2 + c[:, 0].view(1, -1, 1, 1))

    def stats(self):
        mu = (self.sum_ / self.count).cpu()
        var = (self.sumsq.cpu() / self.count - mu ** 2).clamp_min(1e-12)
        return mu, var.sqrt(), self.lo.cpu(), self.hi.cpu()

    def quantiles(self, q: float):
        """Per-channel (q, 1-q) quantiles of the calibration subsample."""
        s = torch.cat(self.samples, dim=1)
        return (torch.quantile(s, q, dim=1),
                torch.quantile(s, 1.0 - q, dim=1))

    def configure(self, center, half, degrees):
        half = half.clamp_min(1e-6)
        t0 = (-center / half).clamp(-1.5, 1.5)
        K = int(max(degrees))
        coef = torch.zeros(len(degrees), K + 1)
        for ch, d in enumerate(degrees):
            cc = cheb_coeffs(int(d), int(round(float(t0[ch]) * 1000)))
            coef[ch, :len(cc)] = torch.tensor(cc, dtype=torch.float32)
        self.coef = coef
        self.center = center.view(1, -1, 1, 1)
        self.half = half.view(1, -1, 1, 1)
        self.mode = "poly"


def install_cifar_resnet(model: nn.Module) -> dict[str, PolyAct]:
    """replace ReLU use-sites with distinct PolyAct modules"""
    acts: dict[str, PolyAct] = {}
    model.relu = acts["stem"] = PolyAct("stem")
    for lname in ("layer1", "layer2", "layer3"):
        for i, blk in enumerate(getattr(model, lname)):
            a = acts[f"{lname}.{i}.bn1"] = PolyAct(f"{lname}.{i}.bn1")
            b = acts[f"{lname}.{i}.bn2"] = PolyAct(f"{lname}.{i}.bn2")
            blk.act_a, blk.act_b = a, b

            def fwd(self, x):
                identity = x
                out = self.act_a(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = out + identity
                return self.act_b(out)

            blk.forward = types.MethodType(fwd, blk)
    return acts
