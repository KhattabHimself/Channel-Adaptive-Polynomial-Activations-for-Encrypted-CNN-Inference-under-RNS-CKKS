"""
PURPOSE
    Phase 3 / Gate B core: per-channel invariants, and the degree each channel
    needs under the decided error budget.

BUDGET (decided by Khattab 2026-08-22, DECISIONS.md — no longer pending)
    Relative, sensitivity-weighted, one network-wide constant B:

        d_c = min { d : s_c * eps_rel(d ; t0_c) <= B }

    s_c    = |a_c| * ||V[:,c]||_2      sensitivity-weighted product
    k_c    = b_c / a_c                 kink location
    Omega  = [b_c - sigma|a_c|, b_c + sigma|a_c|]   the channel's interval
    eps_rel(d ; t0) = sup |g - p_d| on [-1,1], where g(t) = max(0, t - t0)

    The normalisation is the point. Mapping Omega onto [-1,1] divides the
    activation by the interval half-width, so eps_rel depends ONLY on the
    degree and the normalised kink position t0 = -center/half_width — never on
    |a_c|. That is what makes the budget gauge-invariant: ReLU is positively
    homogeneous, so an absolute error bound would measure the gauge rather than
    the approximation difficulty. s_c then restores the physical weighting, and
    s_c * eps_rel is the downstream-weighted absolute error (up to the constant
    sigma).

APPROXIMATION NOTES (both deliberate, both stated rather than hidden)
    1. eps_rel uses Chebyshev INTERPOLATION at d+1 nodes, not a Remez minimax
       fit. Interpolation is near-minimax within a factor of O(log d) and is
       what an implementation would actually evaluate; a minimax fit would
       lower every eps_rel slightly and uniformly, shifting the B that hits a
       given degree distribution but not the dispersion between channels.
    2. ||V[:,c]||_2 takes V to be the next weight-bearing layer in module
       order. For residual blocks this ignores the shortcut path, which also
       consumes the channel. This is the proxy RQ2 asks about, and
       proxy_validation.py tests it rather than assuming it.

INPUTS
    Model weights via torch.hub / torchvision (cached; hashes in
    results/phase0/checkpoint_provenance.json for resnet20).

OUTPUTS
    results/phase3/invariants_<net>.csv — one row per (site, channel):
        net, site, site_index, channel_idx, gamma, beta, running_mean,
        running_var, eps, a_c, b_c, s_c, k_c, v_norm, lo, hi, t0,
        d_A, d_B   (required degree under each delta model's plateau map)
"""

from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.provenance import require_clean, write_csv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO.parent / "results"
SIGMA = 3.0            # interval half-width in units of |a_c|; archive default
MAX_DEGREE = 63        # ceiling on the degree search


# ── the approximation error, normalised ────────────────────────────────────
@lru_cache(maxsize=None)
def eps_rel(degree: int, t0_quantised: int) -> float:
    """sup|g - p_d| on [-1,1] for g(t) = max(0, t - t0), t0 on a 1e-3 grid.

    Quantising t0 makes the cache effective: adjacent channels differ in the
    fourth decimal and the error surface is smooth, so the loss is far below
    the modelling error already accepted above.
    """
    t0 = t0_quantised / 1000.0
    if t0 <= -1.0:          # kink left of the interval: g is linear, exact
        return 0.0
    if t0 >= 1.0:           # kink right of it: g is identically zero, exact
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


# ── model plumbing ─────────────────────────────────────────────────────────
def load_model(net: str) -> nn.Module:
    if net == "resnet20":
        m = torch.hub.load("chenyaofo/pytorch-cifar-models",
                           "cifar10_resnet20", pretrained=True,
                           trust_repo=True)
    elif net == "mobilenetv2":
        import torchvision
        m = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"unknown net {net!r}")
    m.eval()
    return m


def _consumer_norms(consumer: nn.Module, n_channels: int) -> np.ndarray:
    """||V[:,c]||_2 for each channel c consumed by `consumer`."""
    W = consumer.weight.detach()
    if isinstance(consumer, nn.Linear):                 # (out, in)
        return W.norm(dim=0).numpy()
    groups = getattr(consumer, "groups", 1)
    if groups == consumer.in_channels and groups > 1:   # depthwise: (C,1,k,k)
        return W.reshape(W.shape[0], -1).norm(dim=1).numpy()
    if groups > 1:                                      # grouped
        per = consumer.in_channels // groups
        out = np.zeros(n_channels)
        for c in range(min(n_channels, consumer.in_channels)):
            out[c] = W[:, c % per].norm().item()
        return out
    return W.transpose(0, 1).reshape(W.shape[1], -1).norm(dim=1).numpy()


def _in_channels(mod: nn.Module) -> int:
    return getattr(mod, "in_channels", getattr(mod, "in_features", -1))


def bn_sites(model: nn.Module) -> list[tuple[str, nn.BatchNorm2d, nn.Module]]:
    """(name, bn, consuming weight-bearing module) in module order.

    The consumer is the next Conv2d/Linear that ACCEPTS the BN's channel
    count. Taking simply "the next one in module order" is wrong at a
    downsampling block: PyTorch lists the shortcut convolution immediately
    after `bn2`, but it consumes the block's INPUT (16 channels), not `bn2`'s
    output (32) — the real consumer is the next block's `conv1`. Matching on
    channel count repairs that and, on the 21 ResNet-20 sites, changes the
    consumer for exactly the two downsampling `bn2`s.

    Still approximate, deliberately: the output also flows through the residual
    add into the shortcut path, which this ignores. That is the proxy RQ2 asks
    about, and proxy_validation.py measures it rather than assuming it.
    """
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


def invariants(net: str, budget: float, sigma: float = SIGMA) -> pd.DataFrame:
    model = load_model(net)
    rows = []
    for site_index, (name, bn, consumer) in enumerate(bn_sites(model)):
        gamma = bn.weight.detach()
        beta = bn.bias.detach()
        mean = bn.running_mean.detach()
        var = bn.running_var.detach()
        vnorms = _consumer_norms(consumer, bn.num_features)
        for c in range(bn.num_features):
            denom = math.sqrt(var[c].item() + bn.eps)
            a_c = gamma[c].item() / denom
            b_c = beta[c].item() - gamma[c].item() * mean[c].item() / denom
            half = sigma * abs(a_c)
            if half <= 0:                # dead channel: constant output
                lo = hi = b_c
                t0 = math.inf
            else:
                lo, hi = b_c - half, b_c + half
                t0 = -b_c / half         # kink position in normalised coords
            v = float(vnorms[c]) if c < len(vnorms) else float("nan")
            s_c = abs(a_c) * v
            t0c = max(-1.5, min(1.5, t0))
            rows.append({
                "net": net, "site": name, "site_index": site_index,
                "channel_idx": c,
                "gamma": gamma[c].item(), "beta": beta[c].item(),
                "running_mean": mean[c].item(), "running_var": var[c].item(),
                "eps": bn.eps, "a_c": a_c, "b_c": b_c,
                "v_norm": v, "s_c": s_c,
                "k_c": (b_c / a_c) if a_c != 0 else math.nan,
                "lo": lo, "hi": hi, "t0": t0c,
                "degree": required_degree(s_c, t0c, budget),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="resnet20",
                    choices=["resnet20", "mobilenetv2"])
    ap.add_argument("--budget", type=float, default=0.01,
                    help="network-wide B in s_c * eps_rel(d) <= B")
    ap.add_argument("--sigma", type=float, default=SIGMA)
    args = ap.parse_args()

    require_clean()
    df = invariants(args.net, args.budget, args.sigma)
    out = RESULTS / "sensitivity_proxy" / f"invariants_{args.net}.csv"
    write_csv(df, out, extra={"budget_B": args.budget, "sigma": args.sigma,
                              "eps_rel": "chebyshev interpolation, d+1 nodes"})
    print(f"{args.net}: {len(df)} channel-sites across "
          f"{df.site.nunique()} sites -> {out}")
    print(df.groupby("site_index").degree.agg(["min", "median", "max"])
          .head(12).to_string())


if __name__ == "__main__":
    main()
