"""
PURPOSE
    Phase 3 / Gate B, RQ2: does the cheap sensitivity proxy actually predict
    downstream influence, or does the budget presuppose its own answer?

    The budget weights each channel's approximation error by
    s_c = |a_c| * ||V[:,c]||_2. The |a_c| factor is exact — it is the interval
    half-width, and it is where the error physically enters. The
    ||V[:,c]||_2 factor is a PROXY for "how much the rest of the network cares
    about a perturbation of channel c", and it is the only approximate part.
    LAB_NOTEBOOK Sec. 1.5 flags that adopting it presupposes RQ2. This measures
    it instead.

METHOD
    A FIXED ABSOLUTE perturbation is added to one channel's post-BN activation
    and the resulting change in logits is measured over a batch:

        influence_c = mean_over_images || f(x ; +delta on channel c) - f(x) ||_2

    Fixed and absolute matters. Perturbing by delta proportional to |a_c| would
    make influence proportional to s_c by construction and the test would
    confirm itself. With a constant delta the measurement isolates exactly the
    quantity ||V[:,c]||_2 claims to stand for.

    Reported: Spearman rank correlation of influence against ||V[:,c]||_2, and
    for contrast against |a_c| and s_c. Rank correlation, not Pearson, because
    the budget only uses the proxy to ORDER and threshold channels.

    Correlations are computed as Pearson-on-ranks (identical to Spearman) by
    elementwise arithmetic. scipy, pandas' method="spearman", and numpy's
    corrcoef all abort in a torch process on this machine: numpy is MKL-backed
    and torch bundles its own OpenMP runtime, so the second initialisation
    kills the process. Significance is reported as a large-sample normal
    approximation z = rho * sqrt(n-1) (n = 784), not an exact t-test.

INPUTS
    data/cifar-10-batches-py (staged from the archive in Phase 0),
    results/phase3/invariants_resnet20.csv (for v_norm, a_c, s_c).

OUTPUTS
    results/phase3/proxy_validation_<net>.csv — per channel-site:
        site, channel_idx, v_norm, a_c, s_c, influence
    Correlations printed and stamped into the CSV header.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.provenance import require_clean, write_csv  # noqa: E402
from src.channel_degrees import bn_sites, load_model  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "results"


def _pearson(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation by elementwise arithmetic.

    numpy.corrcoef routes through MKL's BLAS, whose OpenMP runtime collides
    with the one torch bundles; calling it in a process that has imported
    torch aborts with "libiomp5md.dll already initialized". Sums and means do
    not take that path.
    """
    x = x.astype(float)
    y = y.astype(float)
    xm, ym = x - x.mean(), y - y.mean()
    denom = ((xm ** 2).sum() ** 0.5) * ((ym ** 2).sum() ** 0.5)
    return float((xm * ym).sum() / denom) if denom else float("nan")
# Legacy CIFAR normalisation — the one that reproduced the documented 92.60%
# top-1 exactly [results/phase0/checkpoint_provenance.json].
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def cifar_batch(n: int) -> torch.Tensor:
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = torchvision.datasets.CIFAR10(root=REPO / "data", train=False,
                                      download=False, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=False)
    x, _ = next(iter(loader))
    return x


@torch.no_grad()
def measure(net: str, n_images: int, delta: float) -> pd.DataFrame:
    model = load_model(net)
    x = cifar_batch(n_images)
    base = model(x)

    rows = []
    for name, bn, _consumer in bn_sites(model):
        state = {"channel": -1}

        def hook(_m, _inp, out, state=state):
            if state["channel"] >= 0:
                out = out.clone()
                out[:, state["channel"]] += delta
            return out

        handle = bn.register_forward_hook(hook)
        try:
            for c in range(bn.num_features):
                state["channel"] = c
                perturbed = model(x)
                infl = (perturbed - base).norm(dim=1).mean().item()
                rows.append({"site": name, "channel_idx": c,
                             "influence": infl})
        finally:
            handle.remove()
        print(f"  {name}: {bn.num_features} channels", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="resnet20", choices=["resnet20"])
    ap.add_argument("--images", type=int, default=128)
    ap.add_argument("--delta", type=float, default=0.1)
    args = ap.parse_args()
    require_clean()

    # Read the invariants rather than recomputing them: v_norm, a_c and s_c
    # do not depend on the budget (only `degree` does), and loading the model
    # a second time in one process re-initialises OpenMP and aborts on this
    # machine (torch and its hub import path link separate runtimes).
    inv_path = RESULTS / f"sensitivity_proxy/invariants_{args.net}.csv"
    if not inv_path.exists():
        raise SystemExit(f"missing {inv_path} — run invariants.py first")
    infl = measure(args.net, args.images, args.delta)
    inv = pd.read_csv(inv_path, comment="#")[
        ["site", "channel_idx", "v_norm", "a_c", "s_c"]]
    df = inv.merge(infl, on=["site", "channel_idx"], how="inner")
    df["abs_a_c"] = df.a_c.abs()

    stats = {}
    n = len(df)
    for col in ("v_norm", "abs_a_c", "s_c"):
        # Pearson on ranks IS Spearman.
        rho = _pearson(df[col].rank(), df.influence.rank())
        stats[f"spearman_{col}"] = round(rho, 4)
        stats[f"z_{col}"] = round(rho * (n - 1) ** 0.5, 2)

    out = RESULTS / f"sensitivity_proxy/proxy_validation_{args.net}.csv"
    write_csv(df, out, extra={"n_images": args.images, "delta": args.delta,
                              **stats})

    print(f"\n=== RQ2 proxy validation, {args.net} "
          f"({args.images} images, delta={args.delta}) ===")
    print(f"n = {len(df)} channel-sites")
    for col in ("v_norm", "abs_a_c", "s_c"):
        print(f"  Spearman(influence, {col:8s}) = "
              f"{stats['spearman_' + col]:+.4f}  "
              f"(z ~ {stats['z_' + col]:+.1f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
