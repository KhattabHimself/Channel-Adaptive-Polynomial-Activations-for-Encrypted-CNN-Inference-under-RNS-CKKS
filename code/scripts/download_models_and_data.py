"""
Download everything the experiments need, on the LOGIN node.

Compute nodes on DRAC have no internet, so every checkpoint and dataset has to
be in the cache before a job starts. This downloads them, then reloads every
model in offline mode to prove the jobs will find them.

run (setup_login_node.sh does this for you):
    python prefetch.py --data-root $DATA
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.network_tools import load_model  # noqa: E402
from experiments.spread_and_rotation.spread_and_rotation_six_networks import NETWORKS  # noqa: E402
from experiments.range_vs_degree.range_vs_degree import TARGETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    import torchvision

    names = list(dict.fromkeys(list(NETWORKS) + list(TARGETS)))
    failed = []

    print(f"TORCH_HOME = {os.environ.get('TORCH_HOME', '(unset)')}")
    print(f"\n--- models ({len(names)}) ---")
    for n in names:
        try:
            load_model(n)
            print(f"  ok      {n}")
        except Exception as e:  # noqa: BLE001  keep going, report at the end
            print(f"  FAILED  {n}: {e}")
            failed.append(n)

    print("\n--- datasets ---")
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    for cls in (torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100):
        for train in (False, True):
            split = "train" if train else "test"
            try:
                cls(root=root, train=train, download=True)
                print(f"  ok      {cls.__name__} {split}")
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED  {cls.__name__} {split}: {e}")
                failed.append(f"{cls.__name__}-{split}")

    # what the jobs will do: load with no network
    print("\n--- offline reload check ---")
    os.environ["THESIS_OFFLINE"] = "1"
    for n in names:
        if n in failed:
            continue
        try:
            load_model(n)
            print(f"  ok      {n}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED  {n} offline: {e}")
            failed.append(f"{n}-offline")

    if failed:
        print(f"\nPREFETCH INCOMPLETE, {len(failed)} problem(s): {failed}")
        print("Jobs that need these will fail on the compute node.")
        raise SystemExit(1)
    print("\nPREFETCH OK, everything is cached for offline jobs.")


if __name__ == "__main__":
    main()
