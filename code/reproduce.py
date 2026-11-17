"""One command per main result, and a check of the saved results.

    python reproduce.py list              what can be reproduced, and how long it takes
    python reproduce.py check             compare the saved result files with the expected numbers
    python reproduce.py check <name>      the same for one result
    python reproduce.py run <name>        rerun one result (overwrites its files in ../results)
    python reproduce.py show <name>       print the commands without running them

Run from the code/ folder. Environment: see environment.yml (Python 3.12,
PyTorch 2.4.1 CPU) and scripts/openfhe_docker/Dockerfile (OpenFHE 1.6.1 in a
Linux container, needed for the entries marked "docker").

`check` reruns nothing. It reads the result files that are in the repository
and compares the numbers the thesis quotes with the values written here. If
you rerun an experiment, run `check` again: accuracy numbers are
deterministic on one machine, timings are not and are checked loosely.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
RES = PROJECT / "results"
PY = sys.executable


def read(rel):
    return pd.read_csv(RES / rel, comment="#")


def docker(module_and_args: str) -> str:
    return (f'docker run --rm -v "{PROJECT.as_posix()}:/work" -w /work/code thesis-openfhe '
            f"python -m {module_and_args}")


def near(got, want, tol, what):
    ok = abs(float(got) - want) <= tol
    return ok, f"{what}: {float(got):.4g} (expected {want:g} +/- {tol:g})"


# ------------------------------------------------------------------ checks
def chk_window():
    g = read("ciphertexts_per_layer/G_table.csv")
    r = g[(g.network == "resnet20") & (g.N == 65536) & (g.packing == "multiplexed") & (g.dup == 1)]
    return [(bool((r.G == 1).all()), f"ResNet-20 at N = 2^16: G = {sorted(r.G.unique())} (expected [1])")]


def chk_realign():
    out = []
    for f in ("VERDICT_exp1_realignment.txt", "VERDICT_exp1_realignment_N16.txt"):
        t = (RES / "level_realignment" / f).read_text(encoding="utf8")
        out.append(("CONFIRMED" in t, f"{f}: {t.strip()[:70]}"))
    return out


def chk_search():
    p = read("level_saving_search/level_objective_probe.csv")
    return [(len(p) == 1488, f"instances: {len(p)} (expected 1488)")]


def chk_rotation():
    r = read("spread_and_rotation/exp2_rotation.csv")
    rate = 100 * r.groupby(["net", "G", "delta_model"]).rotates.mean()
    return [near(rate.max(), 31.25, 0.01, "largest rotation rate, percent"),
            near(rate.min(), 1.25, 0.01, "smallest rotation rate, percent")]


def chk_any_grouping():
    s = read("spread_and_rotation/ext_any_grouping_summary.csv")
    return [near(s.any_grouping_can_save_pct.max(), 16.25, 0.01,
                 "largest share of cases where some grouping can save a level, percent"),
            near(s.sorted_saves_a_level_pct.max(), 11.25, 0.01,
                 "largest share where the sorted grouping saves a level, percent")]


def chk_dead():
    j = json.loads((RES / "dead_channels" / "exp2c_mobilenet_data.json").read_text(encoding="utf8"))
    t = json.dumps(j)
    return [("7095" in t, "7,095 constant channels appear in the data file"),
            ("94.05" in t, "unmodified accuracy 94.05% appears in the data file")]


def chk_composite():
    c = read("activation_accuracy/oa_composite.csv")
    s = c[c.sequence == "7-15-15"].iloc[0]
    return [(int(c.within_1_point.sum()) == 30, f"sequences within 1 point: {int(c.within_1_point.sum())} (expected 30)"),
            near(s.top1, 91.94, 0.005, "7-15-15 top-1")]


def chk_range():
    g = read("range_vs_degree/exp3_range_grid_train.csv")
    a = g[(g.target == "cifar10_resnet20") & (g.rule == "minmax") & (g.degree == 15)].top1.iloc[0]
    b = g[(g.target == "cifar10_resnet20") & (g.rule == "sigma3") & (g.degree == 31)].top1.iloc[0]
    return [near(a, 90.29, 0.2, "ResNet-20, observed range, degree 15"),
            near(b, 10.00, 0.2, "ResNet-20, three standard deviations, degree 31")]


def chk_time():
    t = read("grouping_time/repeat_runs/summary_time_saving.csv").set_index("budget_B")
    d = read("grouping_time/repeat_runs/summary_per_degree.csv").set_index("degree")
    return [near(t.loc[0.01, "mean"], 14.7, 3.0, "activation time saved at B = 0.01, percent (timing: loose)"),
            (bool(d.levels_same_in_all_runs.all()), "levels identical in all five runs at all 63 degrees"),
            near(d.loc[15, "seconds_mean"], 0.668, 0.2, "seconds for one degree-15 activation (timing: loose)")]


def chk_proxy():
    p = read("grouping_accuracy_proxy/grouping_accuracy_proxy_cap63_first2000.csv")
    b = p[p.budget_B.notna() & (p.interval == "observed")]
    f = read("grouping_accuracy_proxy/grouping_accuracy_proxy_cap31_floor15_margin0.25.csv")
    lw = f[(f.budget_B == 0.01) & (f.degrees == "layerwise")].top1.iloc[0]
    lim = read("grouping_time/repeat_runs/summary_time_saving_degree_limits.csv")
    s = lim[(lim.degree_floor == 15) & (lim.degree_cap == 31) & (lim.budget_B == 0.01)].saving_pct_mean.iloc[0]
    return [(b.top1.max() < 12, f"thesis degrees: best accuracy {b.top1.max():.2f}% (expected chance, below 12)"),
            near(lw, 92.32, 0.2, "degrees 15 to 31, layerwise, B = 0.01"),
            near(s, 0.6, 3.0, "time saved with degrees 15 to 31 at B = 0.01, percent (timing: loose)")]


def chk_stats():
    r = read("statistics/rotation_intervals.csv")
    p = read("statistics/proxy_intervals.csv").iloc[0]
    return [(r.p_value_rate_below_reference.max() < 1e-6, "every rotation rate is below its unrelated reference (p < 1e-6)"),
            near(p.mean_rank_correlation, 0.20, 0.01, "proxy: mean within-site rank correlation")]


def chk_cost():
    lv = read("cost_model/levels_model.csv")
    return [(bool((lv.model == lv.measured).all()), "closed form for the levels matches all 63 measured degrees")]


def chk_rule():
    t = read("degree_rule_fix/test_saving_best_layerwise.csv")
    return [(len(t) >= 1, f"{len(t)} accuracy bars reported"),
            (bool((t.grouped_test_top1 > 90).all()), "the chosen rules keep the network above 90% on the test set")]


def chk_margin():
    a = read("range_vs_degree/ext_margin_sweep.csv")
    return [(len(a) >= 15, f"{len(a)} margin and degree settings scored")]


def chk_e2e():
    e = read("end_to_end_mnist/encrypted_run.csv")
    a = e[e.config.str.startswith("A")].iloc[0]
    return [near(a.encrypted_top1, a.plain_polynomial_top1, 0.05,
                 "encrypted accuracy equals the plain polynomial network (config A)"),
            (int(e.images.iloc[0]) == 10000, "all 10,000 test images")]


M = "experiments"
ENTRIES = {
    "window": dict(what="Nothing to group on small networks (thesis 6.2)", time="1 s", docker=False,
                   cmds=[f"{PY} -m {M}.ciphertexts_per_layer.ciphertexts_per_layer"], check=chk_window),
    "realignment": dict(what="Addition brings ciphertexts to the same level (6.3)", time="1 min", docker=True,
                        cmds=[docker(f"{M}.level_realignment.level_realignment"),
                              docker(f"{M}.level_realignment.level_realignment --logN 16 --depth 30")],
                        check=chk_realign),
    "search": dict(what="Exhaustive search over groupings of small instances (6.4)", time="2 min", docker=False,
                   cmds=[f"{PY} -m {M}.level_saving_search.level_saving_search"], check=chk_search),
    "rotation": dict(what="Spread and rotation on six networks (6.5, 6.6)", time="20 s", docker=False,
                     cmds=[f"{PY} -m {M}.spread_and_rotation.spread_and_rotation_six_networks"],
                     check=chk_rotation),
    "any_grouping": dict(what="Extension: can any grouping save a level?", time="30 s", docker=False,
                         cmds=[f"{PY} -m {M}.spread_and_rotation.any_grouping_bound"], check=chk_any_grouping),
    "dead_channels": dict(what="Dead channels of the CIFAR-10 MobileNetV2 (6.7)", time="13 min", docker=False,
                          cmds=[f"{PY} -m {M}.dead_channels.dead_channels_on_images --data-root data"],
                          check=chk_dead),
    "composite": dict(what="Depth of an accurate activation (6.10)", time="39 min", docker=False,
                      cmds=[f"{PY} -m {M}.activation_accuracy.composite_sequences"], check=chk_composite),
    "interval": dict(what="Interval against degree (6.11)", time="3 h on CPU", docker=False,
                     cmds=[f"{PY} -m {M}.range_vs_degree.range_vs_degree --calib-split train --data-root data"],
                     check=chk_range),
    "margin": dict(what="Extension: margin, per-layer interval, calibration sample, paired tests",
                   time="1 h", docker=False,
                   cmds=[f"{PY} -m {M}.range_vs_degree.margin_and_layer_interval --data-root data"],
                   check=chk_margin),
    "time": dict(what="Time saved by grouping, five timing runs (6.14)", time="45 min", docker=True,
                 cmds=["cd .. && bash code/scripts/repeat_timing_runs.sh 5",
                       f"cd .. && {PY} code/scripts/summarize_timing_runs.py"], check=chk_time),
    "accuracy_check": dict(what="Accuracy behind the time saving, CIFAR-10 MobileNetV2 (6.14.3)",
                           time="15 min + 70 min", docker=False,
                           cmds=[f"{PY} -m {M}.grouping_accuracy_proxy.grouping_accuracy_proxy --data-root data --images 2000",
                                 f"{PY} -m {M}.grouping_accuracy_proxy.grouping_accuracy_proxy --data-root data "
                                 "--cap 31 --floor 15 --margin 0.25 --budgets 0.1 0.03 0.01",
                                 f"cd .. && {PY} code/scripts/time_saving_with_degree_limits.py"],
                           check=chk_proxy),
    "degree_rule": dict(what="Extension: a degree rule that works, and what grouping saves under it",
                        time="1.5 h + 2 h", docker=False,
                        cmds=[f"{PY} -m {M}.degree_rule_fix.degree_rule_fix sweep --data-root data",
                              f"{PY} -m {M}.degree_rule_fix.degree_rule_fix test --data-root data"],
                        check=chk_rule),
    "end_to_end": dict(what="Extension: MNIST end to end under CKKS, decrypted accuracy",
                       time="1 min + 1 h", docker=True,
                       cmds=[f"{PY} -m {M}.end_to_end_mnist.train_and_plan --data-root data",
                             docker(f"{M}.end_to_end_mnist.encrypted_run")], check=chk_e2e),
    "statistics": dict(what="Extension: confidence intervals and tests from saved results", time="1 min",
                       docker=False, cmds=[f"cd .. && {PY} code/scripts/statistics_existing_results.py"],
                       check=chk_stats),
    "cost_model": dict(what="Extension: cost model checked against the measurements", time="5 s",
                       docker=False, cmds=[f"cd .. && {PY} code/scripts/cost_model.py"], check=chk_cost),
    "tests": dict(what="The automated tests", time="20 s", docker=False,
                  cmds=[f"{PY} -m pytest tests -q"], check=None),
    "figures": dict(what="The four figures of the thesis", time="10 s", docker=False,
                    cmds=[f"{PY} scripts/make_figures.py"], check=None),
}


def do_check(names) -> int:
    bad = 0
    for n in names:
        e = ENTRIES[n]
        if e["check"] is None:
            continue
        try:
            res = e["check"]()
        except FileNotFoundError as err:
            print(f"[MISSING] {n}: {Path(err.filename).name} is not there; run `reproduce.py run {n}`")
            bad += 1
            continue
        for ok, msg in res:
            print(f"[{'ok' if ok else 'DIFFERENT'}] {n}: {msg}")
            bad += not ok
    print("all saved results match" if not bad else f"{bad} item(s) need a look")
    return bad


def main() -> int:
    a = sys.argv[1:]
    if not a or a[0] == "list":
        for n, e in ENTRIES.items():
            print(f"{n:<15} {e['time']:<16} {'docker' if e['docker'] else '      '}  {e['what']}")
        return 0
    if a[0] == "check":
        return do_check(a[1:] or list(ENTRIES))
    if a[0] in ("run", "show") and len(a) == 2 and a[1] in ENTRIES:
        for c in ENTRIES[a[1]]["cmds"]:
            print("$", c, flush=True)
            if a[0] == "run":
                r = subprocess.run(c, shell=True, cwd=CODE)
                if r.returncode:
                    return r.returncode
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
