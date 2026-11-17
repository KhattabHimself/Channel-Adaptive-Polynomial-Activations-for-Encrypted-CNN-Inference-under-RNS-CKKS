# grouping time

**Question:** Does grouping save real time in OpenFHE, and how much?

Three steps. `degrees` works out which degree every group runs at on MobileNetV2. `time` measures one activation on a full ciphertext at every degree from 1 to 63 in OpenFHE (ring size 2^16, 128-bit security), then runs every group directly at two budgets. `combine` adds it up. Only the activations are timed. Convolutions and bootstraps are not.

## Run

From the `code/` folder:

    python -m experiments.grouping_time.grouping_time degrees
    docker run --rm -v "<project folder>:/work" -w /work/code thesis-openfhe python -m experiments.grouping_time.grouping_time time --reps 5 --depths 12 24
    python -m experiments.grouping_time.grouping_time combine
    docker run --rm -v "<project folder>:/work" -w /work/code thesis-openfhe python -c "import experiments.grouping_time.grouping_time as e; print(e.time_bootstrap(after=2))"

Time: 20 seconds; 6 minutes in Docker with `--reps 3 --depths 12`; the full command was not timed again; 1 second; 20 seconds.

## Output

Written to `results/grouping_time/`:

- `exp4_bucket_degrees.csv`: degree of every group
- `exp4_degree_timing.csv`: seconds and levels per degree
- `exp4_direct_regions.csv`: the direct check
- `exp4_time_gain.csv`: **the table in the thesis**
- `exp4_bootstrap_after2.json`: one bootstrap, 11.3 s
- `exp4_bootstrap_after10.json`: ran out of memory, not used
- `check_rerun_2026-11-16/`: a second timing run, made during verification

## Where it is used

Sections 6.1 and 6.14, Appendix B (called exp4).

## Checked?

The timing was repeated five times on 16 Nov 2026 (`bash code/scripts/repeat_timing_runs.sh 5` from the project folder, then `python code/scripts/summarize_timing_runs.py`). Results are in `repeat_runs/`: every run, plus `summary_time_saving.csv` (mean and standard deviation per budget), `summary_per_degree.csv` and `summary_direct_check.csv`. **The thesis quotes these means**: 14.7 +/- 0.9 % of activation time saved at the tightest budget, 12.6 +/- 1.0 % with the exact interval. Levels were identical in every run. The single run of 1 Nov (`exp4_*.csv` in this folder) gave 13.5 % and is kept. Activations only: no convolution or bootstrap is timed. No accuracy was measured on this network at any budget, so which budget is usable is not known. The bootstrap uses different settings and no security level; it is an order of magnitude only. See VERIFICATION_REPORT.md, problems 3, 4 and 6.

Old name: `exp4_time_gain.py`.
