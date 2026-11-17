#!/bin/bash
# Repeat the OpenFHE activation timing several times and keep every run.
#
#   bash code/scripts/repeat_timing_runs.sh 5        (from the project folder)
#
# Each run is the "time" step of the grouping time experiment (5 repetitions
# per degree, 12-level chain, plus the direct check). The step overwrites
# results/grouping_time/exp4_degree_timing.csv and exp4_direct_regions.csv, so
# this script copies both into results/grouping_time/repeat_runs/run_K/ after
# each run and puts the original files back at the end with git.
# Summarize with:  python code/scripts/summarize_timing_runs.py
set -e
N="${1:-5}"
cd "$(dirname "$0")/../.."
ROOT="$(pwd -W 2>/dev/null || pwd)"
OUT=results/grouping_time/repeat_runs
mkdir -p "$OUT"
for k in $(seq 1 "$N"); do
  echo "=== run $k of $N, $(date +%H:%M:%S)"
  MSYS_NO_PATHCONV=1 docker run --rm -v "$ROOT:/work" -w /work/code thesis-openfhe \
    python -m experiments.grouping_time.grouping_time time --reps 5 --depths 12 \
    > "$OUT/run_${k}.log" 2>&1
  mkdir -p "$OUT/run_$k"
  cp results/grouping_time/exp4_degree_timing.csv results/grouping_time/exp4_direct_regions.csv "$OUT/run_$k/"
  mv "$OUT/run_${k}.log" "$OUT/run_$k/timing_run.log"
done
git checkout -- results/grouping_time/exp4_degree_timing.csv results/grouping_time/exp4_direct_regions.csv
echo "=== done $(date +%H:%M:%S); original timing files restored"
