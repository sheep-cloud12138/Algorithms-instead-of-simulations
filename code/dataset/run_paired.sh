#!/usr/bin/env bash
# P2-paired: for each (Ra, BC) configuration, run the SAME sample at 5 values of eps.
# seed0 fixes the whole rng stream (numT and Ra draws + wall draws), and dT is
# passed as a plain CLI arg (no --dT_min draw), so the boundary conditions in
# theta = (T-T0)/dT are BIT-IDENTICAL across the 5 eps values.
set -uo pipefail
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
OUT="$HOME/nc_run/paired/dataset"; mkdir -p "$OUT"

run_one() { # tag seed0 Ra nL nR nT nB eps
  local tag="$1" seed="$2" Ra="$3" nL="$4" nR="$5" nT="$6" nB="$7" eps="$8"
  local dir="$OUT/$tag"
  "$PY" genData_stochastic_fourierBC.py --out_root "$dir" --nsamples 1 --start_id 0 \
    --seed0 "$seed" --res 64 --length 1.0 --T_init 1.0 --dT "$eps" \
    --Ra_min "$Ra" --Ra_max "$Ra" --Ma_expo 0.16666666666666666 \
    --numT_left_min "$nL" --numT_left_max "$nL" \
    --numT_right_min "$nR" --numT_right_max "$nR" \
    --numT_top_min "$nT" --numT_top_max "$nT" \
    --numT_bot_min "$nB" --numT_bot_max "$nB" \
    --fr_mode derived --buoyancy shifted --save_plot 0 --max_tries 1 --skip_existing 1 \
    > "$OUT/$tag.log" 2>&1
  echo "[$tag] rc=$? $(date +%T)" >> "$HOME/nc_run/paired/progress.txt"
}

EPS_LIST="0.4 0.8 1.2 1.6 1.9"
CFG=0
for spec in "1e2 1 1 0 0" "1e2 2 3 0 0" "1e2 4 2 1 1" "1e2 3 4 2 0" \
            "1e3 1 1 0 0" "1e3 2 3 0 0" "1e3 4 2 1 1" "1e3 3 4 2 0" \
            "1e4 1 1 0 0" "1e4 2 3 0 0" "1e4 4 2 1 1" "1e4 3 4 2 0" \
            "1e5 1 1 0 0" "1e5 2 3 0 0" "1e5 4 2 1 1" "1e5 3 4 2 0" \
            "1e6 1 1 0 0" "1e6 2 3 0 0" "1e6 4 2 1 1" "1e6 3 4 2 0" ; do
  set -- $spec
  Ra=$1; nL=$2; nR=$3; nT=$4; nB=$5
  seed=$((777000 + 13*CFG))
  for e in $EPS_LIST; do
    tag=$(printf "c%02d_Ra%s_%s%s%s%s_e%s" $CFG "$Ra" "$nL" "$nR" "$nT" "$nB" "$e")
    run_one "$tag" "$seed" "$Ra" "$nL" "$nR" "$nT" "$nB" "$e" &
    while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done
  done
  CFG=$((CFG+1))
done
wait
echo "PAIRED COMPLETE $(date +%F' '%T)" >> "$HOME/nc_run/paired/progress.txt"
