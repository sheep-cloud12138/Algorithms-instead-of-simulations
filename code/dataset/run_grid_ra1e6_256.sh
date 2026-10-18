#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
OUT="$HOME/nc_run/grid_ra1e6_256"; mkdir -p "$OUT"
run_one() {
  local tag="$1" seed="$2" eps="$3"
  "$PY" genData_stochastic_fourierBC.py --out_root "$OUT/$tag" --nsamples 1 --start_id 0 \
    --seed0 "$seed" --res 256 --length 1.0 --T_init 1.0 --dT "$eps" \
    --Ra_min 1e6 --Ra_max 1e6 --Ma_expo 0.16666666666666666 \
    --numT_left_min 1 --numT_left_max 1 --numT_right_min 1 --numT_right_max 1 \
    --numT_top_min 0 --numT_top_max 0 --numT_bot_min 0 --numT_bot_max 0 \
    --fr_mode derived --buoyancy shifted --save_plot 0 --max_tries 1 --skip_existing 1 \
    > "$OUT/$tag.log" 2>&1
  echo "[$tag] rc=$? $(date +%T)" >> "$OUT/progress.txt"
}
for e in 1.2 1.9; do
  run_one "c16_Ra1e6_1100_e$e" 20265260 "$e" &
done
wait
echo "GRID1E6_256 DONE $(date +'%F %T')" >> "$OUT/progress.txt"
