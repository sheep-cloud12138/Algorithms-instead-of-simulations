#!/usr/bin/env bash
# Pr = 4.38 (water at 20 C) family: bit-identical geometry/BC/Ra draws to the 104 air
# samples of the main batch (same --seed0 20261016, ids 0..103), ONLY --Pr differs.
set -uo pipefail
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
B="$HOME/nc_run/water"; mkdir -p "$B/logs"
echo "WATER (Pr=4.38) START $(date +%T)"
NS=26; SHARDS=4
for s in $(seq 0 $((SHARDS-1))); do
  start=$((s*NS))
  ( "$PY" genData_stochastic_fourierBC.py \
      --out_root "$B/dataset" --nsamples $NS --start_id $start --seed0 20261016 \
      --res 64 --length 1.0 --T_init 1.0 \
      --dT_min 0.2 --dT_max 1.9 \
      --Ra_min 1e2 --Ra_max 1e6 --Ma_expo 0.16666666666666666 \
      --numT_left_min 1 --numT_left_max 4 \
      --numT_right_min 1 --numT_right_max 4 \
      --numT_top_min 0 --numT_top_max 2 \
      --numT_bot_min 0 --numT_bot_max 2 \
      --Pr 4.38 \
      --fr_mode derived --buoyancy shifted --save_plot 0 \
      --max_tries 2 --skip_existing 1 \
      > "$B/logs/shard_${s}.log" 2>&1
    echo "[water shard $s rc=$?] $(date +%T)" >> "$B/logs/done" ) &
done
wait
echo "WATER DONE $(date +%T)  samples=$(ls $B/dataset | wc -l)"