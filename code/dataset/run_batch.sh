#!/usr/bin/env bash
# 600-sample 2D paired dataset.
#   fixed : T_init = 1.0, Pr = 0.71, gamma = 1.4, res = 64, L = 1
#   varies: Ra ~ logU[1e2, 1e6], eps = dT ~ U[0.2, 1.9], numT(L,R,T,B)
#   Fr = sqrt(dT/T_init)  (--fr_mode derived, same default as genData_3d.py)
#   Ma = 8.27e-4 * (Ra/1e4)^(1/6) (the relation stated in the source paper)
#   buoyancy shifted, stopping steady, newton seed w0, scale_aware tolerances
set -uo pipefail
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
B="$HOME/nc_run/batch"
NS=75            # 8 shards x 75 = 600
SHARDS=8
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
      --fr_mode derived --buoyancy shifted --save_plot 0 \
      --max_tries 2 --skip_existing 1 \
      > "$B/logs/shard_${s}.log" 2>&1
    echo "[shard $s done rc=$?] $(date +%T)" >> "$B/logs/shards.done" ) &
done
wait
echo "BATCH COMPLETE $(date +%F' '%T)" >> "$B/logs/shards.done"
