#!/bin/bash
set -uo pipefail
mkdir -p /home/wjh/nc_run/cavity2/logs /home/wjh/nc_run/tight/logs
exec > /home/wjh/nc_run/cpu_queue.log 2>&1
echo "CPU QUEUE START $(date +%T)"
while pgrep -f "run_cavity.sh" > /dev/null; do sleep 20; done
echo "--- 方腔 104 组结束 $(date +%T) ---"
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"

# ================= 1) 宽高比 2:1 的随机壁面方腔，104 组 =================
B="$HOME/nc_run/cavity2"; NS=13
for s in $(seq 0 7); do
  start=$((s*NS))
  ( "$PY" genData_stochastic_fourierBC.py \
      --out_root "$B/dataset" --nsamples $NS --start_id $start --seed0 20261018 \
      --res 64 --length 1.0 --aspect 2.0 --T_init 1.0 \
      --dT_min 0.2 --dT_max 1.9 \
      --Ra_min 1e2 --Ra_max 1e6 --Ma_expo 0.16666666666666666 \
      --numT_left_min 1 --numT_left_max 4 \
      --numT_right_min 1 --numT_right_max 4 \
      --numT_top_min 0 --numT_top_max 2 \
      --numT_bot_min 0 --numT_bot_max 2 \
      --fr_mode derived --buoyancy shifted --save_plot 0 \
      --max_tries 2 --skip_existing 1 \
      > "$B/logs/shard_${s}.log" 2>&1
    echo "[aspect2 shard $s rc=$?] $(date +%T)" >> "$B/logs/done" ) &
done
wait
echo "--- 宽高比2 完成 $(date +%T)  样本数 $(ls $B/dataset | wc -l) ---"

# ================= 2) 最差 10 组收紧容差重跑（同 seed，问题完全相同） =================
B="$HOME/nc_run/tight"
for id in 241 99 349 587 293 89 432 511 344 591; do
  ( "$PY" genData_stochastic_fourierBC.py \
      --out_root "$B/dataset" --nsamples 1 --start_id $id --seed0 20261016 \
      --res 64 --length 1.0 --T_init 1.0 \
      --dT_min 0.2 --dT_max 1.9 \
      --Ra_min 1e2 --Ra_max 1e6 --Ma_expo 0.16666666666666666 \
      --numT_left_min 1 --numT_left_max 4 \
      --numT_right_min 1 --numT_right_max 4 \
      --numT_top_min 0 --numT_top_max 2 \
      --numT_bot_min 0 --numT_bot_max 2 \
      --fr_mode derived --buoyancy shifted --save_plot 0 \
      --steady_tol 1e-8 --max_tries 1 --skip_existing 0 \
      > "$B/logs/id${id}.log" 2>&1
    echo "[tight id=$id rc=$?] $(date +%T)" >> "$B/logs/done" ) &
done
wait
echo "CPU QUEUE DONE $(date +%T)  tight 样本数 $(ls $B/dataset | wc -l)"