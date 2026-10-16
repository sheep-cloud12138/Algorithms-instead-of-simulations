#!/bin/bash
# 3D 迁移试探：同一个问题（同 seed0/start_id）在不同 eps 下各解一次 => 天然配对。
# 结论指导：小 eps 的源项是 O(eps)，应当比 released 的 eps=1.2 好收敛。
set -uo pipefail
exec > /home/wjh/nc_run/d3_probe.log 2>&1
echo "3D PROBE WAIT $(date +%T)"
while pgrep -f "run_cpu_queue.sh" > /dev/null; do sleep 30; done
echo "--- CPU 队列已结束，开始 3D 试探 $(date +%T) ---"
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
OK() { ls "$HOME/nc_run/d3/$1/dataset" 2>/dev/null | wc -l; }

for stage in "0.2:e02" "0.8:e08" "1.2:e12"; do
  EPS="${stage%%:*}"; TAG="${stage##*:}"
  D="$HOME/nc_run/d3/$TAG"; mkdir -p "$D/logs"
  echo "=== stage eps=$EPS  $(date +%T) ==="
  for id in 0 1 2; do
    ( timeout 1800 "$PY" genData_3d.py \
        --out_root "$D/dataset" --nsamples 1 --start_id $id --seed0 20261019 \
        --res 16 --length 1.0 --T_init 1.0 --dT $EPS \
        --Ra_min 1e4 --Ra_max 1e4 --Ma_expo 0.16666666666666666 \
        --n_cells_x_min 1 --n_cells_x_max 3 \
        --fr_mode derived --buoyancy shifted --save_plot 0 \
        --ra_cap v5_3d --max_tries 1 --skip_existing 1 \
        > "$D/logs/id${id}.log" 2>&1
      echo "[3D eps=$EPS id=$id rc=$?] $(date +%T)" >> "$D/logs/done" ) &
  done
  wait
  n=$(OK "$TAG"); echo "stage eps=$EPS 完成：成功样本数 $n"
  if [ "$n" -eq 0 ]; then echo "eps=$EPS 全军覆没，停止 3D 试探"; break; fi
done
echo "3D PROBE DONE $(date +%T)"
for t in e02 e08 e12; do echo "$t: $(OK $t)"; done