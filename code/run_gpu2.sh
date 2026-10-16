#!/bin/bash
set -uo pipefail
exec >> /home/wjh/nc_run/p7/chain.log 2>&1
cd "/mnt/d/Desktop/算法代替仿真/code" || exit 1
PY=/home/wjh/miniconda3/envs/agent/bin/python
echo "=== 公开数据实验重跑 START $(date +%T)（串行，避免内存冲突）==="
$PY exp_public_transfer.py; echo "[public_transfer rc=$?] $(date +%T)"

# ---- 3D 试探：一次只跑一个样本，res=16，先小 eps ----
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY2=/home/wjh/micromamba/envs/fx/bin/python
cd /home/wjh/nc_run/DataGeneration || exit 1
echo "=== 3D 串行试探 START $(date +%T) ==="
for spec in "0.2 0" "0.2 1" "0.4 0" "0.8 0" "1.2 0"; do
  set -- $spec; EPS=$1; ID=$2
  D=/home/wjh/nc_run/d3b/e$EPS; mkdir -p "$D/logs"
  timeout 2400 "$PY2" genData_3d.py \
      --out_root "$D/dataset" --nsamples 1 --start_id $ID --seed0 20261019 \
      --res 16 --length 1.0 --T_init 1.0 --dT $EPS \
      --Ra_min 1e4 --Ra_max 1e4 --Ma_expo 0.16666666666666666 \
      --n_cells_x_min 1 --n_cells_x_max 3 \
      --fr_mode derived --buoyancy shifted --save_plot 0 \
      --ra_cap v5_3d --max_tries 1 --skip_existing 1 \
      > "$D/logs/id${ID}.log" 2>&1
  rc=$?
  if [ -f "$D/dataset/sample_00000${ID}/meta.json" ]; then
    ok=$("$PY" -c "import json;m=json.load(open('$D/dataset/sample_00000${ID}/meta.json'));print(m['march']['comp'].get('ok'),m['march']['bouss'].get('ok'))")
  else ok="no-meta"; fi
  echo "[3D eps=$EPS id=$ID rc=$rc ok=$ok] $(date +%T)" | tee -a /home/wjh/nc_run/d3b/done
  if [ "$rc" != "0" ]; then echo "3D: rc=$rc 终止"; break; fi
done
echo "=== 全部 DONE $(date +%T) ==="