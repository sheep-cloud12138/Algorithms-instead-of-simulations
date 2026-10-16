#!/bin/bash
set -uo pipefail
LOG=/home/wjh/nc_run/p7/chain_ml.log
exec >> $LOG 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== 统一消融链排队 $(date +%T)：等公开迁移脚本退出 ==="
ok=0
while [ $ok -lt 4 ]; do
  if pgrep -f exp_public_ > /dev/null; then ok=0; else ok=$((ok+1)); fi
  sleep 30
done
cd "/mnt/d/Desktop/算法代替仿真/code" || exit 1
PY=/home/wjh/miniconda3/envs/agent/bin/python
echo "=== exp_ml_final START $(date +%T) ==="; $PY exp_ml_final.py; echo "[rc=$?] $(date +%T)"
echo "=== exp_fno_final START $(date +%T) ==="; $PY exp_fno_final.py; echo "[rc=$?] $(date +%T)"
echo "=== exp_lc_final START $(date +%T) ==="; $PY exp_lc_final.py; echo "[rc=$?] $(date +%T)"
echo "=== ML CHAIN DONE $(date +%T) ==="