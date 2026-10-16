#!/bin/bash
set -uo pipefail
LOG=/home/wjh/nc_run/p7/chain_rr2.log
exec >> $LOG 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== 重跑公开 4000 迁移（修缓存设备 bug）$(date +%T) ==="
cd "/mnt/d/Desktop/算法代替仿真/code" || exit 1
PY=/home/wjh/miniconda3/envs/agent/bin/python
$PY exp_public_transfer.py; echo "[rc=$?] $(date +%T)"
echo "=== exp_ml_final START $(date +%T) ==="; $PY exp_ml_final.py; echo "[rc=$?] $(date +%T)"
echo "=== exp_fno_final START $(date +%T) ==="; $PY exp_fno_final.py; echo "[rc=$?] $(date +%T)"
echo "=== exp_lc_final START $(date +%T) ==="; $PY exp_lc_final.py; echo "[rc=$?] $(date +%T)"
echo "=== ALL DONE $(date +%T) ==="