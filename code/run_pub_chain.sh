#!/bin/bash
set -uo pipefail
exec >> /home/wjh/nc_run/p7/chain.log 2>&1
while ! grep -q "CHAIN_DONE" /home/wjh/nc_run/p7/chain.log; do sleep 30; done
cd "/mnt/d/Desktop/算法代替仿真/code" || exit 1
PY=/home/wjh/miniconda3/envs/agent/bin/python
echo "=== 公开数据实验 START $(date +%T) ==="
$PY exp_public_transfer.py; echo "[public_transfer rc=$?] $(date +%T)"
echo "=== 全部 GPU 实验 DONE $(date +%T) ==="