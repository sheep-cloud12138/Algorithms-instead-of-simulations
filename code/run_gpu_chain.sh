#!/bin/bash
set -uo pipefail
mkdir -p /home/wjh/nc_run/p7
exec > /home/wjh/nc_run/p7/chain.log 2>&1
cd "/mnt/d/Desktop/算法代替仿真/code" || exit 1
PY=/home/wjh/miniconda3/envs/agent/bin/python
echo "CHAIN START $(date +%T)  (p 通道已修正)"
$PY exp_p2_fixed.py;       echo "[p2_fixed rc=$?] $(date +%T)"
$PY exp_timing.py;         echo "[timing rc=$?] $(date +%T)"
$PY exp_learning_curve.py; echo "[learning_curve rc=$?] $(date +%T)"
$PY exp_prior_ablation.py; echo "[prior_ablation rc=$?] $(date +%T)"
$PY exp_fno.py;            echo "[fno rc=$?] $(date +%T)"
echo "CHAIN_DONE $(date +%T)"