#!/bin/bash
# 重跑 exp_ml_final 的内插阶段；GPU 若再抽风（nvlddmkm 153）自动重试，靠 json 里的逐次检查点续跑。
PY=/home/wjh/miniconda3/envs/agent/bin/python
cd "/mnt/d/Desktop/算法代替仿真/code" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 8); do
  echo "=== exp_ml_final interp attempt $i $(date +%T) ==="
  $PY exp_ml_final.py interp
  rc=$?
  echo "[rc=$rc] $(date +%T)"
  if [ $rc -eq 0 ]; then echo "=== OK $(date +%T) ==="; exit 0; fi
  sleep 30
done
echo "=== GAVE UP after 8 attempts $(date +%T) ==="
exit 1