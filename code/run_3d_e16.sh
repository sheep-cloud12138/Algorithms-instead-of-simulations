#!/bin/bash
set -uo pipefail
D="$HOME/nc_run/d3/batch/e16"
mkdir -p "$D/logs" "$D/dataset"
exec > "$D/batch_e16.log" 2>&1
echo "E16 START $(date +%T)"
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
for id in 0 1 2 3 4 5; do
  if [ -f "$D/dataset/sample_$(printf %06d $id)/meta.json" ]; then
    echo "skip eps=1.6 id=$id (exists)"
    continue
  fi
  echo "eps=1.6 id=$id start $(date +%T)"
  timeout 3600 "$PY" genData_3d.py \
    --out_root "$D/dataset" --nsamples 1 --start_id $id --seed0 20261019 \
    --res 16 --length 1.0 --T_init 1.0 --dT 1.6 \
    --Ra_min 1e4 --Ra_max 1e4 --Ma_expo 0.16666666666666666 \
    --n_cells_x_min 1 --n_cells_x_max 3 \
    --fr_mode derived --buoyancy shifted --save_plot 0 \
    --ra_cap v5_3d --max_tries 1 --skip_existing 1 \
    > "$D/logs/id${id}.log" 2>&1
  echo "eps=1.6 id=$id rc=$? $(date +%T)"
done
echo "E16 DONE $(date +%T)"