#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
OUT="$HOME/nc_run/d3_ra1e4_e19"; mkdir -p "$OUT"
for cell in 3 2 1; do
  case $cell in
    3) seed=20265261;;
    2) seed=20275268;;
    1) seed=20285275;;
  esac
  d="$OUT/c${cell}"
  [ -f "$d/dataset/sample_000000/meta.json" ] && { echo "skip c$cell" >> "$OUT/progress.txt"; continue; }
  t0=$(date +%s)
  "$PY" genData_3d.py --out_root "$d" --nsamples 1 --start_id 0 --seed0 "$seed" \
    --res 16 --length 1.0 --dT 1.9 --Ra_min 1e4 --Ra_max 1e4 \
    --n_cells_x_min "$cell" --n_cells_x_max "$cell" --time 20 --time_bq 20 --dt 0.1 \
    --save_plot 0 --max_tries 1 --skip_existing 1 > "$OUT/c$cell.log" 2>&1
  echo "c$cell rc=$? dt=$(( $(date +%s) - t0 ))s $(date +%T)" >> "$OUT/progress.txt"
done
echo "D3RA1E4E19 DONE $(date +'%F %T')" >> "$OUT/progress.txt"