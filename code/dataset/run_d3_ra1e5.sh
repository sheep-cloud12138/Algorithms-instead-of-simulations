#!/usr/bin/env bash
# Third Rayleigh number for the 3-D check: 3 flows x 3 contrasts = 9 pairs.
set -uo pipefail
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
OUT="$HOME/nc_run/d3_ra1e5"; mkdir -p "$OUT"
for cell in 3 2 1; do
  case $cell in
    3) seed=20265261;;
    2) seed=20275268;;
    1) seed=20285275;;
  esac
  for e in 0.4 1.2 1.9; do
    tag="c${cell}_e${e}"
    d="$OUT/$tag"
    if [ "$tag" = "c1_e1.2" ] && [ -f "$OUT/test_c1_e1.2/sample_000000/meta.json" ]; then
      d="$OUT/test_c1_e1.2"
    fi
    if [ -f "$d/dataset/sample_000000/meta.json" ] || [ -f "$d/sample_000000/meta.json" ]; then
      echo "skip $tag" >> "$OUT/progress.txt"; continue
    fi
    t0=$(date +%s)
    timeout 3600 "$PY" genData_3d.py --out_root "$d" --nsamples 1 --start_id 0 --seed0 "$seed" \
      --res 16 --length 1.0 --dT "$e" --Ra_min 1e5 --Ra_max 1e5 \
      --n_cells_x_min "$cell" --n_cells_x_max "$cell" --time 20 --time_bq 20 --dt 0.1 \
      --save_plot 0 --max_tries 1 --skip_existing 1 > "$OUT/$tag.log" 2>&1
    echo "$tag rc=$? dt=$(( $(date +%s) - t0 ))s $(date +%T)" >> "$OUT/progress.txt"
  done
done
echo "D3RA1E5 DONE $(date +'%F %T')" >> "$OUT/progress.txt"
