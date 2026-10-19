#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
OUT="$HOME/nc_run/d3_ra1e4_matched"; mkdir -p "$OUT"
# start only after the Ra=3e4 repair has finished and the box is idle
while ! grep -q "D3RA3E4 DONE" "$HOME/nc_run/d3_ra3e4/progress.txt" 2>/dev/null; do sleep 30; done
while pgrep -f genData_3d.py > /dev/null; do sleep 30; done
for cell in 3 2 1; do
  case $cell in
    3) seed=20265261;;
    2) seed=20275268;;
    1) seed=20285275;;
  esac
  for e in 0.4 1.2; do
    tag="c${cell}_e${e}"
    d="$OUT/$tag"
    [ -f "$d/sample_000000/meta.json" ] && { echo "skip $tag" >> "$OUT/progress.txt"; continue; }
    t0=$(date +%s)
    "$PY" genData_3d.py --out_root "$d" --nsamples 1 --start_id 0 --seed0 "$seed" \
      --res 16 --length 1.0 --dT "$e" --Ra_min 1e4 --Ra_max 1e4 \
      --n_cells_x_min "$cell" --n_cells_x_max "$cell" --time 20 --time_bq 20 --dt 0.1 \
      --save_plot 0 --max_tries 1 --skip_existing 1 > "$OUT/$tag.log" 2>&1
    echo "$tag rc=$? dt=$(( $(date +%s) - t0 ))s $(date +%T)" >> "$OUT/progress.txt"
  done
done
echo "D3RA1E4MATCH DONE $(date +'%F %T')" >> "$OUT/progress.txt"