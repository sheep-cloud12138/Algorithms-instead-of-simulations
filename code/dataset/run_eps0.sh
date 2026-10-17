set -uo pipefail
exec > /home/wjh/nc_run/eps0/run.log 2>&1
cd "$HOME/nc_run/DataGeneration" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/micromamba/envs/fx/bin/python"
B="$HOME/nc_run/eps0"
i=0
for eps in 0.02 0.05 0.10; do
 for Ra in 1e3 1e4 1e5; do
  start=$((i*3)); i=$((i+1))
  ( "$PY" genData_stochastic_fourierBC.py \
      --out_root "$B/dataset" --nsamples 3 --start_id $start --seed0 $((20261016+7919*start)) \
      --res 64 --length 1.0 --T_init 1.0 \
      --dT_min $eps --dT_max $eps \
      --Ra_min $Ra --Ra_max $Ra --ra_cap off --Ma_expo 0.16666666666666666 \
      --numT_left_min 1 --numT_left_max 4 \
      --numT_right_min 1 --numT_right_max 4 \
      --numT_top_min 0 --numT_top_max 2 \
      --numT_bot_min 0 --numT_bot_max 2 \
      --fr_mode derived --buoyancy shifted --save_plot 0 \
      --max_tries 2 --skip_existing 1 \
      > "$B/logs/e${eps}_Ra${Ra}.log" 2>&1
    echo "[eps=$eps Ra=$Ra rc=$?] $(date +%T)" >> "$B/logs/done" ) &
 done
done
wait
echo "EPS0_DONE $(date +%T)"
ls "$B/dataset" | wc -l
