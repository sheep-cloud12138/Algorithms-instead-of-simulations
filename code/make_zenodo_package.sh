#!/usr/bin/env bash
# Assemble the Zenodo deposit package for the IJHMT submission.
# Run inside WSL:  bash /mnt/d/Desktop/算法代替仿真/code/make_zenodo_package.sh
# Output: ~/zenodo_package/ (folder) + ~/zenodo_package.tar (upload this to Zenodo,
# then paste the reserved DOI into manuscript_submission.tex "Data availability").
set -euo pipefail

REPO=/mnt/d/Desktop/算法代替仿真
RUN=~/nc_run
OUT=~/zenodo_package

rm -rf "$OUT" "$OUT".tar
mkdir -p "$OUT/data" "$OUT/results" "$OUT/code"

# --- curated paired-solution datasets (~1.3 GB) ---
cp -r "$RUN/p2"                 "$OUT/data/p2"                  # 600 main pairs (pairs600_fixedp.npz)
cp -r "$RUN/paired/dataset"     "$OUT/data/paired_dataset"      # 20 flows x 5 contrasts
cp -r "$RUN/eps0/dataset"       "$OUT/data/eps0_dataset"        # 27 low-contrast checks
cp -r "$RUN/water/dataset"      "$OUT/data/water_dataset"       # 104 pairs at Pr=4.38
cp -r "$RUN/cavity2/dataset"    "$OUT/data/cavity2_dataset"     # 104 pairs, 2:1 geometry

# --- 3-D paired checks (~350 MB) ---
for d in d3 d3_ra1e4_matched d3_ra1e4_e19 d3_ra3e4 d3_ra1e5 d3_res20; do
    [ -d "$RUN/$d" ] && cp -r "$RUN/$d" "$OUT/data/$d"
done

# --- every result JSON behind the paper's numbers ---
cp -r "$RUN/p7" "$OUT/results/p7"

# --- repository snapshot (tracked files only, no .git / no PDF build artefacts) ---
git -C "$REPO" archive HEAD | tar -x -C "$OUT/code"
rm -rf "$OUT/code/论文/figs" "$OUT/code/research/data"

cp "$REPO/code/zenodo_README.md" "$OUT/README.md"

# --- manifest with sizes and checksums ---
cd "$OUT"
find . -type f -not -name SHA256SUMS -print0 | xargs -0 sha256sum > SHA256SUMS
cd ~
tar -cf zenodo_package.tar -C ~ zenodo_package
du -sh "$OUT" ~/zenodo_package.tar
echo "Done. Upload ~/zenodo_package.tar to Zenodo, then replace"
echo "10.5281/zenodo.XXXXXXX in 论文/manuscript_submission.tex with the reserved DOI."
