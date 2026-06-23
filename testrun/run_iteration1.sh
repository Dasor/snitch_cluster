#!/bin/bash
# run_iteration1.sh — First GBDT loop iteration for a 128×128×128 GEMM
#
# Usage (from /repo inside the Docker container):
#   bash testrun/run_iteration1.sh
#
# What this does:
#   STEP 1  Generate 50 random valid tile configs   → testrun/iter1_configs.csv
#   STEP 2  Compile + simulate on Snitch cluster    → testrun/128x128x128w*/
#                                                      testrun/iter1_configs-results.csv
#   STEP 3  Train first XGBoost model               → testrun/iter1_model.json

set -euo pipefail

# Must run from repo root so make and myrtle-experiments/ paths resolve correctly
if [ ! -f "myrtle-experiments/many_gemms.sh" ]; then
    echo "Error: run this script from /repo (the repo root), not from testrun/"
    exit 1
fi

# ── Environment variables expected by many_gemms.sh ──────────────────────────
export gemmDir="/repo/sw/kernels/blas/gemm_boundary"
export experimentDir="$(cd testrun && pwd)"   # absolute path; build dirs land here
export beta=0
export spm_opt=0   # partition_banks requires n_tiles=k_tiles=1; disabled for multi-tile search
export TIMEOUT=0
export SIM_BATCH_SIZE=$(nproc)  # run this many simulations in parallel

CONFIGS_CSV="testrun/iter1_configs.csv"
RESULTS_CSV="testrun/iter1_configs-results.csv"
MODEL_JSON="testrun/iter1_model.json"

# ─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════"
echo " STEP 1 — Generate 1000 random configs (128×128×128)"
echo "════════════════════════════════════════════════════"
python testrun/generate_configs.py 128 128 128 1000 --seed 42 --out "$CONFIGS_CSV" --allow-boundary


# ─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════"
echo " STEP 1b — Pick top 5 configs (for quick testing)"
echo "════════════════════════════════════════════════════"

head -n 6 "$CONFIGS_CSV" > "$CONFIGS_CSV.tmp"
mv "$CONFIGS_CSV.tmp" "$CONFIGS_CSV"
rm -f "$CONFIGS_CSV.tmp"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " STEP 2a — Compile top 5 tile configurations"
echo "════════════════════════════════════════════════════"
bash myrtle-experiments/many_gemms.sh "./$CONFIGS_CSV" compile no no no

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " STEP 2b — Run simulator (Verilator) on each ELF"
echo "════════════════════════════════════════════════════"
bash myrtle-experiments/many_gemms.sh "./$CONFIGS_CSV" check run no no

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " STEP 2c — Merge per-config results → single CSV"
echo "════════════════════════════════════════════════════"
python myrtle-experiments/combineTilingSchemeDataIntoSingleCSV.py \
    "./$CONFIGS_CSV" "$experimentDir"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "Error: expected results file not found: $RESULTS_CSV"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " STEP 3 — Train XGBoost model on simulation data"
echo "════════════════════════════════════════════════════"
python testrun/train/boost.py train \
    "$CONFIGS_CSV" \
    "$RESULTS_CSV" \
    --save "$MODEL_JSON"

echo ""
echo "Done. Outputs:"
echo "  Configs:  $CONFIGS_CSV"
echo "  Results:  $RESULTS_CSV"
echo "  Model:    $MODEL_JSON"
