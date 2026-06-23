#!/bin/bash
# run_loop.sh — Generic Ansor-style GBDT loop for GEMM tile-size search
#
# Iteration 1: random sampling → compile → simulate → train initial model
# Iterations 2+: GA (seeded from all previous configs, guided by latest model)
#              → compile → simulate → retrain on ALL accumulated data
#
# Usage (from /repo inside Docker):
#   bash testrun/run_loop.sh [OPTIONS]
#
# Options:
#   --M INT              GEMM M dimension           (default: 128)
#   --N INT              GEMM N dimension           (default: 128)
#   --K INT              GEMM K dimension           (default: 128)
#   --iterations INT     total loop iterations      (default: 5)
#   --init-pool INT      random configs generated in iter 1 (default: 1000)
#   --init-configs INT   top-N of the pool to actually simulate (default: 50)
#   --ga-candidates INT  GA top-N per iter 2+       (default: 30)
#   --ga-pop INT         GA population size         (default: 200)
#   --ga-generations INT GA generations per run     (default: 100)
#   --allow-boundary     enable boundary/remainder tile mode
#   --workdir PATH       artifact directory         (default: testrun)
#   --seed INT           random seed for reproducibility

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
M=128
N=128
K=128
N_ITERS=5
INIT_POOL=1000
INIT_CONFIGS=50
GA_CANDIDATES=30
GA_POP=200
GA_GENS=100
ALLOW_BOUNDARY=""
WORKDIR="testrun"
SEED=""

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --M)               M="$2";              shift 2 ;;
        --N)               N="$2";              shift 2 ;;
        --K)               K="$2";              shift 2 ;;
        --iterations)      N_ITERS="$2";        shift 2 ;;
        --init-pool)       INIT_POOL="$2";      shift 2 ;;
        --init-configs)    INIT_CONFIGS="$2";   shift 2 ;;
        --ga-candidates)   GA_CANDIDATES="$2";  shift 2 ;;
        --ga-pop)          GA_POP="$2";         shift 2 ;;
        --ga-generations)  GA_GENS="$2";        shift 2 ;;
        --allow-boundary)  ALLOW_BOUNDARY="--allow-boundary"; shift ;;
        --workdir)         WORKDIR="$2";        shift 2 ;;
        --seed)            SEED="$2";           shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [ ! -f "myrtle-experiments/many_gemms.sh" ]; then
    echo "Error: run this script from /repo (the repo root), not from testrun/"
    exit 1
fi

mkdir -p "$WORKDIR"

# ── Environment variables required by many_gemms.sh ──────────────────────────
export gemmDir="/repo/sw/kernels/blas/gemm_boundary"
export experimentDir
experimentDir="$(cd "$WORKDIR" && pwd)"
export beta=0
export spm_opt=0
export TIMEOUT=0

# ── Helper: seed arg (empty string if no seed provided) ───────────────────────
seed_arg() {
    local iter="$1"
    if [ -n "$SEED" ]; then
        echo "--seed $((SEED + iter - 1))"
    fi
}

# ── Helper: print a section banner ────────────────────────────────────────────
banner() {
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for ITER in $(seq 1 "$N_ITERS"); do

    CONFIGS_CSV="$WORKDIR/iter${ITER}_configs.csv"
    RESULTS_CSV="$WORKDIR/iter${ITER}_configs-results.csv"
    MODEL_JSON="$WORKDIR/iter${ITER}_model.json"

    banner "ITERATION $ITER / $N_ITERS  (${M}×${N}×${K})"

    # ── STEP 1: Generate candidate configs ────────────────────────────────────
    echo ""
    echo "── Step 1: Generate configs ──"

    if [ "$ITER" -eq 1 ]; then
        # First iteration: generate a large random pool, keep the top INIT_CONFIGS
        # by tile size (largest tiles use more of the scratchpad → better perf).
        # generate_configs.py already sorts by Space Needed in L1 descending.
        POOL_CSV="$WORKDIR/iter1_pool.csv"
        python testrun/generate_configs.py \
            "$M" "$N" "$K" "$INIT_POOL" \
            $(seed_arg "$ITER") $ALLOW_BOUNDARY \
            --out "$POOL_CSV"

        # Keep header + top INIT_CONFIGS data rows
        { head -n 1 "$POOL_CSV"; tail -n +2 "$POOL_CSV" | head -n "$INIT_CONFIGS"; } \
            > "$CONFIGS_CSV"
        echo "Selected top $INIT_CONFIGS configs (by tile size) from pool of $INIT_POOL → $CONFIGS_CSV"
    else
        # Subsequent iterations: GA guided by the previous model.
        # Seed from ALL configs evaluated so far (richer gene pool).
        PREV_MODEL="$WORKDIR/iter$((ITER - 1))_model.json"
        GA_SEED_CSV="$WORKDIR/iter${ITER}_ga_seed.csv"

        python3 - <<PYEOF
import pandas as pd
dfs = [pd.read_csv(f"$WORKDIR/iter{i}_configs.csv") for i in range(1, $ITER)]
pd.concat(dfs).drop_duplicates("FakeNN JSON Name").to_csv("$GA_SEED_CSV", index=False)
PYEOF

        python testrun/train/ga.py \
            --M "$M" --N "$N" --K "$K" \
            --model "$PREV_MODEL" \
            --seed-csv "$GA_SEED_CSV" \
            --out "$CONFIGS_CSV" \
            --pop "$GA_POP" \
            --generations "$GA_GENS" \
            --top-n "$GA_CANDIDATES" \
            $(seed_arg "$ITER") $ALLOW_BOUNDARY
    fi

    # ── STEP 2a: Compile ──────────────────────────────────────────────────────
    echo ""
    echo "── Step 2a: Compile ──"
    bash myrtle-experiments/many_gemms.sh "./$CONFIGS_CSV" compile no no no

    # ── STEP 2b: Run simulations ──────────────────────────────────────────────
    echo ""
    echo "── Step 2b: Simulate ──"
    bash myrtle-experiments/many_gemms.sh "./$CONFIGS_CSV" check run no no

    # ── STEP 2c: Merge per-config results into a single CSV ───────────────────
    echo ""
    echo "── Step 2c: Merge results ──"
    python myrtle-experiments/combineTilingSchemeDataIntoSingleCSV.py \
        "./$CONFIGS_CSV" "$experimentDir"

    if [ ! -f "$RESULTS_CSV" ]; then
        echo "Error: $RESULTS_CSV not found — simulation may have failed"
        exit 1
    fi

    # ── STEP 3: Train on ALL data accumulated so far ──────────────────────────
    echo ""
    echo "── Step 3: Train model on iterations 1–${ITER} ──"

    ALL_FEATURES_CSV="$WORKDIR/all_configs_iter1_to_${ITER}.csv"
    ALL_RESULTS_CSV="$WORKDIR/all_results_iter1_to_${ITER}.csv"

    python3 - <<PYEOF
import pandas as pd

feat_dfs    = [pd.read_csv(f"$WORKDIR/iter{i}_configs.csv")         for i in range(1, $ITER + 1)]
results_dfs = [pd.read_csv(f"$WORKDIR/iter{i}_configs-results.csv") for i in range(1, $ITER + 1)]

(pd.concat(feat_dfs)
   .drop_duplicates("FakeNN JSON Name")
   .to_csv("$ALL_FEATURES_CSV", index=False))

(pd.concat(results_dfs)
   .drop_duplicates("FakeNN JSON Name")
   .to_csv("$ALL_RESULTS_CSV", index=False))

total = len(pd.read_csv("$ALL_FEATURES_CSV"))
print(f"Combined dataset: {total} unique configs across {$ITER} iteration(s)")
PYEOF

    python testrun/train/boost.py train \
        "$ALL_FEATURES_CSV" \
        "$ALL_RESULTS_CSV" \
        --save "$MODEL_JSON"

    echo ""
    echo "Iteration $ITER complete → model: $MODEL_JSON"

done

# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
banner "LOOP COMPLETE — ${N_ITERS} iteration(s) for ${M}×${N}×${K}"

python3 - <<PYEOF
import glob
import pandas as pd

files = sorted(glob.glob("$WORKDIR/iter*_configs-results.csv"))
if not files:
    print("No results found.")
else:
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df["Global Sim E2E_dma"] > 0]
    if df.empty:
        print("No valid (non-timeout) simulation results.")
    else:
        best = df.loc[df["Global Sim E2E_dma"].idxmin()]
        worst = df.loc[df["Global Sim E2E_dma"].idxmax()]
        print(f"Configs evaluated : {len(df)}")
        print(f"Best  → {best['FakeNN JSON Name']:40s}  {best['Global Sim E2E_dma']:.0f} cycles")
        print(f"Worst → {worst['FakeNN JSON Name']:40s}  {worst['Global Sim E2E_dma']:.0f} cycles")
        improvement = (worst["Global Sim E2E_dma"] - best["Global Sim E2E_dma"]) / worst["Global Sim E2E_dma"] * 100
        print(f"Range : {improvement:.1f}% spread between best and worst evaluated")
PYEOF
