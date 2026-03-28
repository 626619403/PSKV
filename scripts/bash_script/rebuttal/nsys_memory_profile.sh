#!/bin/bash
#SBATCH --job-name=nsys_mem_profile
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=./scripts/logs/rebuttal/nsys_mem_profile_%j.out
#SBATCH --error=./scripts/logs/rebuttal/nsys_mem_profile_%j.err

source ~/.bashrc
conda activate attack
module load cuda
# Resolve absolute paths before any cd
PROJECT_ROOT=$(pwd)/..
TRACE_DIR="${PROJECT_ROOT}/trace/traces"
SAVE_BASE="${PROJECT_ROOT}/results/rebuttal/memory_profile"
SRC_DIR="${PROJECT_ROOT}/src"

mkdir -p "${TRACE_DIR}"
mkdir -p "${PROJECT_ROOT}/scripts/logs/rebuttal"

echo "=========================================="
echo "Nsys Memory Profiling (separate sessions)"
echo "Start time: $(date)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "=========================================="

# Common parameters
MODEL=meta-llama/Llama-2-7b-chat-hf
STEPS=5
SEARCH_WIDTH=64
BATCH_SIZE=16
NUM_PROMPTS=50
NUM_PROMPTS_WILDJB=50

cd ../src
# ============================================================
# Helper function: run nsys profile for one (dataset, sfx, kv) combo
# ============================================================
run_nsys_profile() {
    local DATASET=$1
    local SFX_LEN=$2
    local KV_MODE=$3
    local N_PROMPTS=$4
    local TAG=$5       # e.g. sfx20_harmbench

    local TRACE_NAME="trace_${TAG}_${KV_MODE}"
    local SAVE_DIR="${SAVE_BASE}/${TAG}/nsys"

    mkdir -p "${SAVE_DIR}"

    echo ""
    echo ">>> nsys profile: ${TAG} / kv=${KV_MODE}"
    echo "    trace: ${TRACE_DIR}/${TRACE_NAME}.nsys-rep"

    cd "${SRC_DIR}"
    nsys profile \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        --output="${TRACE_DIR}/${TRACE_NAME}" \
        --force-overwrite=true \
        python nsys_memory_profile.py \
            --model-id ${MODEL} \
            --dataset ${DATASET} \
            --num-prompts ${N_PROMPTS} \
            --batch-size ${BATCH_SIZE} \
            --suffix-length ${SFX_LEN} \
            --steps ${STEPS} \
            --search-width ${SEARCH_WIDTH} \
            --save-dir "${SAVE_DIR}" \
            --kv-modes ${KV_MODE}

    echo "    Completed at: $(date)"
}

# ============================================================
# Part 1: HarmBench, suffix=20 (baseline)
# ============================================================
echo ""
echo "=== Part 1: HarmBench sfx=20 (baseline) ==="

for kv in None Normal Ours; do
    run_nsys_profile harmbench-test50 20 ${kv} ${NUM_PROMPTS} sfx20_harmbench
done

# ============================================================
# Part 2: HarmBench, suffix=40
# ============================================================
echo ""
echo "=== Part 2: HarmBench sfx=40 ==="

for kv in None Normal Ours; do
    run_nsys_profile harmbench-test50 40 ${kv} ${NUM_PROMPTS} sfx40_harmbench
done

# ============================================================
# Part 3: HarmBench, suffix=60
# ============================================================
echo ""
echo "=== Part 3: HarmBench sfx=60 ==="

for kv in None Normal Ours; do
    run_nsys_profile harmbench-test50 60 ${kv} ${NUM_PROMPTS} sfx60_harmbench
done

# ============================================================
# Part 4: WildJailbreak, suffix=20 (longer prompts)
# ============================================================
echo ""
echo "=== Part 4: WildJailbreak sfx=20 ==="

for kv in None Normal Ours; do
    run_nsys_profile wildjailbreak-50 20 ${kv} ${NUM_PROMPTS_WILDJB} sfx20_wildjailbreak
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "All nsys profiling completed!"
echo "End time: $(date)"
echo ""
echo "Traces (open in Nsight Systems GUI):"
echo "  ${TRACE_DIR}/trace_sfx20_harmbench_*.nsys-rep"
echo "  ${TRACE_DIR}/trace_sfx40_harmbench_*.nsys-rep"
echo "  ${TRACE_DIR}/trace_sfx60_harmbench_*.nsys-rep"
echo "  ${TRACE_DIR}/trace_sfx20_wildjailbreak_*.nsys-rep"
echo ""
echo "Memory snapshots (open at https://pytorch.org/memory_viz):"
echo "  ${SAVE_BASE}/sfx20_harmbench/nsys/"
echo "  ${SAVE_BASE}/sfx40_harmbench/nsys/"
echo "  ${SAVE_BASE}/sfx60_harmbench/nsys/"
echo "  ${SAVE_BASE}/sfx20_wildjailbreak/nsys/"
echo ""
echo "CLI stats example:"
echo "  nsys stats ${TRACE_DIR}/trace_sfx20_harmbench_Ours.nsys-rep --report nvtx_pushpop_sum"
echo "=========================================="

exit