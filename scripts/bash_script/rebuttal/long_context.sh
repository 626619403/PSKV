#!/bin/bash
#SBATCH --job-name=long_context_mem
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=./scripts/logs/rebuttal/long_context_mem_%j.out
#SBATCH --error=./scripts/logs/rebuttal/long_context_mem_%j.err

source ~/.bashrc
conda activate attack
cd ./scripts

module load cuda
echo "=========================================="
echo "Long-context experiments + Memory Profiling"
echo "Start time: $(date)"
echo "=========================================="

# Common memory profiling parameters (from gcg config, with fewer steps)
PROFILE_STEPS=5
TOPK=256
SEARCH_WIDTH=64
BATCH_SIZE=16
WIDTH_BS=16
NUM_PROMPTS=50
NUM_PROMPTS_WILDJB=10
SAVE_BASE=../results/rebuttal/memory_profile

# ============================================================
# Part 1: Longer suffix lengths with standard HarmBench dataset
# ============================================================
echo "=== Part 1: Longer Suffix Lengths (GCG on HarmBench) ==="

# --- suffix=40 ---
for kv in None Normal Ours; do
    echo "[Llama2-GCG-sfx40-${kv}]"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg long-sfx40 --dataset harmbench-test50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

echo "[Memory Profile: sfx40 on Llama2]"
cd ../src
python test/memory_profile.py \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --dataset harmbench-test50 \
    --num-prompts ${NUM_PROMPTS} \
    --suffix-length 40 \
    --steps ${PROFILE_STEPS} \
    --search-width ${SEARCH_WIDTH} \
    --save-dir ${SAVE_BASE}/sfx40_harmbench
cd ../scripts
echo "Memory profile sfx40 completed at: $(date)"

# --- suffix=60 ---
for kv in None Normal Ours; do
    echo "[Llama2-GCG-sfx60-${kv}]"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg long-sfx60 --dataset harmbench-test50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

echo "[Memory Profile: sfx60 on Llama2]"
cd ../src
python test/memory_profile.py \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --dataset harmbench-test50 \
    --num-prompts ${NUM_PROMPTS} \
    --suffix-length 60 \
    --steps ${PROFILE_STEPS} \
    --search-width ${SEARCH_WIDTH} \
    --save-dir ${SAVE_BASE}/sfx60_harmbench
cd ../scripts
echo "Memory profile sfx60 completed at: $(date)"

# ============================================================
# Part 2: Longer prompts using WildJailbreak dataset
# ============================================================
echo "=== Part 2: Longer Prompts (WildJailbreak) ==="

for kv in Ours Normal None; do
    echo "[Llama2-GCG-WildJB-${kv}]"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset wildjailbreak-50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

echo "[Memory Profile: sfx20 on WildJailbreak]"
cd ../src
python test/memory_profile.py \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --dataset wildjailbreak-50 \
    --num-prompts ${NUM_PROMPTS_WILDJB} \
    --suffix-length 20 \
    --steps ${PROFILE_STEPS} \
    --search-width ${SEARCH_WIDTH} \
    --save-dir ${SAVE_BASE}/sfx20_wildjailbreak
cd ../scripts
echo "Memory profile WildJailbreak completed at: $(date)"

echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo ""
echo "Memory snapshots saved under:"
echo "  ${SAVE_BASE}/sfx20_harmbench/"
echo "  ${SAVE_BASE}/sfx40_harmbench/"
echo "  ${SAVE_BASE}/sfx60_harmbench/"
echo "  ${SAVE_BASE}/sfx20_wildjailbreak/"
echo ""
echo "Open .pickle files at https://pytorch.org/memory_viz"
echo "=========================================="

exit
