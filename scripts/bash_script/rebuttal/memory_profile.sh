#!/bin/bash
#SBATCH --job-name=mem_profile
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=/ibex/user/wangx0t/project/PSKV/scripts/logs/rebuttal/memory_profile_%j.out
#SBATCH --error=/ibex/user/wangx0t/project/PSKV/scripts/logs/rebuttal/memory_profile_%j.err

source ~/.bashrc
conda activate attack

cd /ibex/user/wangx0t/project/PSKV/scripts

echo "=========================================="
echo "Per-layer Memory Profiling for GCG + PSKV"
echo "Start time: $(date)"
echo "=========================================="

cd ../src

# Profile Llama-2-7B
echo "[1/2] Profiling Llama-2-7b-chat-hf"
python memory_profile.py \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --dataset harmbench-test50 \
    --num-prompts 5 \
    --suffix-length 20 \
    --steps 3 \
    --search-width 64 \
    --save-dir ../results/rebuttal/memory_profile
echo "Completed at: $(date)"

# Profile Llama-3-8B
echo "[2/2] Profiling Meta-Llama-3-8B-Instruct"
python memory_profile.py \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset harmbench-test50 \
    --num-prompts 5 \
    --suffix-length 20 \
    --steps 3 \
    --search-width 64 \
    --save-dir ../results/rebuttal/memory_profile
echo "Completed at: $(date)"

echo "=========================================="
echo "Memory profiling completed!"
echo "End time: $(date)"
echo "=========================================="

exit
