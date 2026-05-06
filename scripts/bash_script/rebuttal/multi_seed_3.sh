#!/bin/bash
#SBATCH --job-name=seed_batch2
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=./scripts/logs/rebuttal/seed_batch2_%j.out
#SBATCH --error=./scripts/logs/rebuttal/seed_batch2_%j.err

source ~/.bashrc
conda activate attack
module load cuda/12.4.1
cd ./scripts

echo "=========================================="
echo "Multi-seed ASR — Batch 2 (seeds 5-9)"
echo "GCG + Llama2, None vs Ours"
echo "Start time: $(date)"
echo "=========================================="

SEEDS="5 6 7 8 9"
DATASET="harmbench-test50"

for seed in ${SEEDS}; do
    echo "--- seed=${seed} ---"

    echo "[Llama2-GCG-None] seed=${seed}"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack gcg   --kv-cache None --random-seed ${seed}
    echo "Completed at: $(date)"

    echo "[Llama2-GCG-Ours] seed=${seed}"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack gcg --kv-cache Ours --random-seed ${seed}
    echo "Completed at: $(date)"
done

echo "=========================================="
echo "Batch 2 completed! End time: $(date)"
echo "=========================================="
exit
