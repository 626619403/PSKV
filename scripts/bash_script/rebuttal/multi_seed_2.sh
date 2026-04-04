#!/bin/bash
#SBATCH --job-name=seed_batch1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=./scripts/logs/rebuttal/seed_batch1_%j.out
#SBATCH --error=./scripts/logs/rebuttal/seed_batch1_%j.err

source ~/.bashrc
conda activate attack
module load cuda/12.4.1
cd ./scripts

echo "=========================================="
echo "Multi-seed ASR — Batch 1 (seeds 0-4)"
echo "BEAST + Llama2, None vs Ours"
echo "Start time: $(date)"
echo "=========================================="

SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
DATASET="harmbench-test50"

for seed in ${SEEDS}; do
    echo "--- seed=${seed} ---"

    echo "[Llama2-BEAST-None] seed=${seed}"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack beast --kv-cache None --random-seed ${seed}
    echo "Completed at: $(date)"

    echo "[Llama2-BEAST-Ours] seed=${seed}"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack beast --kv-cache Ours --random-seed ${seed}
    echo "Completed at: $(date)"
done

echo "=========================================="
echo "Batch 1 completed! End time: $(date)"
echo "=========================================="
exit