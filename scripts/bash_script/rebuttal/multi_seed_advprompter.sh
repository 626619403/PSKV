#!/bin/bash
#SBATCH --job-name=multi_seed_advprompter
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=../logs/rebuttal/multi_seed_advprompter_%j.out
#SBATCH --error=../logs/rebuttal/multi_seed_advprompter_%j.err

source ~/.bashrc
conda activate attack

echo "=========================================="
echo "Multi-seed AdvPrompter experiments for rebuttal"
echo "Start time: $(date)"
echo "=========================================="

SEEDS="0 1 2 3 4"

# Experiment 6: Llama3-8B + AdvPrompter (Ours vs None)
echo "=== Llama3-8B + AdvPrompter ==="
for seed in ${SEEDS}; do
    echo "[Llama3-AdvPrompter-Ours] seed=${seed}"
    bash ./train/llama3.sh --src-path ../src --train-cfg 2 --dataset harmbench --attack adv-prompter --kv-cache Ours --random-seed ${seed}
    echo "Completed at: $(date)"

    echo "[Llama3-AdvPrompter-None] seed=${seed}"
    bash ./train/llama3.sh --src-path ../src --train-cfg 2 --dataset harmbench --attack adv-prompter --kv-cache None --random-seed ${seed}
    echo "Completed at: $(date)"
done

echo "=========================================="
echo "All AdvPrompter multi-seed experiments completed!"
echo "End time: $(date)"
echo "=========================================="

exit
