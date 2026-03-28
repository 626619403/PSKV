#!/bin/bash
#SBATCH --job-name=multi_seed_asr
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=./scripts/logs/rebuttal/multi_seed_asr_%j.out
#SBATCH --error=./scripts/logs/rebuttal/multi_seed_asr_%j.err

source ~/.bashrc
conda activate attack
module load cuda/12.4.1

# cd to scripts/ so ./eval/ and ./train/ paths resolve correctly
cd ./scripts

echo "=========================================="
echo "Multi-seed ASR experiments for rebuttal"
echo "Start time: $(date)"
echo "=========================================="

SEEDS="0 1 2 3 4"
DATASET="harmbench-test50"

# Experiment 1: Llama2-7B + GCG (Ours vs None)
echo "=== Llama2-7B + GCG ==="
for seed in ${SEEDS}; do
    echo "[Llama2-GCG-Ours] seed=${seed}"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack gcg --kv-cache Ours --random-seed ${seed}
    echo "Completed at: $(date)"

    echo "[Llama2-GCG-None] seed=${seed}"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack gcg --kv-cache None --random-seed ${seed}
    echo "Completed at: $(date)"
done

# Experiment 2: Llama2-7B + BEAST (Ours vs None)
echo "=== Llama2-7B + BEAST ==="
for seed in ${SEEDS}; do
    echo "[Llama2-BEAST-Ours] seed=${seed}"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack beast --kv-cache Ours --random-seed ${seed}
    echo "Completed at: $(date)"

    echo "[Llama2-BEAST-None] seed=${seed}"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack beast --kv-cache None --random-seed ${seed}
    echo "Completed at: $(date)"
done

# Experiment 3: Llama3-8B + BEAST (Ours vs None)
echo "=== Llama3-8B + BEAST ==="
for seed in ${SEEDS}; do
    echo "[Llama3-BEAST-Ours] seed=${seed}"
    bash ./eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset ${DATASET} --attack beast --kv-cache Ours --random-seed ${seed}
    echo "Completed at: $(date)"

    echo "[Llama3-BEAST-None] seed=${seed}"
    bash ./eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset ${DATASET} --attack beast --kv-cache None --random-seed ${seed}
    echo "Completed at: $(date)"
done

# Experiment 4: Vicuna-7B + GCQ (Ours vs None)
echo "=== Vicuna-7B + GCQ ==="
for seed in ${SEEDS}; do
    echo "[Vicuna-GCQ-Ours] seed=${seed}"
    bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack gcq --kv-cache Ours --random-seed ${seed}
    echo "Completed at: $(date)"

    echo "[Vicuna-GCQ-None] seed=${seed}"
    bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack gcq --kv-cache None --random-seed ${seed}
    echo "Completed at: $(date)"
done

# Experiment 5: Vicuna-7B + BEAST (Ours vs None)
echo "=== Vicuna-7B + BEAST ==="
for seed in ${SEEDS}; do
    echo "[Vicuna-BEAST-Ours] seed=${seed}"
    bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack beast --kv-cache Ours --random-seed ${seed}
    echo "Completed at: $(date)"

    echo "[Vicuna-BEAST-None] seed=${seed}"
    bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset ${DATASET} --attack beast --kv-cache None --random-seed ${seed}
    echo "Completed at: $(date)"
done

echo "=========================================="
echo "All multi-seed ASR experiments completed!"
echo "End time: $(date)"
echo "=========================================="

exit
