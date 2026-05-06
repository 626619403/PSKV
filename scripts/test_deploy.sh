#!/bin/bash
#SBATCH --job-name=test_deploy
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=../logs/test_deploy_%j.out
#SBATCH --error=../logs/test_deploy_%j.err

set -euo pipefail

source ~/.bashrc
module load cuda/12.4.1
conda activate attack

mkdir -p ../logs/test_deploy
cd "$(dirname "$0")"

MODEL="${MODEL:-llama2}"
EVAL_CFG="${EVAL_CFG:-1}"
TRAIN_CFG="${TRAIN_CFG:-1}"
DATASET="${DATASET:-harmbench-test50}"
TRAIN_DATASET="${TRAIN_DATASET:-harmbench}"
KV_CACHE="${KV_CACHE:-None}"

echo "=== STARTING DEPLOYMENT SMOKE TEST ==="
echo "MODEL=${MODEL}, EVAL_CFG=${EVAL_CFG}, DATASET=${DATASET}, KV_CACHE=${KV_CACHE}"

echo "--- Training entrypoint: AdvPrompter ---"
bash "./train/${MODEL}.sh" \
    --src-path ../src \
    --train-cfg "${TRAIN_CFG}" \
    --dataset "${TRAIN_DATASET}" \
    --attack adv-prompter \
    --kv-cache "${KV_CACHE}"

echo "--- Training entrypoint: AmpleGCG ---"
bash "./train/${MODEL}.sh" \
    --src-path ../src \
    --train-cfg "${TRAIN_CFG}" \
    --dataset "${TRAIN_DATASET}" \
    --attack ample-gcg \
    --kv-cache "${KV_CACHE}"

for attack in gcg gcq beast autodan-zhu beast-vllm beast-sglang; do
    echo "--- Evaluation entrypoint: ${attack} ---"
    bash "./eval/${MODEL}.sh" \
        --src-path ../src \
        --eval-cfg "${EVAL_CFG}" \
        --dataset "${DATASET}" \
        --attack "${attack}" \
        --kv-cache "${KV_CACHE}" \
        --random-seed 1
done

echo "=== DEPLOYMENT SMOKE TEST COMPLETED ==="
