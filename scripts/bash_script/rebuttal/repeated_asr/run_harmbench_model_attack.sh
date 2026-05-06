#!/bin/bash
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

source ~/.bashrc
conda activate attack
module load cuda/12.4.1

set -e

cd ./scripts

DATASET="harmbench-test50"

if [[ -z "${TASKS}" ]]; then
    echo "Error: TASKS must be provided via sbatch --export"
    exit 1
fi

echo "=========================================="
echo "Repeated ASR HarmBench"
echo "BATCH_NAME=${BATCH_NAME:-unknown}"
echo "EST_MIN=${EST_MIN:-unknown}"
echo "TASKS=${TASKS}"
echo "Start time: $(date)"
echo "=========================================="

IFS=';' read -ra TASK_ARRAY <<< "${TASKS}"
for task in "${TASK_ARRAY[@]}"; do
    read -r model attack eval_cfg kv_cache seed <<< "${task}"
    if [[ -z "${model}" || -z "${attack}" || -z "${eval_cfg}" || -z "${kv_cache}" || -z "${seed}" ]]; then
        echo "Error: malformed task '${task}'"
        exit 1
    fi
    if [[ ! -f "./eval/${model}.sh" ]]; then
        echo "Error: missing eval script ./eval/${model}.sh"
        exit 1
    fi

    echo "[${model}-${attack}-${kv_cache}] seed=${seed}, cfg=${eval_cfg}"
    bash "./eval/${model}.sh" \
        --src-path ../src \
        --eval-cfg "${eval_cfg}" \
        --dataset "${DATASET}" \
        --attack "${attack}" \
        --kv-cache "${kv_cache}" \
        --random-seed "${seed}"
    echo "Completed at: $(date)"
done

echo "=========================================="
echo "Completed ${BATCH_NAME:-unknown}. End time: $(date)"
echo "=========================================="
