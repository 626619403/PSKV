#!/usr/bin/env bash
set -e

# --- Argument Variables Initialization ---
src_path=""
i=""        # train_cfg
dataset=""
attack=""
kv_cache=""
random_seed=""

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --src-path)
            src_path=$2
            shift 2
            ;;

        --train-cfg)
            i=$2
            shift 2
            ;;
        --dataset)
            dataset=$2
            shift 2
            ;;
        --attack)
            attack=$2
            shift 2
            ;;
        --kv-cache)
            kv_cache=$2
            shift 2
            ;;
        --random-seed)
            random_seed=$2
            shift 2
            ;;
        *)
            shift 1
            ;;
    esac
done

# --- Validation ---
if [[ -z ${src_path} ]]; then
    echo "Error: --src-path is required" >&2
    exit 1
elif [[ -z $i ]]; then
    echo "Error: --train-cfg is required" >&2
    exit 1
elif [[ -z $dataset ]] || [[ $dataset != "harmbench" ]] && [[ $dataset != "advbench" ]]; then
    echo "Error: --dataset is required and should be either 'harmbench' or 'advbench'" >&2
    exit 1
elif [[ -z $attack ]] || [[ $attack != "ample-gcg" ]] && [[ $attack != "adv-prompter" ]]; then
    echo "Error: --attack is required and should be either 'ample-gcg' or 'adv-prompter'" >&2
    exit 1
elif [[ -z $kv_cache ]] || [[ $kv_cache != "None" ]] && [[ $kv_cache != "Normal" ]] && [[ $kv_cache != "Ours" ]]; then
    echo "Error: --kv-cache is required and should be one of 'None', 'Normal', or 'Ours'" >&2
    exit 1
fi

# --- Configuration ---
cd "${src_path}"

SEED_ARG=""
SEED_SUFFIX=""
if [[ -n ${random_seed} ]]; then
    SEED_ARG="--random-seed ${random_seed}"
    SEED_SUFFIX="_seed${random_seed}"
fi

model_id="meta-llama/Meta-Llama-3-8B-Instruct"
datacollator="llama3-chat"
save_path="../results/llama3/train-${attack}/cfg-${i}/kv-cache-${kv_cache}${SEED_SUFFIX}"

# --- Execute Training ---
echo "--- Starting Training: ${attack} ---"
python train.py \
    --model-id "${model_id}" \
    --lora-cfg-path "../configs/train/lora.yaml" \
    --atker-cfg-path "../configs/train/${attack}/cfg-${i}.yaml" \
    --dataset "${dataset}" \
    --attack-type "${attack}" \
    --save-dir "${save_path}" \
    --save-name attack_train \
    --target-llm "${model_id}" \
    --kv-cache "${kv_cache}" \
    ${SEED_ARG}

echo "--- Training Complete ---"