#!/bin/bash

src_path=""
i=""
k=""
dataset=""
attack=""
kv_cache=""
random_seed=""

while [[ $# -gt 0 ]]; do
    if [[ $1 == "--src-path" ]]; then
        src_path=$2
        shift 2

    elif [[ $1 == "--train-cfg" ]]; then
        i=$2
        shift 2
    elif [[ $1 == "--eval-cfg" ]]; then
        k=$2
        shift 2
    elif [[ $1 == "--dataset" ]]; then
        dataset=$2
        shift 2
    elif [[ $1 == "--attack" ]]; then
        attack=$2
        shift 2
    elif [[ $1 == "--kv-cache" ]]; then
        kv_cache=$2
        shift 2
    elif [[ $1 == "--random-seed" ]]; then
        random_seed=$2
        shift 2
    else
        shift 1
    fi
done

if [[ -z ${src_path} ]]; then
    echo "Error: --src-path is required"
    exit 1
elif [[ -z $k ]]; then
    echo "Error: --eval-cfg is required"
    exit 1
elif [[ -z $dataset ]] || [[ $dataset != "harmbench-test50" ]] && [[ $dataset != "advbench-first50" ]] && [[ $dataset != "wildjailbreak-50" ]]; then
    echo "Error: --dataset is required and should be 'harmbench-test50', 'advbench-first50', or 'wildjailbreak-50'"
    exit 1
elif [[ -z $attack ]]; then
    echo "Error: --attack is required"
    exit 1
elif [[ -z $kv_cache ]] || [[ $kv_cache != "None" ]] && [[ $kv_cache != "Normal" ]] && [[ $kv_cache != "Ours" ]]; then
    echo "Error: --kv-cache is required and should be one of 'None', 'Normal', or 'Ours'"
    exit 1
fi

if [[ $dataset == "harmbench-test50" ]]; then
    ds_name="harmbench"
elif [[ $dataset == "advbench-first50" ]]; then
    ds_name="advbench"
elif [[ $dataset == "wildjailbreak-50" ]]; then
    ds_name="wildjailbreak"
fi

attack_cfg="$attack"
case "$attack_cfg" in
    beast-vllm) attack_cfg="beast_vllm" ;;
    beast-sglang) attack_cfg="beast_sglang" ;;
esac
case "${attack_cfg}" in
    gcg|gcq|beast|beast_vllm|beast_sglang|autodan-zhu|ample-gcg|adv-prompter)
        ;;
    *)
        echo "Error: --attack should be one of 'gcg', 'gcq', 'beast', 'beast-vllm', 'beast-sglang', 'autodan-zhu', 'ample-gcg', or 'adv-prompter'"
        exit 1
        ;;
esac

cd ${src_path}

model_id="lmsys/vicuna-7b-v1.5"
datacollator="vicuna-chat"
if [[ -z "${LAUNCHER}" ]]; then
    LAUNCHER="python3"
fi


SEED_ARG=""
SEED_SUFFIX=""
if [[ -n ${random_seed} ]]; then
    SEED_ARG="--random-seed ${random_seed}"
    SEED_SUFFIX="_seed${random_seed}"
fi

save_path="../results/vicuna/eval-${attack_cfg}/cfg-$k/kv-cache-${kv_cache}${SEED_SUFFIX}"

${LAUNCHER} evaluate.py \
    --model-id ${model_id} \
    --dataset ${dataset} \
    --datacollator ${datacollator} \
    --evalset-cfg-path ../configs/eval/evalset.yaml \
    --atker-cfg-path ../configs/eval/${attack_cfg}/cfg-$k.yaml \
    --save-dir ${save_path} \
    --exp-type build-evalset \
    --save-name build-${ds_name} \
    --kv-cache ${kv_cache} \
    ${SEED_ARG}

${LAUNCHER} evaluate.py \
    --judger-cfg-path ../configs/eval/judge.yaml \
    --evalset-path ${save_path}/build-${ds_name}_evalset.json \
    --save-dir ${save_path} \
    --exp-type judge-evalset \
    --save-name judge-${ds_name}
