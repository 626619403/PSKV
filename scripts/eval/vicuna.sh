#!/bin/bash

src_path=""
i=""
k=""
dataset=""
attack=""
kv_cache=""

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
elif [[ -z $dataset ]] || [[ $dataset != "harmbench-test50" ]] && [[ $dataset != "advbench-first50" ]]; then
    echo "Error: --dataset is required and should be either 'harmbench-test50' or 'advbench-first50'"
    exit 1
elif [[ -z $attack ]]; then
    echo "Error: --attack is required and should be either 'ample-gcg' 'gcg', 'beast', 'autodan-zhu' or 'gcq'"
    exit 1
elif [[ -z $kv_cache ]] || [[ $kv_cache != "None" ]] && [[ $kv_cache != "Normal" ]] && [[ $kv_cache != "Ours" ]]; then
    echo "Error: --kv-cache is required and should be one of 'None', 'Normal', or 'Ours'"
    exit 1
fi

if [[ $dataset == "harmbench-test50" ]]; then
    ds_name="harmbench"
elif [[ $dataset == "advbench-first50" ]]; then
    ds_name="advbench"
fi

cd ${src_path}

model_id="lmsys/vicuna-7b-v1.5"
datacollator="vicuna-chat"
if LAUNCHER="" ; then
    LAUNCHER="python3"
fi


save_path="../results/vicuna/eval-${attack}/cfg-$k/kv-cache-${kv_cache}"

${LAUNCHER} evaluate.py \
    --model-id ${model_id} \
    --dataset ${dataset} \
    --datacollator ${datacollator} \
    --evalset-cfg-path ../configs/eval/evalset.yaml \
    --atker-cfg-path ../configs/eval/${attack}/cfg-$k.yaml \
    --save-dir ${save_path} \
    --exp-type build-evalset \
    --save-name build-${ds_name} \
    --kv-cache ${kv_cache} 


${LAUNCHER} evaluate.py \
    --judger-cfg-path ../configs/eval/judge.yaml \
    --evalset-path ${save_path}/build-${ds_name}_evalset.json \
    --save-dir ${save_path} \
    --exp-type judge-evalset \
    --save-name judge-${ds_name}
