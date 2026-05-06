#!/bin/bash
set -e

mkdir -p ./scripts/logs/rebuttal/repeated_asr

submit_batch() {
    local name="$1"
    local time_limit="$2"
    local est_min="$3"
    local tasks="$4"

    sbatch \
        --job-name="${name}" \
        --time="${time_limit}" \
        --exclude=dgpu609-10 \
        --output="./scripts/logs/rebuttal/repeated_asr/${name}_%j.out" \
        --error="./scripts/logs/rebuttal/repeated_asr/${name}_%j.err" \
        --export=ALL,BATCH_NAME="${name}",EST_MIN="${est_min}",TASKS="${tasks}" \
        ./scripts/bash_script/rebuttal/repeated_asr/run_harmbench_model_attack.sh
}

submit_batch asr_24h_01 23:50:00 1323 "llama3 gcg 2 None 1;llama3 gcg 2 None 2;qwen25 gcg 2 None 1;qwen25 gcg 2 None 2;vicuna gcq 1 None 1"
submit_batch asr_24h_02 23:50:00 975 "vicuna gcq 1 None 2;qwen25 gcq 2 None 1;qwen25 gcq 2 None 2;mistral gcq 1 Ours 1;mistral gcq 1 Ours 2"

submit_batch asr_24h_03 23:50:00 1340 "vicuna gcg 1 None 1;vicuna gcg 1 None 2;llama2 gcg 1 None 1;llama3 gcq 2 None 1;llama2 gcg 1 None 2;llama3 gcq 2 None 2"

submit_batch asr_24h_04 23:50:00 1358 "mistral gcg 1 None 1;vicuna beast 1 Ours 1;mistral gcg 1 None 2;qwen25 beast 2 Ours 2;llama2 gcq 1 None 1;llama2 beast 1 Ours 1;llama2 gcq 1 None 2;llama3 beast 2 Ours 1;mistral gcq 1 None 1;llama2 beast 1 None 1;mistral gcq 1 None 2;qwen25 beast 2 None 1"

submit_batch asr_24h_05 23:50:00 1360 "vicuna gcq 1 Ours 1;mistral autodan-zhu 1 None 1;vicuna gcq 1 Ours 2;mistral autodan-zhu 1 None 2;vicuna gcg 1 Ours 1;qwen25 autodan-zhu 2 None 1;vicuna gcg 1 Ours 2;qwen25 autodan-zhu 2 None 2;llama2 gcq 1 Ours 1;vicuna autodan-zhu 1 Ours 1;llama2 beast 1 None 2;llama2 gcq 1 Ours 2;vicuna autodan-zhu 1 Ours 2;qwen25 beast 2 None 2"

submit_batch asr_24h_06 23:50:00 1366 "llama2 gcg 1 Ours 1;llama3 autodan-zhu 2 None 1;llama2 gcg 1 Ours 2;vicuna autodan-zhu 1 None 1;llama3 gcg 2 Ours 1;llama3 autodan-zhu 2 None 2;llama3 gcg 2 Ours 2;vicuna autodan-zhu 1 None 2;llama3 gcq 2 Ours 1;llama2 autodan-zhu 1 None 1;llama3 gcq 2 Ours 2;llama2 autodan-zhu 1 None 2"

submit_batch asr_24h_07 23:50:00 1368 "qwen25 gcg 2 Ours 1;llama2 autodan-zhu 1 Ours 1;llama3 autodan-zhu 2 Ours 1;qwen25 gcg 2 Ours 2;llama2 autodan-zhu 1 Ours 2;llama3 autodan-zhu 2 Ours 2;mistral gcg 1 Ours 1;qwen25 autodan-zhu 2 Ours 1;qwen25 autodan-zhu 2 Ours 2;qwen25 beast 2 Ours 1;mistral gcg 1 Ours 2;mistral autodan-zhu 1 Ours 1;mistral autodan-zhu 1 Ours 2;vicuna beast 1 Ours 2;qwen25 gcq 2 Ours 1;mistral beast 1 None 1;mistral beast 1 None 2;llama3 beast 2 None 1;llama3 beast 2 None 2;qwen25 gcq 2 Ours 2;vicuna beast 1 None 1;vicuna beast 1 None 2;mistral beast 1 Ours 1;llama2 beast 1 Ours 2;llama3 beast 2 Ours 2;mistral beast 1 Ours 2"
