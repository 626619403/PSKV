#!/bin/bash
#SBATCH --job-name=test_deploy
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=../logs/test_deploy_%j.out
#SBATCH --error=../logs/test_deploy_%j.err

source ~/.bashrc
module load cuda/12.4.1
conda activate attack

echo "=== STARTING DEPLOYMENT TEST ==="
mkdir -p ../logs/test_deploy

# 1. Test Standard Attacks (AdvPrompter, GCG, Autodan, BEAST, GCQ)
# Using lighter settings if possible, otherwise standard.
# We use existing eval scripts but override dataset to something small if possible or just run first part.
# Since we can't easily chop dataset, we run the script and kill it? 
# No, user said "run once(on only one dataset...)". 
# I will use harmbench-test50 (50 examples) with llama2.

MODEL_CONFIG="1" # llama2 config
DATASET="harmbench-test50" 

echo "--- Testing AdvPrompter ---"
echo "--- Testing AdvPrompter ---"
bash ./train/llama2.sh --src-path ../src --train-cfg $MODEL_CONFIG --dataset $DATASET --attack adv-prompter --kv-cache Normal
echo "AdvPrompter Test Done"

echo "--- Testing AmpleGCG ---"
# AmpleGCG usually requires training first? Or uses trained prompter.
# If no prompter, it might fail. We skip if assumed training needed, but user said "all attacks".
# I'll try running it.
bash ./train/llama2.sh --src-path ../src --train-cfg $MODEL_CONFIG --dataset $DATASET --attack ample-gcg --kv-cache Normal
echo "AmpleGCG Test Done"

echo "--- Testing GCG ---"
bash ./eval/llama2.sh --src-path ../src --eval-cfg $MODEL_CONFIG --dataset $DATASET --attack gcg --kv-cache Normal
echo "GCG Test Done"

echo "--- Testing AutoDAN ---"
bash ./eval/llama2.sh --src-path ../src --eval-cfg $MODEL_CONFIG --dataset $DATASET --attack autodan --kv-cache Normal
echo "AutoDAN Test Done"

echo "--- Testing BEAST (Standard) ---"
bash ./eval/llama2.sh --src-path ../src --eval-cfg $MODEL_CONFIG --dataset $DATASET --attack beast --kv-cache Normal
echo "BEAST Standard Test Done"

echo "--- Testing GCQ (Standard) ---"
bash ./eval/llama2.sh --src-path ../src --eval-cfg $MODEL_CONFIG --dataset $DATASET --attack gcq --kv-cache Normal
echo "GCQ Standard Test Done"


# 2. Test New Baselines
# Models path on ibex: Need to know where models are. 
# Existing scripts use 'meta-llama/Llama-2-7b-chat-hf', I assume it's cached or available.
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"

echo "--- Testing BEAST (vLLM) ---"
python ../src/baseline/beast_vllm.py --model-path $MODEL_PATH --suffix-length 10 --beam-size 2 --search-width 2
echo "BEAST vLLM Test Done"

echo "--- Testing GCQ (vLLM) ---"
python ../src/baseline/gcq_vllm.py --model-path $MODEL_PATH --suffix-length 10 --num-steps 5
echo "GCQ vLLM Test Done"

# SGLang requires server. We try to launch it in background?
# Or we skip sglang for this simple batch test if launching server is complex.
# User asked to test it. I will try to launch a server.
echo "--- Testing SGLang Baselines ---"
echo "Launching SGLang server..."
python -m sglang.launch_server --model-path $MODEL_PATH --port 30000 &
SERVER_PID=$!
sleep 60 # wait for server

echo "Running BEAST (SGLang)..."
python ../src/baseline/beast_sglang.py --model-path $MODEL_PATH --port 30000 --suffix-length 10
echo "Running GCQ (SGLang)..."
python ../src/baseline/gcq_sglang.py --model-path $MODEL_PATH --port 30000 --suffix-length 10

kill $SERVER_PID
echo "SGLang Tests Done"

echo "=== ALL TESTS COMPLETED ==="
