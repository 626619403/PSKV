#!/bin/bash
#SBATCH --job-name=gcq_ours
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=../logs/gcq/ours/run_%j.out
#SBATCH --error=../logs/gcq/ours/run_%j.err

# kv_cache=Ours, attack=gcq
source ~/.bashrc  
conda activate attack
echo "=========================================="
echo "Starting experiments with attack=gcq, kv_cache=Ours"
echo "Start time: $(date)"
echo "=========================================="

# Dataset 1: harmbench-test50
echo "Dataset: harmbench-test50"

echo "[1/10] Running llama2 with harmbench-test50, kv_cache=Ours"
bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "[2/10] Running llama3 with harmbench-test50, kv_cache=Ours"
bash ./eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "[3/10] Running mistral with harmbench-test50, kv_cache=Ours"
bash ./eval/mistral.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "[4/10] Running qwen25 with harmbench-test50, kv_cache=Ours"
bash ./eval/qwen25.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "[5/10] Running vicuna with harmbench-test50, kv_cache=Ours"
bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

# Dataset 2: advbench-first50
echo "Dataset: advbench-first50"

echo "[6/10] Running llama2 with advbench-first50, kv_cache=Ours"
bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset advbench-first50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "[7/10] Running llama3 with advbench-first50, kv_cache=Ours"
bash ./eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset advbench-first50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "[8/10] Running mistral with advbench-first50, kv_cache=Ours"
bash ./eval/mistral.sh --src-path ../src --eval-cfg 1 --dataset advbench-first50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "[9/10] Running qwen25 with advbench-first50, kv_cache=Ours"
bash ./eval/qwen25.sh --src-path ../src --eval-cfg 2 --dataset advbench-first50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "[10/10] Running vicuna with advbench-first50, kv_cache=Ours"
bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset advbench-first50 --attack gcq --kv-cache Ours
echo "Completed at: $(date)"

echo "=========================================="
echo "All experiments with attack=gcq, kv_cache=Ours completed!"
echo "End time: $(date)"
echo "=========================================="

exit