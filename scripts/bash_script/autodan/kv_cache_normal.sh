#!/bin/bash
#SBATCH --job-name=autodan_normal
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=../logs/autodan/normal/run_%j.out
#SBATCH --error=../logs/autodan/normal/run_%j.err

source ~/.bashrc  
conda activate attack
echo "=========================================="
echo "Starting experiments with attack=autodan-zhu, kv_cache=Normal"
echo "Start time: $(date)"
echo "=========================================="

# Dataset 1: harmbench-test50
echo "Dataset: harmbench-test50"

echo "[1/10] Running llama2 with harmbench-test50, kv_cache=Normal"
bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "[2/10] Running llama3 with harmbench-test50, kv_cache=Normal"
bash ./eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "[3/10] Running mistral with harmbench-test50, kv_cache=Normal"
bash ./eval/mistral.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "[4/10] Running qwen25 with harmbench-test50, kv_cache=Normal"
bash ./eval/qwen25.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "[5/10] Running vicuna with harmbench-test50, kv_cache=Normal"
bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

# Dataset 2: advbench-first50
echo "Dataset: advbench-first50"

echo "[6/10] Running llama2 with advbench-first50, kv_cache=Normal"
bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset advbench-first50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "[7/10] Running llama3 with advbench-first50, kv_cache=Normal"
bash ./eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset advbench-first50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "[8/10] Running mistral with advbench-first50, kv_cache=Normal"
bash ./eval/mistral.sh --src-path ../src --eval-cfg 1 --dataset advbench-first50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "[9/10] Running qwen25 with advbench-first50, kv_cache=Normal"
bash ./eval/qwen25.sh --src-path ../src --eval-cfg 2 --dataset advbench-first50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "[10/10] Running vicuna with advbench-first50, kv_cache=Normal"
bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset advbench-first50 --attack autodan-zhu --kv-cache Normal
echo "Completed at: $(date)"

echo "=========================================="
echo "All experiments with attack=autodan-zhu, kv_cache=Normal completed!"
echo "End time: $(date)"
echo "=========================================="

exit