#!/bin/bash
#SBATCH --job-name=baseline_sglang
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=../logs/baseline/sglang/run_%j.out
#SBATCH --error=../logs/baseline/sglang/run_%j.err

#INSTALL SGLANG DOCKER AND CHANGE ALL "LAUNCHER" in "./eval/*.sh" INTO PYTHON3 FIRST!!!!!

source ~/.bashrc  
conda activate attack
echo "[1/10] Running llama2 with harmbench-test50"
bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack beast_sglang --kv-cache None
echo "Completed at: $(date)"

echo "[2/10] Running llama3 with harmbench-test50"
bash ./eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack beast_sglang --kv-cache None
echo "Completed at: $(date)"

echo "[3/10] Running mistral with harmbench-test50"
bash ./eval/mistral.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack beast_sglang --kv-cache None
echo "Completed at: $(date)"

echo "[4/10] Running qwen25 with harmbench-test50"
bash ./eval/qwen25.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack beast_sglang --kv-cache None
echo "Completed at: $(date)"

echo "[5/10] Running vicuna with harmbench-test50"
bash ./eval/vicuna.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack beast_sglang --kv-cache None
echo "Completed at: $(date)"