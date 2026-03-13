#!/bin/bash
#SBATCH --job-name=amplegcg_none_advbench
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=../logs/amplegcg/none/advbench.out
#SBATCH --error=../logs/amplegcg/none/advbench.err

source ~/.bashrc  
conda activate attack
module load cuda/12.4.1
echo "=========================================="
echo "Starting: attack=ample-gcg, kv_cache=None, dataset=advbench-first50"
echo "Start time: $(date)"
echo "=========================================="

echo "[1/5] Running llama2"
bash ./train/llama2.sh --src-path ../src --train-cfg  1 --dataset advbench --attack ample-gcg --kv-cache None
echo "Completed at: $(date)"

echo "[2/5] Running llama3"
bash ./train/llama3.sh --src-path ../src --train-cfg  2 --dataset advbench --attack ample-gcg --kv-cache None
echo "Completed at: $(date)"

echo "[3/5] Running mistral"
bash ./train/mistral.sh --src-path ../src --train-cfg  1 --dataset advbench --attack ample-gcg --kv-cache None
echo "Completed at: $(date)"

echo "[4/5] Running qwen25"
bash ./train/qwen25.sh --src-path ../src --train-cfg  2 --dataset advbench --attack ample-gcg --kv-cache None
echo "Completed at: $(date)"

echo "[5/5] Running vicuna"
bash ./train/vicuna.sh --src-path ../src --train-cfg  1 --dataset advbench --attack ample-gcg --kv-cache None
echo "Completed at: $(date)"

echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "=========================================="

exit
