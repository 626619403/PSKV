#!/bin/bash
#SBATCH --job-name=long_context
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=../logs/rebuttal/long_context_%j.out
#SBATCH --error=../logs/rebuttal/long_context_%j.err

source ~/.bashrc
conda activate attack

echo "=========================================="
echo "Long-context experiments for rebuttal"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================
# Part 1: Longer suffix lengths with standard HarmBench dataset
# Test GCG with suffix_length=40 and suffix_length=60
# ============================================================
echo "=== Part 1: Longer Suffix Lengths (GCG on HarmBench) ==="

# GCG suffix=40 on Llama2
for kv in Ours None; do
    echo "[Llama2-GCG-sfx40-${kv}]"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg long-sfx40 --dataset harmbench-test50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

# GCG suffix=60 on Llama2
for kv in Ours None; do
    echo "[Llama2-GCG-sfx60-${kv}]"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg long-sfx60 --dataset harmbench-test50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

# GCG suffix=40 on Llama3
for kv in Ours None; do
    echo "[Llama3-GCG-sfx40-${kv}]"
    bash ./eval/llama3.sh --src-path ../src --eval-cfg long-sfx40 --dataset harmbench-test50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

# GCG suffix=60 on Llama3
for kv in Ours None; do
    echo "[Llama3-GCG-sfx60-${kv}]"
    bash ./eval/llama3.sh --src-path ../src --eval-cfg long-sfx60 --dataset harmbench-test50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

# BEAST suffix=40 on Llama2
for kv in Ours None; do
    echo "[Llama2-BEAST-sfx40-${kv}]"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg long-sfx40 --dataset harmbench-test50 --attack beast --kv-cache ${kv}
    echo "Completed at: $(date)"
done

# ============================================================
# Part 2: Longer prompts using WildJailbreak dataset
# Test GCG and BEAST with standard suffix on longer prompts
# ============================================================
echo "=== Part 2: Longer Prompts (WildJailbreak) ==="

# GCG on WildJailbreak with Llama2
for kv in Ours None; do
    echo "[Llama2-GCG-WildJB-${kv}]"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset wildjailbreak-50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

# GCG on WildJailbreak with Llama3
for kv in Ours None; do
    echo "[Llama3-GCG-WildJB-${kv}]"
    bash ./eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset wildjailbreak-50 --attack gcg --kv-cache ${kv}
    echo "Completed at: $(date)"
done

# BEAST on WildJailbreak with Llama2
for kv in Ours None; do
    echo "[Llama2-BEAST-WildJB-${kv}]"
    bash ./eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset wildjailbreak-50 --attack beast --kv-cache ${kv}
    echo "Completed at: $(date)"
done

echo "=========================================="
echo "All long-context experiments completed!"
echo "End time: $(date)"
echo "=========================================="

exit
