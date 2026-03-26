#!/usr/bin/env python3
"""Collect and summarize multi-seed ASR results for the rebuttal."""
import os, json, glob, re
import numpy as np

RESULTS_ROOT = "../results"

EXPERIMENTS = [
    ("llama2", "gcg", "1"),
    ("llama2", "beast", "1"),
    ("llama3", "beast", "2"),
    ("vicuna", "gcq", "1"),
    ("vicuna", "beast", "1"),
]

SEEDS = [0, 1, 2, 3, 4]
KV_MODES = ["Ours", "None"]
DATASET = "harmbench"


def get_asr_from_log(log_path):
    """Extract ASR from judge log file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        content = f.read()
    match = re.search(r"Overall attack success rate:\s*([\d.]+)%?", content)
    if match:
        val = float(match.group(1))
        return val if val <= 1.0 else val / 100.0
    match = re.search(r"attack success rate:\s*([\d.]+)", content)
    if match:
        val = float(match.group(1))
        return val if val <= 1.0 else val / 100.0
    return None


def main():
    print("=" * 80)
    print("Multi-Seed ASR Results Summary")
    print("=" * 80)
    print(f"{'Model':<12} {'Attack':<10} {'KV Cache':<10} {'Seeds Found':<12} {'ASR Mean':<10} {'ASR Std':<10} {'ASR Values'}")
    print("-" * 80)

    for model, attack, cfg in EXPERIMENTS:
        for kv in KV_MODES:
            asrs = []
            for seed in SEEDS:
                seed_suffix = f"_seed{seed}"
                log_dir = os.path.join(
                    RESULTS_ROOT, model, f"eval-{attack}", f"cfg-{cfg}",
                    f"kv-cache-{kv}{seed_suffix}"
                )
                log_path = os.path.join(log_dir, f"judge-{DATASET}_log.txt")
                asr = get_asr_from_log(log_path)
                if asr is not None:
                    asrs.append(asr)

            if asrs:
                mean_asr = np.mean(asrs)
                std_asr = np.std(asrs)
                vals_str = ", ".join(f"{v:.2%}" for v in asrs)
                print(f"{model:<12} {attack:<10} {kv:<10} {len(asrs):<12} {mean_asr:<10.2%} {std_asr:<10.2%} [{vals_str}]")
            else:
                print(f"{model:<12} {attack:<10} {kv:<10} {'0':<12} {'N/A':<10} {'N/A':<10} []")

    # AdvPrompter results (from train logs)
    print("\n--- AdvPrompter (Llama3-8B) ---")
    for kv in KV_MODES:
        asrs = []
        for seed in SEEDS:
            seed_suffix = f"_seed{seed}"
            log_dir = os.path.join(
                RESULTS_ROOT, "llama3", "train-adv-prompter", "cfg-2",
                f"kv-cache-{kv}{seed_suffix}"
            )
            log_path = os.path.join(log_dir, "attack_train_log.txt")
            asr = get_asr_from_log(log_path)
            if asr is not None:
                asrs.append(asr)

        if asrs:
            mean_asr = np.mean(asrs)
            std_asr = np.std(asrs)
            vals_str = ", ".join(f"{v:.2%}" for v in asrs)
            print(f"{'llama3':<12} {'advprompter':<10} {kv:<10} {len(asrs):<12} {mean_asr:<10.2%} {std_asr:<10.2%} [{vals_str}]")
        else:
            print(f"{'llama3':<12} {'advprompter':<10} {kv:<10} {'0':<12} {'N/A':<10} {'N/A':<10} []")

    print("=" * 80)


if __name__ == "__main__":
    main()
