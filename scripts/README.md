
# Tutorial on Running Scripts

[Scripts](./) can be divided into 2 categories:

- [scripts/train](./train) stores all scripts for adversarial training with **AmpleGCG** and **Adv-Prompter**.
- [scripts/eval](./eval) stores all scripts for robust evaluation with **GCG**, **BEAST**, **BEAST-vLLM**, **BEAST-SGLang**, **AutoDAN-Zhu**, **AmpleGCG** and **GCQ** attacks.

Here we present a brief tutorial on how to run experiments with these scripts.

**Step 0:** Enter the script folder:

```bash
cd ./scripts
```

---

## Option 1: Adversarial Training (with automatic evaluation)

**Perform adversarial training** (includes automatic evaluation after training):

```bash
# AmpleGCG training
bash train/vicuna.sh --src-path ../src --train-cfg 1 --dataset harmbench --attack ample-gcg --kv-cache Ours

# Adv-Prompter training
bash train/vicuna.sh --src-path ../src --train-cfg 2 --dataset harmbench --attack adv-prompter --kv-cache None
```

where:
- `--train-cfg` should exist in `../configs/train/{attack_method}`
- `--dataset` should be either `harmbench` or `advbench`
- `--attack` should be either `ample-gcg` or `adv-prompter`
- `--kv-cache` should be one of `None`, `Normal`, or `Ours`

**Note:** `adv-prompter` training scripts automatically evaluate the trained model during the training period.

---

## Option 2: Direct Evaluation with Jailbreak Attacks

**Generate harmful responses via jailbreak attacks** and calculate attack success rate (ASR) based on induced harmful responses with the [LLM judger from Harmbench](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls).

```bash
# Evaluate vanilla pre-trained model with GCG
bash eval/llama3.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack gcg --kv-cache Ours

# Evaluate adversarially trained model with AmpleGCG
bash eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack ample-gcg --kv-cache Normal

# Evaluate with standard BEAST
bash eval/llama3.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack beast --kv-cache Ours

# Evaluate with BEAST-vLLM baseline (requires vllm installed)
bash eval/llama3.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack beast-vllm --kv-cache None

# Evaluate with BEAST-SGLang baseline (requires sglang installed)
bash eval/llama3.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack beast-sglang --kv-cache None
```

where:

- `--eval-cfg` should exist in `../configs/eval/{attack_method}`
- `--dataset` should be either `harmbench-test50` or `advbench-first50`
- `--attack` should be one of `gcg`, `beast`, `beast-vllm`, `beast-sglang`, `autodan-zhu`, `ample-gcg`, or `gcq`
- `--kv-cache` should be one of `None`, `Normal`, or `Ours`

> **Note:** `beast-vllm` and `beast-sglang` use inference engines that manage KV-cache internally. It is recommended to set `--kv-cache None` for these two baselines.

**Note:** Evaluation scripts automatically:
1. Build the evaluation set with the specified attack
2. Judge the generated responses using the Harmbench classifier

---

## New Baselines: BEAST-vLLM and BEAST-SGLang

Two new inference-accelerated baselines are available for the BEAST attack:

| Baseline | `--attack` flag | Backend | KV-cache mechanism | Install |
|---|---|---|---|---|
| BEAST-vLLM | `beast-vllm` | [vLLM](https://vllm.ai) | `enable_prefix_caching=True` | `pip install vllm` |
| BEAST-SGLang | `beast-sglang` | [SGLang](https://sglang.readthedocs.io) | `RadixAttention` | `pip install sglang==0.4.9` |

Config files are located in `../configs/eval/beast_vllm/` and `../configs/eval/beast_sglang/` respectively.

