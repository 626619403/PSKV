# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for the paper *"Accelerating Suffix Jailbreak Attacks with Prefix-Shared KV-cache"* (PSKV). The core contribution is a custom KV-cache (`src/utils/kv_cache.py`) that shares prefix computations across adversarial suffix candidates to speed up LLM jailbreak attacks.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio
pip install peft==0.14.0 safetensors==0.4.5 datasets==3.2.0 accelerate==1.2.1 \
    protobuf==5.29.1 sentencepiece==0.2.0 bitsandbytes==0.45.0 alpaca-eval==0.6.6

# Optional baselines
pip install vllm          # for beast-vllm
pip install sglang==0.4.9 # for beast-sglang
```

## Running Experiments

All experiment scripts are run from the `scripts/` directory:

```bash
cd scripts

# Evaluation (two-stage: build attack outputs, then judge ASR)
bash eval/llama3.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack gcg --kv-cache Ours
bash eval/llama2.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack beast --kv-cache None

# Adversarial training
bash train/llama2.sh --src-path ../src --train-cfg 1 --dataset harmbench --attack ample-gcg --kv-cache Ours
```

**Key argument values:**
- `--attack`: `gcg`, `beast`, `beast-vllm`, `beast-sglang`, `autodan-zhu`, `ample-gcg`, `gcq`, `adv-prompter`
- `--kv-cache`: `None`, `Normal`, `Ours` (use `None` for `beast-vllm`/`beast-sglang`)
- `--dataset` (eval): `harmbench-test50`, `advbench-first50`, `wildjailbreak-50`
- `--dataset` (train): `harmbench`, `advbench`

### Running directly via Python

The eval pipeline has two stages that map to `--exp-type`:

```bash
# Stage 1: generate adversarial responses
python src/evaluate.py --exp-type build-evalset --model-id <hf_model_id> \
    --atker-cfg-path configs/eval/gcg/cfg-1.yaml --dataset harmbench-test50 \
    --kv-cache Ours --save-dir results/... --save-name ...

# Stage 2: judge responses with HarmBench classifier
python src/evaluate.py --exp-type judge-evalset \
    --judger-cfg-path configs/eval/judge.yaml \
    --evalset-path results/.../<name>_evalset.json \
    --save-dir results/... --save-name ...
```

## Architecture

### Core Pipeline

```
configs/eval/{attack}/cfg-N.yaml  →  src/evaluate.py  →  results/
configs/train/{attack}/cfg-N.yaml →  src/train.py     →  results/
```

`evaluate.py` orchestrates: load model → run attack (`build-evalset`) → judge with HarmBench LLM classifier (`judge-evalset`).

### Key Modules

- **`src/attacks/`** — One file per attack: `gcg.py`, `beast.py`, `beast_vllm.py`, `beast_sglang.py`, `gcq.py`, `AmpleGCG.py`, `autodan_zhu.py`, `advPrompter.py`
- **`src/utils/kv_cache.py`** — The PSKV contribution. `BaseCache`/`BaseCacheLayer` extend HuggingFace `DynamicCache`/`DynamicLayer`. KV-cache mode is normalized via `normalize_cache_mode()` to one of `{"ours", "normal", "none"}`.
- **`src/utils/argument.py`** — CLI argument definitions (`add_shared_args`, `add_eval_args`, `add_attack_train_args`)
- **`src/utils/datasets.py`** — Dataset loading and datacollators (`vicuna-chat`, `llama2-chat`, `llama3-chat`, `qwen2-chat`)
- **`src/evaltools/`** — `EvalsetConfig`, `JudgerConfig`, `build_evalset`, `build_judger`, `AlpacaEvalConfig`
- **`src/utils/monitor.py`** — Resource monitoring context manager (`utils.resource_monitor`)

### Configuration System

YAML configs are loaded and passed as dataclass constructors. Attack configs (e.g., `configs/eval/gcg/cfg-1.yaml`) set hyperparameters like `suffix_length`, `steps`, `topk`, `search_width`, `batch_size`. The `kv_cache` field is injected at runtime from the CLI argument.

Results are saved to `results/{model}/{eval-or-train-attack}/{cfg-name}/kv-cache-{mode}/`.

### BEAST Variants

- **`beast.py`**: Standard HuggingFace beam search with PSKV's custom KV-cache
- **`beast_vllm.py`**: Replaces HF forward pass with vLLM (`enable_prefix_caching=True`); no HF model loaded
- **`beast_sglang.py`**: Replaces HF forward pass with SGLang `Engine` (`RadixAttention`); no HF model loaded
