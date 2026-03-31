# Accelerating Suffix Jailbreak Attacks with Prefix-Shared KV-cache

This repository contains the source code and experimental setup for the paper *"Accelerating Suffix Jailbreak Attacks with Prefix-Shared KV-cache"*. It provides the necessary tools to reproduce our findings on adversarial attacks against large language models.

## For reviewer

All of memory snapshots and layer-by-layer analysis results are under "memory_snapshot" folder. All of scripts used to run these experiment are under script/bash_script/rebuttal folder. 

## 1. Setup

### Installation with Pip

```bash
# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio
pip install peft==0.14.0 safetensors==0.4.5 datasets==3.2.0 accelerate==1.2.1 \
    protobuf==5.29.1 sentencepiece==0.2.0 bitsandbytes==0.45.0 alpaca-eval==0.6.6
```

> **Note:** Ensure you have a compatible version of CUDA installed for GPU support.

### Optional: vLLM (for `beast-vllm` baseline)

The `beast-vllm` baseline requires [vLLM](https://vllm.ai) to be installed separately:

```bash
pip install vllm
```

> If you encounter errors related to `common_ops`, please refer to the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html).

### Optional: SGLang (for `beast-sglang` baseline)

The `beast-sglang` baseline requires [SGLang](https://sglang.readthedocs.io):

```bash
pip install sglang==0.4.9
```

---

## 2. Repository Structure

```
.
├── configs/                # Configuration files for training and evaluation
│   ├── train/
│   └── eval/
│       ├── beast/          # BEAST attack configs
│       ├── beast_vllm/     # BEAST-vLLM baseline configs
│       ├── beast_sglang/   # BEAST-SGLang baseline configs
│       ├── gcg/
│       ├── gcq/
│       ├── ample-gcg/
│       └── autodan-zhu/
├── data/                   # Datasets
│   ├── advbench/
│   └── harmbench/
├── results/                # Output directory for trained models and logs
├── scripts/                # Shell scripts to run experiments
│   ├── train/
│   └── eval/
├── src/                    # Source code
│   ├── attacks/            # Implementation of attack methods
│   │   ├── beast.py        # BEAST (standard, HuggingFace)
│   │   ├── beast_vllm.py   # BEAST accelerated with vLLM
│   │   ├── beast_sglang.py # BEAST accelerated with SGLang
│   │   ├── gcg.py
│   │   ├── gcq.py
│   │   ├── AmpleGCG.py
│   │   ├── autodan_zhu.py
│   │   └── advPrompter.py
│   ├── utils/              # Utility functions
│   ├── train.py            # Main training script
│   └── evaluate.py         # Main evaluation script
├── Dockerfile              # Dockerfile for environment setup
└── README.md               # This file
```

---

## 3. Data

This project uses the [AdvBench](https://github.com/llm-attacks/llm-attacks) and [HarmBench](https://github.com/centerforaisafety/HarmBench) datasets. Please download them and place them into the `data/advbench` and `data/harmbench` directories respectively.

---

## 4. Running Experiments

Experiments are run using the `src/train.py` and `src/evaluate.py` scripts. We provide configuration files and helper scripts in the `configs/` and `scripts/` directories.

Please refer to [scripts/README.md](./scripts/README.md) for a detailed tutorial on running all experiments.

### Training

Use the shell scripts located in `scripts/train/` to run adversarial training. Supported attack methods for training are `ample-gcg` and `adv-prompter`.

**Example:**

```bash
cd scripts

# AmpleGCG training on Llama-2-7b with our KV-cache
bash train/llama2.sh \
    --src-path ../src \
    --train-cfg 1 \
    --dataset harmbench \
    --attack ample-gcg \
    --kv-cache Ours
```

**Arguments:**

| Argument | Description | Valid Values |
|---|---|---|
| `--src-path` | Path to the `src/` directory | e.g., `../src` |
| `--train-cfg` | Config ID from `configs/train/{attack}/cfg-<ID>.yaml` | e.g., `1`, `2` |
| `--dataset` | Dataset to use | `harmbench`, `advbench` |
| `--attack` | Attack method | `ample-gcg`, `adv-prompter` |
| `--kv-cache` | KV-cache strategy | `None`, `Normal`, `Ours` |

### Evaluation

Use the shell scripts in `scripts/eval/` to run jailbreak evaluation. Supported attack methods include the two new **vLLM** and **SGLang** accelerated baselines.

**Example:**

```bash
cd scripts

# Evaluate Llama-3-8B with standard BEAST
bash eval/llama3.sh \
    --src-path ../src \
    --eval-cfg 1 \
    --dataset harmbench-test50 \
    --attack beast \
    --kv-cache Ours

# Evaluate with BEAST-vLLM baseline
bash eval/llama3.sh \
    --src-path ../src \
    --eval-cfg 1 \
    --dataset harmbench-test50 \
    --attack beast-vllm \
    --kv-cache None

# Evaluate with BEAST-SGLang baseline
bash eval/llama3.sh \
    --src-path ../src \
    --eval-cfg 1 \
    --dataset harmbench-test50 \
    --attack beast-sglang \
    --kv-cache None
```

**Arguments:**

| Argument | Description | Valid Values |
|---|---|---|
| `--src-path` | Path to the `src/` directory | e.g., `../src` |
| `--eval-cfg` | Config ID from `configs/eval/{attack}/cfg-<ID>.yaml` | e.g., `1`, `2` |
| `--dataset` | Evaluation dataset | `harmbench-test50`, `advbench-first50` |
| `--attack` | Attack method | `gcg`, `beast`, `beast-vllm`, `beast-sglang`, `autodan-zhu`, `ample-gcg`, `gcq` |
| `--kv-cache` | KV-cache strategy | `None`, `Normal`, `Ours` |

---

## 5. New Baselines: BEAST-vLLM and BEAST-SGLang

We introduce two new inference-accelerated baselines for the [BEAST attack](https://arxiv.org/pdf/2402.15570) that replace the standard HuggingFace model forward pass with high-throughput inference engines. Both baselines leverage **prefix KV-cache sharing** natively provided by the inference engine to speed up the beam search process.

### BEAST-vLLM (`beast-vllm`)

- **Implementation:** [`src/attacks/beast_vllm.py`](./src/attacks/beast_vllm.py)
- **Config directory:** [`configs/eval/beast_vllm/`](./configs/eval/beast_vllm/)
- **Backend:** [vLLM](https://vllm.ai) with `enable_prefix_caching=True`
- **Key parameters** (set in `configs/eval/beast_vllm/cfg-*.yaml`):
  - `suffix_length`: Length of the adversarial suffix to generate.
  - `beam_size`: Number of beams in beam search.
  - `search_width`: Number of candidate tokens expanded per beam per step.
  - `gpu_memory_utilization`: Fraction of GPU memory for vLLM (default: `0.9`).
  - `tensor_parallel_size`: Number of GPUs for tensor parallelism (default: `1`).

### BEAST-SGLang (`beast-sglang`)

- **Implementation:** [`src/attacks/beast_sglang.py`](./src/attacks/beast_sglang.py)
- **Config directory:** [`configs/eval/beast_sglang/`](./configs/eval/beast_sglang/)
- **Backend:** [SGLang](https://sglang.readthedocs.io) `Engine` with `RadixAttention` for automatic KV-cache reuse.
- **Key parameters** (set in `configs/eval/beast_sglang/cfg-*.yaml`):
  - `suffix_length`: Length of the adversarial suffix to generate.
  - `beam_size`: Number of beams in beam search.
  - `search_width`: Number of candidate tokens expanded per beam per step.
  - `gpu_memory_utilization`: Fraction of GPU memory for SGLang (default: `0.9`).
  - `tp_size`: Tensor parallel size (default: `1`).

> **Note:** `beast-vllm` and `beast-sglang` accept a HuggingFace model ID or a local model path directly. The inference engine is initialized internally — no separate model loading is required before calling the eval script.

---

## 6. License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
