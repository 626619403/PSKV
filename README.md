# Accelerating Suffix Jailbreak Attacks with Prefix-Shared KV-cache

This repository contains the code, configurations, and supplementary scripts for the paper [Accelerating Suffix Jailbreak Attacks with Prefix-Shared KV-cache](https://arxiv.org/abs/2603.13420). It supports reproducing the main jailbreak attack experiments and the supplementary memory/logit-equivalence analyses.

## Setup

Create an environment and install the pinned dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The Dockerfile uses the same `requirements.txt`:

```bash
docker build -t pskv .
```

Optional backends:

- `beast_vllm` / `beast-vllm` requires vLLM. Install it separately if you run that baseline.
- `beast_sglang` / `beast-sglang` uses `sglang==0.4.9`, which is included in `requirements.txt`.

The experiments require CUDA-capable GPUs and access to the Hugging Face model weights used by the scripts.

## Data

The repository expects the AdvBench and HarmBench data under:

```text
data/advbench/
data/harmbench/
```

The included data files mirror the original public datasets. If recreating from scratch, download AdvBench from `llm-attacks/llm-attacks` and HarmBench from `centerforaisafety/HarmBench`, then place the files in the paths above.

## Repository Structure

```text
configs/                  Training and evaluation YAML configs
data/                     AdvBench and HarmBench data files
memory_snapshot/          Supplementary memory/profile artifacts
results/                  Ignored output directory; selected Markdown summaries may be tracked
scripts/eval/             Model-specific evaluation launchers
scripts/train/            Model-specific training launchers
scripts/bash_script/      Batch scripts for full experiments and supplementary analyses
src/attacks/              Attack implementations
src/evaltools/            Evaluation set and judging utilities
src/utils/                Model, dataset, KV-cache, and logging utilities
src/test/                 Auxiliary profiling and correctness scripts
```

## Evaluation

Run evaluation scripts from `scripts/`. Attack names with hyphens and underscores are both accepted for the vLLM/SGLang BEAST baselines; outputs use the underscore form in result paths.

```bash
cd scripts

# Standard GCG evaluation
bash eval/llama3.sh \
    --src-path ../src \
    --eval-cfg 2 \
    --dataset harmbench-test50 \
    --attack gcg \
    --kv-cache Ours \
    --random-seed 1

# Standard BEAST evaluation
bash eval/llama3.sh \
    --src-path ../src \
    --eval-cfg 2 \
    --dataset harmbench-test50 \
    --attack beast \
    --kv-cache Ours

# BEAST with vLLM
bash eval/llama3.sh \
    --src-path ../src \
    --eval-cfg 2 \
    --dataset harmbench-test50 \
    --attack beast-vllm \
    --kv-cache None

# BEAST with SGLang
bash eval/llama3.sh \
    --src-path ../src \
    --eval-cfg 2 \
    --dataset harmbench-test50 \
    --attack beast-sglang \
    --kv-cache None
```

Supported evaluation arguments:

| Argument | Description |
|---|---|
| `--src-path` | Path to `src/`, usually `../src` from `scripts/` |
| `--eval-cfg` | Config ID from `configs/eval/<attack>/cfg-<ID>.yaml` |
| `--dataset` | `harmbench-test50`, `advbench-first50`, or `wildjailbreak-50` where supported |
| `--attack` | `gcg`, `gcq`, `beast`, `beast-vllm`, `beast-sglang`, `autodan-zhu`, `ample-gcg`, or `adv-prompter` |
| `--kv-cache` | `None`, `Normal`, or `Ours` |
| `--random-seed` | Optional seed; result directory gets `_seed<seed>` suffix |

Each evaluation script first builds an evaluation set and then judges it with the HarmBench classifier. Results are written to:

```text
results/<model>/eval-<attack>/cfg-<cfg>/kv-cache-<None|Normal|Ours>[_seedN]/
```

## Training

Run adversarial training scripts from `scripts/`:

```bash
cd scripts

bash train/llama2.sh \
    --src-path ../src \
    --train-cfg 1 \
    --dataset harmbench \
    --attack ample-gcg \
    --kv-cache Ours
```

Supported training attacks are `ample-gcg` and `adv-prompter`.

## Supplementary Experiments

The supplementary batch scripts are under `scripts/bash_script/rebuttal/`. They include:

- multi-seed ASR runs
- AdvPrompter multi-seed runs
- long-context memory profiling
- nsys memory profiling
- ASR aggregation

Example:

```bash
bash scripts/bash_script/rebuttal/repeated_asr/submit_harmbench_repeated_asr.sh
```

The profiling Python entry points live under `src/test/`; see `src/test/readme.md`.


## License

This project is licensed under the terms of the `LICENSE` file.
