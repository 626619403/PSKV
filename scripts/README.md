# Running Scripts

Run these commands from the `scripts/` directory unless noted otherwise.

## Evaluation

```bash
cd scripts

bash eval/llama3.sh \
    --src-path ../src \
    --eval-cfg 2 \
    --dataset harmbench-test50 \
    --attack gcg \
    --kv-cache Ours \
    --random-seed 1
```

Supported attacks are `gcg`, `gcq`, `beast`, `beast-vllm`, `beast-sglang`, `autodan-zhu`, `ample-gcg`, and `adv-prompter`. The eval scripts also accept `beast_vllm` and `beast_sglang`; result paths use the underscore form.

Examples:

```bash
# Standard BEAST
bash eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack beast --kv-cache Ours

# BEAST-vLLM baseline
bash eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack beast-vllm --kv-cache None

# BEAST-SGLang baseline
bash eval/llama3.sh --src-path ../src --eval-cfg 2 --dataset harmbench-test50 --attack beast-sglang --kv-cache None
```

Each eval script builds the evaluation set and then judges it with the HarmBench classifier. Outputs are written under:

```text
../results/<model>/eval-<attack>/cfg-<cfg>/kv-cache-<mode>[_seedN]/
```

## Training

```bash
cd scripts

bash train/vicuna.sh \
    --src-path ../src \
    --train-cfg 1 \
    --dataset harmbench \
    --attack ample-gcg \
    --kv-cache Ours
```

Supported training attacks are `ample-gcg` and `adv-prompter`.

## Supplementary Batch Scripts

Supplementary and rebuttal batch scripts live in:

```text
scripts/bash_script/rebuttal/
```

The `repeated_asr/` subdirectory submits the HarmBench seed-1/seed-2 ASR runs:

```bash
bash ./scripts/bash_script/rebuttal/repeated_asr/submit_harmbench_repeated_asr.sh
```
