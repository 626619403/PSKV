# Repeated ASR HarmBench Jobs

This folder intentionally uses the same direct style as `scripts/bash_script/rebuttal/multi_seed_3.sh`.

Submit all HarmBench repeated-ASR jobs:

```bash
bash ./scripts/bash_script/rebuttal/repeated_asr/submit_harmbench_repeated_asr.sh
```

Each submitted job runs one packed batch of HarmBench tasks. Seed `0` is intentionally skipped because the existing results are used as one repeat; the scripts only submit seeds `1` and `2`.
Results are written by the existing eval scripts under:

```text
results/<model>/eval-<attack>/cfg-<cfg>/kv-cache-<None|Ours>_seed<seed>/
```

There is no automatic ASR aggregation in this folder.

The packing is based on `scripts/bash_script/time_table.tex`. Each submitted batch requests `23:50:00` because several `None` runs are too long for a short queue.
