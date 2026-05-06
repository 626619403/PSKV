# Test Scripts

This folder stores small, ad hoc test scripts used during development. The scripts are mainly for validating PSKV behavior, such as memory usage, logit equivalence, and cache correctness.

These files are not part of the main project pipeline. They are auxiliary scripts for debugging, profiling, and experimental checks.

## Files

- `mini_cache_test.py`
  - Minimal cache correctness test.
  - Checks whether no-cache, Normal KV cache, and PSKV cache produce matching logits.
  - Covers both `B=1` without padding and `B=4` with padding.

- `logit_eq_test.py`
  - Full logit equivalence test.
  - Compares full recomputation, Normal KV cache, and PSKV using logits, loss values, and predicted tokens.
  - Saves results to `../results/rebuttal/logit_equiv` by default.

- `memory_profile.py`
  - GPU memory profiling script for GCG attacks.
  - Tracks memory usage during prefix cache initialization and attack forward passes.
  - Profiles the `Ours` cache mode by default.

- `nsys_memory_profile.py`
  - Memory profiling script intended for nsys and multi-mode comparisons.
  - Supports `None`, `Normal`, and `Ours` through the `--kv-modes` argument.
  - Produces memory traces, summaries, and PyTorch memory snapshots for deeper analysis.

## Usage

Run these scripts from the `src` directory:

```bash
cd src
python test/mini_cache_test.py
python test/logit_eq_test.py
python test/memory_profile.py
python test/nsys_memory_profile.py
```

Some scripts require GPUs, PyTorch CUDA memory tracking, model weights, and dataset configuration. Make sure the environment can load the project's `utils`, `attacks`, models, and datasets before running them.

## Notes

- These scripts are experimental, so parameters and default paths may change with each test.
- Profiling outputs are usually written under `../results/rebuttal/...` unless a script overrides `--save-dir`.
- For reproducibility, record the model, dataset, batch size, search width, suffix length, and KV cache mode used for each run.
