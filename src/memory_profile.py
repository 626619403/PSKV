#!/usr/bin/env python3
"""
Memory profiling for GCG attack with PSKV vs baseline.

Produces peak memory traces showing how PSKV's layer-wise
dynamic expansion avoids the memory spike of full prefix duplication.
"""
import threading
import time
import traceback
import gc

import argparse
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import transformers
from contextlib import contextmanager

import utils
from attacks import GCG_MEM
from utils import (
    initialize_prefix_cache,
    forward_with_cache,
    get_dataset,
    apply_final_defaults,
)
import sys

class TeeOutput:
    """Write to both stdout and a log file."""
    def __init__(self, filepath, mode="w"):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

class ContinuousMemoryTracker:
    def __init__(self, interval_ms=20):
        self.interval = interval_ms / 1000
        self.records = []
        self.current_phase = "idle"
        self._running = False
        self._thread = None

    def set_phase(self, phase):
        self.current_phase = phase

    def start(self):
        self._running = True
        self.records = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def _sample_loop(self):
        start_time = time.time()
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        while self._running:
            if num_gpus > 0:
                total_allocated = 0
                for i in range(num_gpus):
                    torch.cuda.synchronize(i)
                    total_allocated += torch.cuda.memory_allocated(i)
                self.records.append({
                    "time_s": round(time.time() - start_time, 3),
                    "allocated_mb": round(total_allocated / 1024 / 1024, 2),
                    "phase": self.current_phase,
                })
            time.sleep(self.interval)


def run_profiling(model_id, dataset_name, kv_cache_mode, save_dir,
                  num_prompts=50, suffix_length=20, steps=20,
                  search_width=64, datacollator="llama2-chat"):
    """Run GCG with memory profiling for a few steps."""
    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(save_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    torch.cuda.memory._record_memory_history(max_entries=100000)

    print(f"\n{'='*60}")
    print(f"Profiling: model={model_id}, kv_cache={kv_cache_mode}")
    print(f"{'='*60}")

    cont_tracker = ContinuousMemoryTracker(interval_ms=50)
    cont_tracker.start()

    error_msg = None

    try:
        # Load model
        cont_tracker.set_phase("model_load")
        model, tokenizer = utils.get_model(model_id)
        utils.fix_generation_config(model)

        # Load dataset
        dataset = utils.get_dataset(dataset_name)
        prompts = [dataset[i]["prompt"] for i in range(min(num_prompts, len(dataset)))]
        targets = [dataset[i]["target"] for i in range(min(num_prompts, len(dataset)))]

        # Create attacker
        attacker = GCG_MEM(
            suffix_length=suffix_length,
            steps=steps,
            topk=256,
            mem_tracker=cont_tracker,
            search_width=search_width,
            batch_size=num_prompts,
            width_bs=search_width,
            kv_cache=kv_cache_mode,
            disable_tqdm=True,
        )

        # Prepare inputs
        pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
        ids_mask_dict = attacker._build_ids_and_mask(
            tokenizer, prompts, targets, device=model.device, pad_id=pad_id
        )

        # Reset CUDA stats
        for i in range(num_gpus):
            torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.empty_cache()

        baseline_alloc = sum(
            torch.cuda.memory_allocated(i) for i in range(num_gpus)
        ) / 1024 / 1024 if num_gpus > 0 else 0
        print(f"Baseline GPU allocated (model loaded, {num_gpus} GPUs): {baseline_alloc:.1f} MB")

        # Phase 1: Prefix cache initialization
        cont_tracker.set_phase("prefix_cache_init")
        if kv_cache_mode != "None":
            message_embeds = model.get_input_embeddings()(ids_mask_dict["message_ids"])
            cache = initialize_prefix_cache(
                model=model, search_width=search_width,
                message_embeds=message_embeds,
                message_mask=ids_mask_dict["message_mask"],
                cache_mode=kv_cache_mode,
                grad_bs=num_prompts,
                dataset_size=num_prompts,
            )
            after_cache_alloc = sum(
                torch.cuda.memory_allocated(i) for i in range(num_gpus)
            ) / 1024 / 1024 if num_gpus > 0 else 0
            print(f"After prefix cache init: {after_cache_alloc:.1f} MB (+{after_cache_alloc - baseline_alloc:.1f} MB)")
            del cache
            torch.cuda.empty_cache()


        # Phase 2: Run attack steps
        cont_tracker.set_phase("attack_forward")
        print(f"Running {steps} GCG steps with profiling...")
        attacker.attack_embeds(
            model=model,
            tokenizer=tokenizer,
            device=model.device,
            **ids_mask_dict,
        )

    except Exception as e:
        error_msg = str(e)
        print(f"\nERROR during profiling kv_cache={kv_cache_mode}: {e}")
        traceback.print_exc()

    finally:
        # 1. Stop background tracker
        cont_tracker.stop()

        # 2. Collect peak stats (valid even after OOM — PyTorch keeps the high-water mark)
        peak_alloc = sum(
            torch.cuda.max_memory_allocated(i) for i in range(num_gpus)
        ) / 1024 / 1024 if num_gpus > 0 else 0
        peak_reserved = sum(
            torch.cuda.max_memory_reserved(i) for i in range(num_gpus)
        ) / 1024 / 1024 if num_gpus > 0 else 0
        print(f"Peak GPU allocated ({num_gpus} GPUs): {peak_alloc:.1f} MB")
        print(f"Peak GPU reserved ({num_gpus} GPUs): {peak_reserved:.1f} MB")

        # 3. Dump memory snapshot
        tag = "_FAILED" if error_msg else ""
        snapshot_path = os.path.join(save_dir, f"mem_snapshot_{kv_cache_mode}{tag}.pickle")
        try:
            torch.cuda.memory._dump_snapshot(snapshot_path)
            print(f"Memory snapshot saved to {snapshot_path}")
        except Exception as dump_err:
            print(f"Warning: failed to dump snapshot: {dump_err}")

        # 4. Stop memory history recording
        torch.cuda.memory._record_memory_history(enabled=None)
        del model, tokenizer
        # 5. Free GPU memory for next run
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "model": model_id,
        "kv_cache": kv_cache_mode,
        "num_gpus": num_gpus,
        "tracker_records": cont_tracker.records,
        "baseline_allocated_mb": round(locals().get("baseline_alloc", 0), 2),
        "peak_allocated_mb": round(peak_alloc, 2),
        "peak_reserved_mb": round(peak_reserved, 2),
    }
    if error_msg:
        result["error"] = error_msg
    return result


def print_phase_summary(results):
    """Print per-phase memory summary from ContinuousMemoryTracker records."""
    print(f"\n{'='*60}")
    print("PER-PHASE PEAK MEMORY SUMMARY")
    print(f"{'='*60}")

    for kv_mode in ["None", "Normal", "Ours"]:
        if kv_mode not in results:
            continue
        records = results[kv_mode].get("tracker_records")
        if not records:
            error = results[kv_mode].get("error", "unknown")
            print(f"\n  [{kv_mode}] No tracker records (error: {error})")
            continue

        df = pd.DataFrame(records)
        phase_stats = df.groupby("phase")["allocated_mb"].agg(["max", "mean", "count"])
        phase_stats = phase_stats.sort_values("max", ascending=False)

        print(f"\n  [{kv_mode}] KV Cache Mode")
        if "error" in results[kv_mode]:
            print(f"  *** FAILED (OOM): partial data before crash ***")
        print(f"  {'Phase':<30} {'Peak (MB)':>10} {'Mean (MB)':>10} {'Samples':>8}")
        print(f"  {'-'*62}")
        for phase, row in phase_stats.iterrows():
            print(f"  {phase:<30} {row['max']:>10.1f} {row['mean']:>10.1f} {int(row['count']):>8}")

    # Cross-mode comparison
    modes_with_data = [m for m in ["None", "Normal", "Ours"]
                       if m in results and results[m].get("tracker_records")]
    if len(modes_with_data) >= 2:
        print(f"\n{'='*60}")
        print("CROSS-MODE COMPARISON (peak allocated MB per phase)")
        print(f"{'='*60}")

        peaks = {}
        for mode in modes_with_data:
            df = pd.DataFrame(results[mode]["tracker_records"])
            if df.empty:
                continue
            peaks[mode] = df.groupby("phase")["allocated_mb"].max()

        if peaks:
            all_phases = sorted(set().union(*(p.index for p in peaks.values())))

            header = f"  {'Phase':<30}"
            for mode in peaks:
                label = mode + (" (OOM)" if "error" in results[mode] else "")
                header += f" {label+' (MB)':>16}"
            print(header)
            print(f"  {'-'*(30 + 17 * len(peaks))}")

            for phase in all_phases:
                line = f"  {phase:<30}"
                for mode in peaks:
                    val = peaks[mode].get(phase, 0)
                    line += f" {val:>16.1f}"
                print(line)


def main():
    parser = argparse.ArgumentParser(description="Memory profiling for GCG+PSKV")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset", type=str, default="harmbench-test50")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--suffix-length", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--search-width", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="../results/rebuttal/memory_profile")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "profiling_log.txt")
    sys.stdout = TeeOutput(log_path)
    sys.stderr = TeeOutput(os.path.join(args.save_dir, "profiling_err.txt"))
    results = {}
    for kv_mode in ["None", "Normal", "Ours"]:
        result = run_profiling(
            model_id=args.model_id,
            dataset_name=args.dataset,
            kv_cache_mode=kv_mode,
            save_dir=args.save_dir,
            num_prompts=args.num_prompts,
            suffix_length=args.suffix_length,
            steps=args.steps,
            search_width=args.search_width,
        )
        results[kv_mode] = result

    # Save results
    model_short = args.model_id.split("/")[-1]
    out_path = os.path.join(args.save_dir, f"memory_profile_{model_short}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print overall comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for kv_mode, r in results.items():
        status = " (OOM)" if "error" in r else ""
        print(f"  KV Cache = {kv_mode}{status}:")
        print(f"    Peak Allocated: {r['peak_allocated_mb']:.1f} MB")
        print(f"    Peak Reserved:  {r['peak_reserved_mb']:.1f} MB")

    if "None" in results and "Ours" in results:
        none_peak = results["None"]["peak_allocated_mb"]
        ours_peak = results["Ours"]["peak_allocated_mb"]
        if none_peak > 0 and "error" not in results["None"] and "error" not in results["Ours"]:
            alloc_reduction = (1 - ours_peak / none_peak) * 100
            print(f"\n  Memory reduction (PSKV vs None): {alloc_reduction:.1f}%")

    if "Normal" in results and "error" in results["Normal"]:
        print(f"\n  Normal KV cache: OOM (demonstrates memory overhead of full duplication)")
        if "Ours" in results and "error" not in results["Ours"]:
            print(f"  PSKV succeeded with peak {results['Ours']['peak_allocated_mb']:.1f} MB")

    print_phase_summary(results)

    print(f"\n{'='*60}")
    print("MEMORY SNAPSHOTS (open in https://pytorch.org/memory_viz)")
    print(f"{'='*60}")
    for kv_mode in ["None", "Normal", "Ours"]:
        for suffix in ["", "_FAILED"]:
            path = os.path.join(args.save_dir, f"mem_snapshot_{kv_mode}{suffix}.pickle")
            if os.path.exists(path):
                print(f"  {kv_mode}{suffix}: {path}")


if __name__ == "__main__":
    main()