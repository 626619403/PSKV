#!/usr/bin/env python3
"""
Per-layer memory profiling for GCG attack with PSKV vs baseline.

Produces per-layer peak memory traces showing how PSKV's layer-wise
dynamic expansion avoids the memory spike of full prefix duplication.
"""
import argparse
import os
import sys
import json
import torch
import numpy as np
import transformers
from contextlib import contextmanager

import utils
from attacks import GCG
from utils import (
    initialize_prefix_cache,
    forward_with_cache,
    get_dataset,
    apply_final_defaults,
)


class LayerMemoryTracker:
    """Hook-based tracker that records GPU memory at each transformer layer."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_memory = {}  # {phase: {layer_idx: [mem_values]}}
        self.current_phase = "idle"

    def set_phase(self, phase):
        self.current_phase = phase

    def _hook_fn(self, layer_idx):
        def hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024

                phase = self.current_phase
                if phase not in self.layer_memory:
                    self.layer_memory[phase] = {}
                if layer_idx not in self.layer_memory[phase]:
                    self.layer_memory[phase][layer_idx] = []
                self.layer_memory[phase][layer_idx].append({
                    "allocated_mb": round(allocated_mb, 2),
                    "reserved_mb": round(reserved_mb, 2),
                })
        return hook

    def register_hooks(self):
        layers = None
        # Try common model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers  # Llama, Mistral, Qwen
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h  # GPT-2 style

        if layers is None:
            print("Warning: Could not find transformer layers for hooking.")
            return

        for idx, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._hook_fn(idx))
            self.hooks.append(handle)
        print(f"Registered memory hooks on {len(self.hooks)} layers.")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_summary(self):
        """Return per-phase, per-layer peak allocated memory."""
        summary = {}
        for phase, layers in self.layer_memory.items():
            summary[phase] = {}
            for layer_idx, records in layers.items():
                peak_alloc = max(r["allocated_mb"] for r in records)
                peak_reserved = max(r["reserved_mb"] for r in records)
                summary[phase][layer_idx] = {
                    "peak_allocated_mb": peak_alloc,
                    "peak_reserved_mb": peak_reserved,
                    "num_calls": len(records),
                }
        return summary

    def reset(self):
        self.layer_memory = {}


def run_profiling(model_id, dataset_name, kv_cache_mode, num_prompts=5,
                  suffix_length=20, steps=3, search_width=64, datacollator="llama2-chat"):
    """Run GCG with memory profiling for a few steps."""
    print(f"\n{'='*60}")
    print(f"Profiling: model={model_id}, kv_cache={kv_cache_mode}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = utils.get_model(model_id)
    utils.fix_generation_config(model)

    # Setup tracker
    tracker = LayerMemoryTracker(model)
    tracker.register_hooks()

    # Load dataset
    dataset = utils.get_dataset(dataset_name)
    prompts = [dataset[i]["prompt"] for i in range(min(num_prompts, len(dataset)))]
    targets = [dataset[i]["target"] for i in range(min(num_prompts, len(dataset)))]

    # Create attacker
    attacker = GCG(
        suffix_length=suffix_length,
        steps=steps,
        topk=256,
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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    baseline_alloc = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"Baseline GPU allocated (model loaded): {baseline_alloc:.1f} MB")

    # Phase 1: Prefix cache initialization
    tracker.set_phase("prefix_cache_init")
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
        after_cache_alloc = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        print(f"After prefix cache init: {after_cache_alloc:.1f} MB (+{after_cache_alloc - baseline_alloc:.1f} MB)")

    # Phase 2: Run attack steps with profiling
    tracker.set_phase("attack_forward")
    print(f"Running {steps} GCG steps with profiling...")
    attacker.attack_embeds(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        **ids_mask_dict,
    )

    peak_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"Peak GPU allocated: {peak_alloc:.1f} MB")
    print(f"Peak GPU reserved: {peak_reserved:.1f} MB")

    # Get per-layer summary
    summary = tracker.get_summary()
    tracker.remove_hooks()

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_id,
        "kv_cache": kv_cache_mode,
        "baseline_allocated_mb": round(baseline_alloc, 2),
        "peak_allocated_mb": round(peak_alloc, 2),
        "peak_reserved_mb": round(peak_reserved, 2),
        "per_layer_summary": {
            phase: {str(k): v for k, v in layers.items()}
            for phase, layers in summary.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Per-layer memory profiling for GCG+PSKV")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset", type=str, default="harmbench-test50")
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--suffix-length", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--search-width", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="../results/rebuttal/memory_profile")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    results = {}
    for kv_mode in ["None", "Ours"]:
        result = run_profiling(
            model_id=args.model_id,
            dataset_name=args.dataset,
            kv_cache_mode=kv_mode,
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

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for kv_mode, r in results.items():
        print(f"  KV Cache = {kv_mode}:")
        print(f"    Peak Allocated: {r['peak_allocated_mb']:.1f} MB")
        print(f"    Peak Reserved:  {r['peak_reserved_mb']:.1f} MB")

    if "None" in results and "Ours" in results:
        alloc_reduction = (1 - results["Ours"]["peak_allocated_mb"] / results["None"]["peak_allocated_mb"]) * 100
        print(f"\n  Memory reduction (PSKV vs None): {alloc_reduction:.1f}%")

    # Print per-layer comparison for attack_forward phase
    print(f"\n{'='*60}")
    print("PER-LAYER PEAK ALLOCATED MEMORY (attack_forward phase)")
    print(f"{'Layer':<8} {'None (MB)':<15} {'PSKV (MB)':<15} {'Reduction':<10}")
    print("-" * 48)
    for kv_mode in ["None", "Ours"]:
        if "attack_forward" not in results[kv_mode]["per_layer_summary"]:
            print(f"  No attack_forward data for {kv_mode}")
            continue

    none_layers = results.get("None", {}).get("per_layer_summary", {}).get("attack_forward", {})
    ours_layers = results.get("Ours", {}).get("per_layer_summary", {}).get("attack_forward", {})

    all_layers = sorted(set(list(none_layers.keys()) + list(ours_layers.keys())), key=lambda x: int(x))
    for layer in all_layers:
        none_val = none_layers.get(layer, {}).get("peak_allocated_mb", 0)
        ours_val = ours_layers.get(layer, {}).get("peak_allocated_mb", 0)
        reduction = (1 - ours_val / none_val) * 100 if none_val > 0 else 0
        print(f"  {layer:<8} {none_val:<15.1f} {ours_val:<15.1f} {reduction:<10.1f}%")


if __name__ == "__main__":
    main()
