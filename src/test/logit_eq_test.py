#!/usr/bin/env python3
"""
Logit equivalence test: verify that PSKV produces numerically
equivalent logits to full recomputation (No Cache).

Usage:
    python test/logit_eq_test.py \
        --model-id meta-llama/Llama-2-7b-chat-hf \
        --dataset harmbench-test50 \
        --num-prompts 1 \
        --suffix-length 20
"""
import argparse
import os
import sys
import json
import torch
import numpy as np

import utils
from utils import initialize_prefix_cache, forward_with_cache


def test_logit_equivalence(model_id, dataset_name, num_prompts=10,
                           suffix_length=20, search_width=64, save_dir=None):
    """Compare logits between No Cache and PSKV on identical inputs."""

    print(f"{'='*60}")
    print(f"Logit Equivalence Test")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_name}, {num_prompts} prompts, sfx_len={suffix_length}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = utils.get_model(model_id)
    #if you wanna test in float64 for better numerical precision, uncomment the following line
    #model=model.to(torch.float64)
    
    utils.fix_generation_config(model)
    device = model.device

    # Load dataset
    dataset = utils.get_dataset(dataset_name)
    prompts = [dataset[i]["prompt"] for i in range(min(num_prompts, len(dataset)))]
    targets = [dataset[i]["target"] for i in range(min(num_prompts, len(dataset)))]

    # Build inputs (reuse AttackerBase logic)
    from attacks import GCG
    attacker = GCG(
        suffix_length=suffix_length, steps=1, topk=256,
        search_width=search_width, batch_size=num_prompts,
        width_bs=search_width, kv_cache="None",
    )
    pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
    ids_mask_dict = attacker._build_ids_and_mask(
        tokenizer, prompts, targets, device=device, pad_id=pad_id
    )

    message_ids = ids_mask_dict["message_ids"]
    message_mask = ids_mask_dict["message_mask"]
    target_ids = ids_mask_dict["target_ids"]
    target_mask = ids_mask_dict["target_mask"]

    B = len(message_ids)
    msg_len = message_ids.shape[1]
    tar_len = target_ids.shape[1]
    sfx_len = suffix_length

    embedding_layer = model.get_input_embeddings()

    # Build a fixed suffix (deterministic, same for both modes)
    torch.manual_seed(42)
    advsfx_ids = torch.tensor(
        tokenizer.encode(" x" * (sfx_len + 5), add_special_tokens=False),
        dtype=torch.int64, device=device
    )[:sfx_len].unsqueeze(0).expand(B, -1).clone()

    advsfx_mask = torch.ones((B, sfx_len), dtype=torch.int64, device=device)
    full_mask = torch.cat([message_mask, advsfx_mask, target_mask], dim=1)

    # ================================================================
    # Test 1: Full forward (No Cache baseline)
    # ================================================================
    print("\n--- Test 1: Full forward (No Cache) ---")
    with torch.no_grad():
        full_ids = torch.cat([message_ids, advsfx_ids, target_ids], dim=-1)
        outputs_none = model(
            input_ids=full_ids,
            attention_mask=full_mask,
        )
        logits_none = outputs_none["logits"][:, -(tar_len + 1):-1]
    print(f"  logits_none shape: {logits_none.shape}")

    # ================================================================
    # Test 2: PSKV mode (prefix cache + suffix-only forward)
    # ================================================================
    print("\n--- Test 2: PSKV forward (Ours) ---")
    message_embeds = embedding_layer(message_ids)

    with torch.no_grad():
        cache = initialize_prefix_cache(
            model=model, search_width=search_width,
            message_embeds=message_embeds, message_mask=message_mask,
            cache_mode="Ours", grad_bs=B, dataset_size=B,
        )

        # Build suffix+target embeds
        sfx_embeds = embedding_layer(advsfx_ids)
        tar_embeds = embedding_layer(target_ids)
        sfx_tar_embeds = torch.cat([sfx_embeds, tar_embeds], dim=1)

        logits_ours = forward_with_cache(
            model=model, cache_mode="Ours",
            cache=cache[0] if isinstance(cache, list) else cache,
            message_embeds=message_embeds,
            sfx_tar_embeds=sfx_tar_embeds,
            attention_mask=full_mask,
        )[:, -(tar_len + 1):-1]
    print(f"  logits_ours shape: {logits_ours.shape}")

    # ================================================================
    # Test 3: Normal KV cache mode
    # ================================================================
    print("\n--- Test 3: Normal KV cache forward ---")
    with torch.no_grad():
        cache_normal = initialize_prefix_cache(
            model=model, search_width=search_width,
            message_embeds=message_embeds, message_mask=message_mask,
            cache_mode="Normal", grad_bs=B, dataset_size=B,
        )

        logits_normal = forward_with_cache(
            model=model, cache_mode="Normal",
            cache=cache_normal[0] if isinstance(cache_normal, list) else cache_normal,
            message_embeds=message_embeds,
            sfx_tar_embeds=sfx_tar_embeds,
            attention_mask=full_mask,
        )[:, -(tar_len + 1):-1]
    print(f"  logits_normal shape: {logits_normal.shape}")

    # ================================================================
    # Comparison
    # ================================================================
    print(f"\n{'='*60}")
    print("LOGIT COMPARISON RESULTS")
    print(f"{'='*60}")

    comparisons = [
        ("None vs PSKV(Ours)", logits_none, logits_ours),
        ("None vs Normal", logits_none, logits_normal),
        ("Normal vs PSKV(Ours)", logits_normal, logits_ours),
    ]

    results = {}
    for name, logits_a, logits_b in comparisons:
        diff = (logits_a - logits_b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        median_diff = diff.median().item()

        # Also compare the loss values
        logps_a = logits_a.log_softmax(dim=-1)
        logps_b = logits_b.log_softmax(dim=-1)
        loss_a = -(torch.gather(logps_a, -1, target_ids.unsqueeze(-1)).squeeze(-1) * target_mask).mean().item()
        loss_b = -(torch.gather(logps_b, -1, target_ids.unsqueeze(-1)).squeeze(-1) * target_mask).mean().item()
        loss_diff = abs(loss_a - loss_b)

        # Check if argmax predictions match
        pred_a = logits_a.argmax(dim=-1)
        pred_b = logits_b.argmax(dim=-1)
        pred_match_rate = (pred_a == pred_b).float().mean().item()

        print(f"\n  {name}:")
        print(f"    Max absolute logit diff:    {max_diff:.2e}")
        print(f"    Mean absolute logit diff:   {mean_diff:.2e}")
        print(f"    Median absolute logit diff: {median_diff:.2e}")
        print(f"    Loss A: {loss_a:.6f}, Loss B: {loss_b:.6f}, Diff: {loss_diff:.2e}")
        print(f"    Prediction match rate:      {pred_match_rate:.4f} ({pred_match_rate*100:.2f}%)")

        results[name] = {
            "max_logit_diff": max_diff,
            "mean_logit_diff": mean_diff,
            "median_logit_diff": median_diff,
            "loss_a": loss_a,
            "loss_b": loss_b,
            "loss_diff": loss_diff,
            "prediction_match_rate": pred_match_rate,
        }

    # ================================================================
    # Per-prompt breakdown for None vs Ours
    # ================================================================
    print(f"\n{'='*60}")
    print("PER-PROMPT LOGIT DIFF (None vs PSKV)")
    print(f"{'='*60}")
    print(f"  {'Prompt':>6} {'Max Diff':>12} {'Mean Diff':>12} {'Pred Match':>12}")
    print(f"  {'-'*46}")

    diff = (logits_none - logits_ours).abs()
    pred_none = logits_none.argmax(dim=-1)
    pred_ours = logits_ours.argmax(dim=-1)

    for i in range(B):
        max_d = diff[i].max().item()
        mean_d = diff[i].mean().item()
        match = (pred_none[i] == pred_ours[i]).float().mean().item()
        print(f"  {i:>6} {max_d:>12.2e} {mean_d:>12.2e} {match:>11.2f}%")

    # ================================================================
    # Verdict
    # ================================================================
    max_diff_ours = results["None vs PSKV(Ours)"]["max_logit_diff"]
    print(f"\n{'='*60}")
    if max_diff_ours < 1e-3:
        print(f"VERDICT: PASS — Max logit diff = {max_diff_ours:.2e} (< 1e-3)")
        print("PSKV is numerically equivalent to full recomputation.")
    elif max_diff_ours < 1e-1:
        print(f"VERDICT: MARGINAL — Max logit diff = {max_diff_ours:.2e}")
        print("Differences are small but non-trivial. Check RoPE/position_ids.")
    else:
        print(f"VERDICT: FAIL — Max logit diff = {max_diff_ours:.2e}")
        print("Significant numerical differences detected. Debug required.")
    print(f"{'='*60}")

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "logit_equivalence_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Logit equivalence test: None vs PSKV")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset", type=str, default="harmbench-test50")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--suffix-length", type=int, default=20)
    parser.add_argument("--search-width", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="../results/rebuttal/logit_equiv")
    args = parser.parse_args()

    test_logit_equivalence(
        model_id=args.model_id,
        dataset_name=args.dataset,
        num_prompts=args.num_prompts,
        suffix_length=args.suffix_length,
        search_width=args.search_width,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
