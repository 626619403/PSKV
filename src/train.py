#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import argparse
import os
from typing import Any, Dict

import numpy as np
import torch

import utils
from attacks import AdvPrompterOpt, AmpleGCG
from utils import (
    add_attack_train_args,
    add_shared_args,
    apply_final_defaults,
    fix_generation_config,
    wrap_model_with_lora,
)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adversarial attack prompters.")
    add_shared_args(parser)
    add_attack_train_args(parser)
    return parser.parse_args()


def merge_configs_with_args(
    args: argparse.Namespace,
    lora_cfg: Dict[str, Any],
    attacker_cfg: Dict[str, Any],
) -> argparse.Namespace:
    merged = argparse.Namespace(**vars(args))

    def _merge(source: Dict[str, Any]) -> None:
        for key, value in source.items():
            attr = key.replace("-", "_")
            if hasattr(merged, attr):
                if getattr(merged, attr) is None:
                    setattr(merged, attr, value)
            else:
                setattr(merged, attr, value)

    _merge(lora_cfg)
    _merge(attacker_cfg)
    return merged


def run_ample_gcg_training(
    args: argparse.Namespace,
    attacker_cfg: Dict[str, Any],
    lora_cfg: Dict[str, Any],
    logger: Any,
) -> None:
    
    attack_kwargs = {**attacker_cfg, **vars(args)}
    attack_kwargs["save_dir"] = args.save_dir
    attack_kwargs["save_name"] = args.save_name
    attack_kwargs["llm_save_name"] = getattr(args, "llm_save_name", args.save_name)
    attack_kwargs["target_llm"] = args.target_llm or args.model_id
    attack_kwargs["prompter_model_id"] = attack_kwargs.get("prompter_model_id") or args.model_id
    attack_kwargs["train_cfg_path"] = args.atker_cfg_path
    attack_kwargs["lora_cfg_path"] = args.lora_cfg_path
    attack_kwargs["random_seed"] = args.random_seed
    attack_kwargs["auto_train"] = False

    attacker = AmpleGCG(**attack_kwargs)

    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, f"{args.save_name}_results.pt")

    raw_dataset = utils.get_dataset(args.dataset)
    prompts = [example["prompt"] for example in raw_dataset]
    targets = [example.get("target", "") for example in raw_dataset]

    if os.path.exists(results_path):
        logger.info("Loading cached suffix dataset from %s", results_path)
        suffix_results = torch.load(results_path, map_location="cpu")
    else:
        logger.info("Generating training suffix dataset for AmpleGCG.")
        prompter_model, prompter_tokenizer = utils.get_model(args.model_id)
        target_llm_id = attack_kwargs["target_llm"]
        target_llm, target_tokenizer = utils.get_model(target_llm_id)
        fix_generation_config(prompter_model)
        fix_generation_config(target_llm)

        suffix_results = attacker.generate_training_suffixes_ids(
            model=prompter_model,
            tokenizer=prompter_tokenizer,
            message=prompts,
            target=targets,
            device=prompter_model.device,
            target_llm=target_llm,
            target_llm_tokenizer=target_tokenizer,
            save_dir=args.save_dir,
            save_name=args.save_name,
        )
        torch.save(suffix_results, results_path)

        del prompter_model, prompter_tokenizer, target_llm, target_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "Finetuning prompter using %d suffix samples.",
        len(suffix_results.get("adv_suffix", [])),
    )
    attacker.finetune_prompter_dataset(
        prompts=prompts,
        suffix_results=suffix_results,
        base_model_id=args.model_id,
        lora_cfg=lora_cfg,
        train_cfg=attacker_cfg,
        num_epochs=args.epochs,
        logger=logger,
    )


def run_adv_prompter_training(
    args: argparse.Namespace,
    attacker_cfg: Dict[str, Any],
    lora_cfg: Dict[str, Any],
    logger: Any,
) -> None:
    attack_kwargs = {**attacker_cfg, **vars(args)}
    attack_kwargs["save_dir"] = args.save_dir
    attack_kwargs["save_name"] = args.save_name
    attack_kwargs["dataset"] = args.dataset
    attack_kwargs["target_llm"] = args.target_llm or args.model_id
    attack_kwargs["prompter_model_id"] = attack_kwargs.get("prompter_model_id") or args.model_id
    attack_kwargs["lora_cfg_path"] = args.lora_cfg_path
    attack_kwargs["random_seed"] = args.random_seed
    #print(attack_kwargs)
    advprompter = AdvPrompterOpt(logger=logger, **attack_kwargs)

    prompter, prompter_tokenizer = utils.get_model(args.model_id)
    prompter = wrap_model_with_lora(prompter, lora_cfg)
    fix_generation_config(prompter)

    target_llm_id = attack_kwargs["target_llm"]
    target_llm, target_llm_tokenizer = utils.get_model(target_llm_id)
    fix_generation_config(target_llm)
    target_llm_tokenizer.padding_side = 'left'
    prompter_tokenizer.padding_side = 'left'
    raw_dataset = utils.get_dataset(args.dataset)
    prompts = [example["prompt"] for example in raw_dataset]
    targets = [example.get("target", "") for example in raw_dataset]

    optimizer = torch.optim.Adam(
        prompter.parameters(), **advprompter.train_args["prompter_optim_params"]
    )

    advprompter.attack(
        model=prompter,
        tokenizer=prompter_tokenizer,
        message=prompts,
        target=targets,
        target_llm=target_llm,
        target_llm_tokenizer=target_llm_tokenizer,
        save_dir=args.save_dir,
        save_name=args.save_name,
        optimizer=optimizer,
    )

    del prompter, prompter_tokenizer, target_llm, target_llm_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


TRAIN_HANDLERS = {
    "ample-gcg": run_ample_gcg_training,
    "adv-prompter": run_adv_prompter_training,
}


def main() -> None:
    cli_args = get_args()

    lora_cfg = utils.load_yaml(cli_args.lora_cfg_path)
    attacker_cfg = utils.load_yaml(cli_args.atker_cfg_path)

    args = merge_configs_with_args(cli_args, lora_cfg, attacker_cfg)
    args = apply_final_defaults(args)
    #print(args)
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.generic_init(args)

    handler = TRAIN_HANDLERS.get(args.attack_type)
    if handler is None:
        raise ValueError(f"Unknown attack type: {args.attack_type}")

    with utils.resource_monitor(logger):
        handler(args, attacker_cfg, lora_cfg, logger)


if __name__ == "__main__":
    main()
