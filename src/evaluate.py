import argparse, os, torch, numpy as np, json
# import pandas as pd
from transformers import AutoTokenizer
import utils, evaltools
import time
import psutil
import threading
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
from tqdm import tqdm
import gc
def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)
    utils.add_eval_args(parser)
    args=parser.parse_args()
    return args


def build_alpacaeval(args, logger):
    default_device = "cuda:0"
    
    model, tokenizer = utils.get_model(args.model_id)

    alpacaeval_cfg = utils.load_yaml(args.alpacaeval_cfg_path)

    if "datacollator_config" not in alpacaeval_cfg.keys():
        alpacaeval_cfg["datacollator_config"] = {}
    alpacaeval_cfg["datacollator_config"]["output_raw"] = True

    if args.datacollator is not None:
        alpacaeval_cfg["datacollator_config"]["name"] = args.datacollator

    utils.save_yaml(alpacaeval_cfg, f"{args.save_dir}/{args.save_name}_alpacaeval-cfg.yaml")

    alpacaeval_cfg = evaltools.AlpacaEvalConfig(**alpacaeval_cfg)

    alpacaeval_set = evaltools.build_alpacaeval(
        eval_config=alpacaeval_cfg,
        model=model,
        tokenizer=tokenizer,
        device=default_device,
    )

    json_path = os.path.join(f"{args.save_dir}/{args.save_name}_alpacaeval.json")
    logger.info(f"saving alpacaeval-output to `{json_path}`")
    with open(json_path, "w") as f:
        json.dump(alpacaeval_set, f, indent=2)

def build_evalset(args, logger):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = utils.get_dataset(args.dataset)
    evalset_cfg = utils.load_yaml(args.evalset_cfg_path)
    if "datacollator_config" not in evalset_cfg.keys():
        evalset_cfg["datacollator_config"] = {}
    evalset_cfg["datacollator_config"]["output_raw"] = True
    if args.datacollator is not None:
        evalset_cfg["datacollator_config"]["name"] = args.datacollator
    atker_cfg = utils.load_yaml(args.atker_cfg_path)
    
    if atker_cfg is not None:
        atker_cfg["kv_cache"]=args.kv_cache
        atker_cfg["save_dir"]=args.save_dir.replace("eval", "train")
        evalset_cfg["attacker_config"] = atker_cfg
        

            
    utils.save_yaml(evalset_cfg, f"{args.save_dir}/{args.save_name}_evalset-cfg.yaml")
    evalset_cfg = evaltools.EvalsetConfig(**evalset_cfg)
    model_path = None
    attacker_name = evalset_cfg.attacker_config["name"]
    if attacker_name in ["beast_vllm", "beast_sglang"]:
        model_path = args.model_id
        model = None
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, use_fast=True, padding_side="left"
        )
    else:
        model, tokenizer = utils.get_model(args.model_id)
        if hasattr(model, 'generation_config'):
            gen_config = model.generation_config
            if not gen_config.do_sample and hasattr(gen_config, 'temperature'):
                gen_config.do_sample = True
                logger.info("Fixed generation config: set do_sample=True")
                
    logger.info("Building evalset... This may take a while.")
    final_evalset = evaltools.build_evalset(
        eval_config=evalset_cfg,
        model=model_path if model_path is not None else model,
        tokenizer=tokenizer,
        dataset=dataset,  
        monitor = args.monitor 
    )
    
    json_path = os.path.join(f"{args.save_dir}/{args.save_name}_evalset.json")
    logger.info(f"Saving combined adv-evalset to `{json_path}`")
    with open(json_path, "w") as f:
        json.dump(final_evalset, f, indent=2)
                  
def judge_evalset(args, logger):

    judger_cfg = utils.load_yaml(args.judger_cfg_path, f"{args.save_dir}/{args.save_name}_judger-cfg.yaml")
    
    
    judger_cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    judger_cfg = evaltools.JudgerConfig(**judger_cfg)
    judger = evaltools.build_judger(judger_cfg)


    with open(args.evalset_path, "r") as f:
        evalset = json.load(f)
    logger.info(f"Loaded evalset from {args.evalset_path}")


    logger.info("Judging evalset...")
    attack_success_rate = judger.judge(evalset=evalset)

    logger.info("Overall attack success rate: {:.3%}".format(attack_success_rate))
      
def main(args, logger):
    if args.exp_type == "build-alpacaeval":
        build_alpacaeval(args, logger)
        
    elif args.exp_type == "build-evalset":
        with utils.resource_monitor(logger) as monitor:
            #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            args.monitor = monitor
            build_evalset(args, logger)
            
    elif args.exp_type == "judge-evalset":
        judge_evalset(args, logger)
    else:
        raise ValueError(f"unknown exp-type `{args.exp_type}`")


if __name__ == "__main__":
    args = get_args()
    logger = utils.generic_init(args)
    args=utils.apply_final_defaults(args)
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
