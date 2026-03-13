import os, sys, logging, yaml
from functools import wraps
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import  Tuple,Optional
import torch.nn.functional as F
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType

def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, args.save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    #logger.info("Arguments")
    #for arg in vars(args):
        #logger.info("    {:<22}        {}".format(arg+":", getattr(args,arg)) )
    logger.info("")

    return logger


def load_yaml(path: str = None, dump_path: str = None):
    res = None
    if path is not None:
        with open(path, "r") as f:
            res = yaml.load(f, Loader=yaml.loader.SafeLoader)
        save_yaml(res, dump_path)
    return res


def save_yaml(config: dict, path: str = None):
    if path is not None:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


def get_model(
        model_id: str,
        device_map:Optional[str]="auto"
    ):
    '''if "Qwen" in model_id:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device_map)'''
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device_map)
    #tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="left")
    return model, tokenizer

import torch

def wrap_model_with_lora(
    model,
    config_dict
):
    if "task_type" in config_dict and isinstance(config_dict["task_type"], str):
        try:
            task_type_str = config_dict["task_type"]
            config_dict["task_type"] = getattr(TaskType, task_type_str)
        except AttributeError:
            raise ValueError(
                f"Invalid 'task_type': '{config_dict['task_type']}'. "
                f"It must be a valid member of peft.TaskType (e.g., 'CAUSAL_LM')."
            )
    lora_config = LoraConfig(**config_dict)
    
    model.add_adapter(lora_config)
    return model

def fix_generation_config(model):
    if hasattr(model, 'generation_config'):
        gen_config = model.generation_config
        if hasattr(gen_config, 'temperature') or hasattr(gen_config, 'top_p'):
            gen_config.do_sample = True
            
__optimizer_zoo__ = {
    "sgd"   : torch.optim.SGD,
    "adam"  : torch.optim.Adam,
    "adamw" : torch.optim.AdamW,
}

def build_optimizer(name, parameters, **kwargs):
    return __optimizer_zoo__[name](parameters, **kwargs)


__scheduler_zoo__ = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
}

def build_scheduler(name, optimizer, **kwargs):
    return __scheduler_zoo__[name](optimizer, **kwargs)


def apply_repetition_penalty(logits, prev_ids, penalty):
    _logits = torch.gather(input=logits, dim=1, index=prev_ids)

    # if score < 0 then repetition penalty has to be multiplied to reduce the previous
    # token probability
    _logits = torch.where(_logits < 0, _logits * penalty, _logits / penalty)

    logits_penalized = torch.scatter(input=logits, dim=1, index=prev_ids, src=_logits)
    return logits_penalized





class ReturnStruct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def clone(self):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.clone()
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)

    def detach(self):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.detach()
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)

    def _detach(self):
        for k, v in self.__dict__.items():
            try:
                v._detach()
            except:
                pass

    def to(self, device):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.to(device)
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)


def loss_seqs(logits,target_ids, reweight_loss=False, **kwargs):

    loss = F.cross_entropy(
        logits.transpose(1, 2),  # (batch_size, vocab_size, seq_len)
        target_ids,              # (batch_size, seq_len)
        reduction="none",        
        **kwargs,
    )

    if reweight_loss:
        factor = torch.arange(loss.shape[1], dtype=loss.dtype, device=loss.device) + 1
        loss = loss / factor[None, :]
    

    loss_batch = loss.mean(dim=1) 
    final_loss = loss_batch.mean()
    return final_loss, loss, loss_batch
    

def list_avg(_list):
    return sum(_list) / len(_list)

# def collate_fn(list_of_data):

#     (
#         message_batch,
#         message_mask_batch,
#         suffix_batch,
#     ) = zip(*list_of_data)

#     context = dotdict()
#     context.full_message = torch.stack(message_batch, dim=0)
#     context.full_message_mask = torch.stack(message_mask_batch, dim=0)
#     context.suffix = torch.stack(suffix_batch, dim=0)
    
#    return context

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def check_jailbroken(response_texts, refusal_prefixes, affirmative_prefixes):
    jailbroken_list = [
        (
            
            not any(ref.lower() in text.lower() for ref in refusal_prefixes)
            #and
            #any(aff.lower() in text.lower() for aff in affirmative_prefixes)
        )
        for text in response_texts 
    ]
    
    jailbroken_avg = list_avg(jailbroken_list)
    return jailbroken_avg, jailbroken_list



def check_success(seq, target_seq):
    success_list = [
        target_seq[i].lower() in text.lower() for i, text in enumerate(seq)
    ]
    success_avg = list_avg(success_list)
    return success_avg, success_list


def check_affirmative(seq, affirmative_prefixes):
    affirmative_list = [
        any(
            [
                text[: len(prefix)].lower() == prefix.lower()
                for prefix in affirmative_prefixes
            ]
        )
        for text in seq
    ]
    affirmative_avg = list_avg(affirmative_list)
    return affirmative_avg, affirmative_list


def hit_rate_at_n(jb_stat, n):
    jb_sum_at_n = np.sum(jb_stat[:, :n], axis=1)
    return np.where(jb_sum_at_n > 0, 1.0, jb_sum_at_n).mean()

def split_list_unevenly(flat_list: list, num_groups: int) -> list[list]:

    base_size = len(flat_list) // num_groups

    remainder = len(flat_list) % num_groups
    
    list_of_lists = []
    current_index = 0
    
    for i in range(num_groups):

        group_size = base_size + 1 if i < remainder else base_size
        

        group = flat_list[current_index : current_index + group_size]
        list_of_lists.append(group)
        
        current_index += group_size
        
    return list_of_lists