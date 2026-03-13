
"""
Refactored AdvPrompter implementation.
Modularized and cleaned up version of the original AdvPrompterOpt.
"""
from __future__ import annotations
import os, sys, numpy as np, shutil
import json
from datetime import datetime
from copy import copy
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from torchrl.data import ListStorage, TensorDictPrioritizedReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import utils
from utils import (
    AttackerBase, 
    check_jailbroken,
    dotdict,
    apply_repetition_penalty,
    loss_seqs,
    fix_generation_config,
    initialize_prefix_cache,
    forward_with_cache,
)
from peft import PeftModel
import pdb
import warnings



class AdvPrompterOpt(AttackerBase):
    def __init__(self,
        suffix_length: int,
        epoch: int,
        search_width: int,
        batch_size: int = -1,
        width_bs: int = -1,
        kv_cache: str = "None",
        disable_tqdm: bool = False,
        logger: Optional[utils.logging.Logger] = None,
        **kwargs):
        
        super().__init__(suffix_length)
        self.steps = epoch
        self.search_width = search_width
        self.batch_size = batch_size
        self.width_bs = width_bs
        self.kv_cache = kv_cache
        self.cache_mode = kv_cache
        self.disable_tqdm = disable_tqdm
        self.num=0
        self.step_now=0
        self.device=None #will be replaced as prompter's device in self.attack() 
        self.test_prefixes = kwargs["refusal_prefixes"]
        self.affirmative_prefixes = kwargs['affirmative_prefixes']
        self.generate_params = dict(
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9, 
                            repetition_penalty=kwargs["repetition_penalty"],
                    )
        self.num_beams=kwargs["num_beams"]
        self.num_chunks=kwargs["num_chunks"]
        self.beam_temperature=kwargs["beams"]["temperature"]
        self.repetition_penalty=kwargs["repetition_penalty"]
        self.topk=kwargs["topk"]
        self.lambda_val=kwargs["lambda_val"]
        self.train_args=kwargs["train"]
        self.save_dir=kwargs["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name=kwargs["save_name"]
        self.logger = logger
        self.prompter_model_id = kwargs.get("prompter_model_id") or kwargs.get("model_id")
        self.target_llm_id = kwargs.get("target_llm") or kwargs.get("target_llm_id") or self.prompter_model_id
        self.dataset = kwargs.get("dataset")
        self.lora_cfg_path = kwargs.get("lora_cfg_path")
        self.random_seed = kwargs.get("random_seed")
        self.auto_train = kwargs.get("auto_train", True)
        self.prompter_checkpoint_dir = os.path.join(self.save_dir, "prompter")
        self.prompter_final_dir = os.path.join(self.prompter_checkpoint_dir, "final")
        self._prompter: Optional[transformers.PreTrainedModel] = None

    @torch.no_grad()
    def save_prompter(self, prompter, step: Optional[int] = None):
        os.makedirs(self.prompter_checkpoint_dir, exist_ok=True)
        step_tag = self.step_now if step is None else step
        save_path = os.path.join(self.prompter_checkpoint_dir, f"step_{step_tag}")
        print(f" Saving prompter to {save_path}...")
        prompter.save_pretrained(save_path, save_embedding_layers=True)

    def _latest_prompter_path(self) -> Optional[str]:
        final_path = Path(self.prompter_final_dir)
        if final_path.is_dir():
            return str(final_path)

        base_dir = Path(self.prompter_checkpoint_dir)
        if not base_dir.exists():
            return None

        step_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("step_")])
        if step_dirs:
            return str(step_dirs[-1])
        return None
 
    # def _ensure_prompter(self, target_llm: transformers.PreTrainedModel, target_tokenizer: transformers.PreTrainedTokenizerBase) -> transformers.PreTrainedModel:
    #     if self._prompter is not None:
    #         return self._prompter

    #     checkpoint_path = self._latest_prompter_path()
    #     if checkpoint_path is None:
    #         if not self.auto_train:
    #             raise FileNotFoundError(
    #                 f"AdvPrompter checkpoint missing under {self.prompter_checkpoint_dir}."
    #             )
    #         self._train_prompter(target_llm, target_tokenizer)
    #         checkpoint_path = self._latest_prompter_path()
    #         if checkpoint_path is None:
    #             raise FileNotFoundError(
    #                 f"AdvPrompter training did not produce a checkpoint in {self.prompter_checkpoint_dir}."
    #             )

    #     base_model_id = self.prompter_model_id or getattr(target_llm.config, "_name_or_path", None)
    #     if base_model_id is None:
    #         raise ValueError("Cannot infer base model id for AdvPrompter. Provide 'prompter_model_id'.")

    #     prompter, _ = utils.get_model(base_model_id)

    #     prompter = PeftModel.from_pretrained(prompter, checkpoint_path)
    #     prompter.to(target_llm.device)
    #     prompter.eval()
    #     fix_generation_config(prompter)
    #     self._prompter = prompter
    #     return self._prompter

    # def _train_prompter(self, target_llm: transformers.PreTrainedModel, target_tokenizer: transformers.PreTrainedTokenizerBase) -> None:
    #     required = {
    #         "dataset": self.dataset,
    #         "lora_cfg_path": self.lora_cfg_path,
    #     }
    #     missing = [k for k, v in required.items() if v is None]
    #     if missing:
    #         raise ValueError("Missing AdvPrompter configuration: " + ", ".join(missing))

    #     base_model_id = self.prompter_model_id or getattr(target_llm.config, "_name_or_path", None)
    #     if base_model_id is None:
    #         raise ValueError("Cannot infer base model id for AdvPrompter training.")

    #     if self.logger is None:
    #         log_args = SimpleNamespace(save_dir=self.save_dir, save_name=self.save_name)
    #         self.logger = utils.generic_init(log_args)

    #     if self.random_seed is not None:
    #         torch.manual_seed(self.random_seed)
    #         np.random.seed(self.random_seed)

    #     prompter, prompter_tokenizer = utils.get_model(base_model_id)
    #     lora_cfg = utils.load_yaml(self.lora_cfg_path)
    #     prompter = utils.wrap_model_with_lora(prompter, lora_cfg)
    #     fix_generation_config(prompter)
    #     fix_generation_config(target_llm)

    #     raw_ds = utils.get_dataset(self.dataset)
    #     prompts = [ex["prompt"] for ex in raw_ds]
    #     targets = [ex.get("target", "") for ex in raw_ds]

    #     pad_id = prompter_tokenizer.pad_token_id
    #     if pad_id is None:
    #         pad_id = prompter_tokenizer.eos_token_id
    #     prompter_tokenizer.padding_side = 'left'
    #     ids_mask = self._build_ids_and_mask(
    #         tokenizer=prompter_tokenizer,
    #         message=prompts,
    #         target=targets,
    #         device=prompter.device,
    #         pad_id=pad_id,
    #     )

    #     self.device = prompter.device
    #     self.prompter_optimizer = torch.optim.Adam(
    #         prompter.parameters(), **self.train_args["prompter_optim_params"]
    #     )

    #     sampler = PrioritizedSampler(
    #         max_capacity=self.train_args["replay_buffer"]["size"],
    #         alpha=self.train_args["replay_buffer"]["priority_alpha"],
    #         beta=1.0,
    #     )
    #     self.replay_buffer = ReplayBuffer(
    #         storage=ListStorage(self.train_args["replay_buffer"]["size"]),
    #         batch_size=self.batch_size,
    #         sampler=sampler,
    #         collate_fn=utils.collate_fn,
    #     )

    #     self.train(
    #         prompter=prompter,
    #         prompter_tokenizer=prompter_tokenizer,
    #         message_ids=ids_mask["message_ids"],
    #         message_mask=ids_mask["message_mask"],
    #         target_ids=ids_mask["target_ids"],
    #         target_mask=ids_mask["target_mask"],
    #         target_llm=target_llm,
    #         target_llm_tokenizer=target_tokenizer,
    #         save_dir=self.prompter_checkpoint_dir,
    #         save_name="advprompter",
    #     )

    #     latest = self._latest_prompter_path()
    #     if latest is not None and latest != self.prompter_final_dir:
    #         os.makedirs(self.prompter_checkpoint_dir, exist_ok=True)
    #         if os.path.exists(self.prompter_final_dir):
    #             shutil.rmtree(self.prompter_final_dir)
    #         shutil.copytree(latest, self.prompter_final_dir)

    #     self._prompter = prompter
    #     self._prompter.to(target_llm.device)
    #     self._prompter.eval()

    def train(self,
          prompter: transformers.PreTrainedModel, 
          prompter_tokenizer: transformers.PreTrainedTokenizerBase, 
          message_ids: torch.Tensor, 
          message_mask: torch.Tensor, 
          target_ids: torch.Tensor, 
          target_mask: torch.Tensor, 
          target_llm: transformers.PreTrainedModel=None, 
          target_llm_tokenizer: transformers.PreTrainedTokenizerBase=None, 
          save_dir=None, 
          save_name=None):
        
        B = len(message_ids)
        grad_bs = B if self.batch_size == -1 else self.batch_size
        
        # Prepare caches
        
        cache = initialize_prefix_cache(
            model=prompter,
            message_ids=message_ids,
            message_mask=message_mask,
            cache_mode=self.cache_mode,
            grad_bs=grad_bs,
            dataset_size=B,
            search_width=self.search_width,
        )
        
        # Target LLM cache
        target_message_ids = message_ids.to(target_llm.device)
        target_message_mask = message_mask.to(target_llm.device)

        cache_for_target = cache
        
        pbar = tqdm(range(self.steps), disable=self.disable_tqdm)
        pbar.set_description("Training (epochs)")
        prompter.train()
        target_llm.eval()

        for self.epoch in pbar:
            data = []
            target_loss_opt_list = []
            
            for ii, be in enumerate(range(0, B, grad_bs)):
                ed = min(be + grad_bs, B)
                batch_data = (message_ids[be:ed], message_mask[be:ed], target_ids[be:ed], target_mask[be:ed])
                
                loss_opt = self._train_step(
                    prompter=prompter,
                    prompter_tokenizer=prompter_tokenizer,
                    target_llm=target_llm,
                    target_llm_tokenizer=target_llm_tokenizer,
                    batch_data=batch_data,
                    cache=cache[ii] if cache else None,
                    cache_for_target=cache_for_target[ii] if cache_for_target else None,
                    data_accumulator=data
                )
                target_loss_opt_list.extend(loss_opt)

            # Post-epoch processing
            self._save_suffix_dataset(data)
            avg_loss = sum(target_loss_opt_list) / len(target_loss_opt_list)
            print(f" Train loss epoch {self.epoch}: {avg_loss:.2f}")

            if self._should_eval():
                self.save_prompter(prompter=prompter)
                self.eval(
                    prompter=prompter,
                    prompter_tokenizer=prompter_tokenizer,
                    target_llm=target_llm,
                    target_llm_tokenizer=target_llm_tokenizer,
                    message_ids=message_ids,
                    message_mask=message_mask,
                    target_ids=target_ids,
                    target_mask=target_mask,
                    cache=cache if self.kv_cache!="None" else None 
                )

        if self.save_dir is not None:
            self.save_prompter(prompter=prompter)
            self.eval(
                prompter=prompter,
                prompter_tokenizer=prompter_tokenizer,
                target_llm=target_llm,
                target_llm_tokenizer=target_llm_tokenizer,
                message_ids=message_ids,
                message_mask=message_mask,
                target_ids=target_ids,
                target_mask=target_mask,
                cache=cache if self.kv_cache!="None" else None  
            )

    def _train_step(self, prompter, prompter_tokenizer, target_llm, target_llm_tokenizer,
                   batch_data, cache, cache_for_target, data_accumulator):
        
        batch_message_ids, batch_message_mask, batch_target_ids, batch_target_mask = batch_data
        batch_message_ids = batch_message_ids.to(self.device)
        batch_message_mask = batch_message_mask.to(self.device)
        batch_target_ids = batch_target_ids.to(self.device)
        batch_target_mask = batch_target_mask.to(self.device)
        
        sfx_len = self.suffix_length
        B = batch_message_ids.shape[0]

        with torch.no_grad():
            # 1. Generate Suffixes
            suffix_ids = prompter.generate(
                input_ids=batch_message_ids,
                attention_mask=batch_message_mask,
                max_new_tokens=sfx_len,
                use_cache=False,
                **self.generate_params
            )[:, -sfx_len:]

            # 2. Evaluate Suffixes
            
            target_loss_batch = self._get_target_loss(
                target_llm=target_llm,
                target_tokenizer=target_llm_tokenizer,
                message_ids=batch_message_ids,
                message_mask=batch_message_mask,
                suffix_ids=suffix_ids,
                target_ids=batch_target_ids,
                target_mask=batch_target_mask,
                cache=cache_for_target,
            )

            # 3. Optimize Suffixes (Beam Search + Scoring)
            target_loss_opt_batch, optimized_suffix_ids = self._optimize_suffixes(
                prompter=prompter,
                target_llm=target_llm,
                batch_message_ids=batch_message_ids,
                batch_message_mask=batch_message_mask,
                batch_target_ids=batch_target_ids,
                batch_target_mask=batch_target_mask,
                target_loss_batch=target_loss_batch,
                cache=cache,
                cache_for_target=cache_for_target
            )

            # 4. Accumulate Results
            self._accumulate_data(
                prompter_tokenizer=prompter_tokenizer,
                message_ids=batch_message_ids,
                target_ids=batch_target_ids,
                suffix_ids=optimized_suffix_ids,
                accumulator=data_accumulator
            )
            
            # 5. Get Response
            target_loss_batch, target_response = self._evaluate_suffixes(
                target_llm=target_llm,
                target_tokenizer=target_llm_tokenizer,
                message_ids=batch_message_ids,
                message_mask=batch_message_mask,
                suffix_ids=optimized_suffix_ids,
                target_ids=batch_target_ids, 
                target_mask=batch_target_mask,
                cache=cache_for_target
            )
            
            # 6. Add to Replay Buffer
            self.add_to_replay_buffer(
                message=batch_message_ids,
                message_mask=batch_message_mask,
                suffix=optimized_suffix_ids,
                target_llm_loss=target_loss_batch.to(self.device),
                target_llm_loss_opt=target_loss_opt_batch.to(self.device),
                target_llm_response=target_response,
            )

        # 7. Finetune Prompter
        self.finetune_prompter(prompter, sfx_len)
        self.step_now += B
        
        return target_loss_opt_batch.tolist()

    def _get_target_loss(self, target_llm, target_tokenizer, message_ids, message_mask, suffix_ids, target_ids, target_mask, cache):
        """Compute target LLM loss for given suffixes."""
        target_device = target_llm.device
        
        # Move to target device
        m_ids = message_ids.to(target_device)
        m_mask = message_mask.to(target_device)
        s_ids = suffix_ids.to(target_device)
        t_ids = target_ids.to(target_device)
        t_mask = target_mask.to(target_device)
        
        
        
        new_ids = torch.cat([s_ids, t_ids], dim=1)
        s_mask = torch.ones_like(s_ids)
        combined_mask = torch.cat([m_mask, s_mask, t_mask], dim=1)
        
        logits = forward_with_cache(
            model=target_llm,
            cache_mode=self.cache_mode,
            cache=cache,
            message_ids=m_ids,
            sfx_tar_ids=new_ids,
            attention_mask=combined_mask,
        )
        
        tar_len = t_ids.shape[1]
        pred_target = logits[:, -tar_len - 1 : -1]
        _, _, loss_batch = loss_seqs(pred_target, target_ids=t_ids, reweight_loss=True)
        
        return loss_batch
    
    def _evaluate_suffixes(self, target_llm, target_tokenizer, message_ids, message_mask, suffix_ids, target_ids, target_mask, cache):
        """Evaluate generated suffixes on the target LLM."""
        target_device = target_llm.device
        
        # Move to target device
        m_ids = message_ids.to(target_device)
        m_mask = message_mask.to(target_device)
        s_ids = suffix_ids.to(target_device)
        t_ids = target_ids.to(target_device)
        t_mask = target_mask.to(target_device)
        
        
        new_ids = torch.cat([suffix_ids, target_ids], dim=1)
        # Assuming input_mask handling is implicitly correct or reconstructed. 
        # Reconstructing combined mask:
        s_mask = torch.ones_like(s_ids)
        combined_mask = torch.cat([m_mask, s_mask, t_mask], dim=1)
        
        logits = forward_with_cache(
            model=target_llm,
            cache_mode=self.cache_mode,
            cache=cache,
            message_ids=message_ids,
            sfx_tar_ids=new_ids,
            attention_mask=combined_mask,
        )
        
        tar_len = t_ids.shape[1]
        pred_target = logits[:, -tar_len - 1 : -1]
        _, _, loss_batch = loss_seqs(pred_target, target_ids=t_ids, reweight_loss=True)
        
        # Generate response (for checking jailbreak)
        gen_kwargs = {
            "input_ids": torch.cat([m_ids, s_ids, t_ids], dim=1),
            "attention_mask": combined_mask,
            "use_cache": (self.cache_mode != "None"),
            **self.generate_params,
        }
        res_ids = target_llm.generate(**gen_kwargs)[:, -tar_len:]
        response = target_tokenizer.batch_decode(res_ids, skip_special_tokens=True)
        
        return loss_batch, response

    def _optimize_suffixes(self, prompter, target_llm, batch_message_ids, batch_message_mask,
                           batch_target_ids, batch_target_mask, target_loss_batch, cache, cache_for_target):
        
        sfx_len = self.suffix_length
        B = batch_message_ids.shape[0]
        # grad_bs = B # Assuming process one batch
        
        # Prepare for beam search
        ebs = target_loss_batch.size(0)
        beam_scores = torch.zeros_like(target_loss_batch)
        
        suffix_beams_ids = torch.empty((ebs, 0), dtype=torch.long, device=prompter.device)
        suffix_beams_mask = torch.empty((ebs, 0), dtype=torch.bool, device=prompter.device)

        target_message_ids = batch_message_ids.to(target_llm.device)
        target_message_mask = batch_message_mask.to(target_llm.device)

        msg_len = batch_message_ids.shape[-1]
        tar_len = batch_target_ids.shape[-1]
        
        for idx in range(sfx_len):
            num_beams_in = 1 if idx == 0 else self.num_beams
            num_beams_out = 1 if idx == sfx_len - 1 else self.num_beams

            # Expand inputs for beams
            message_rep = batch_message_ids.unsqueeze(1).expand(B, num_beams_in, -1).reshape(B * num_beams_in, -1) #shape:[B * num_beams_in, msg_len]
            message_rep_mask = batch_message_mask.unsqueeze(1).expand(B, num_beams_in, -1).reshape(B * num_beams_in, -1)
            full_attention_mask = torch.cat([message_rep_mask, suffix_beams_mask], dim=1)
            prompter.disable_adapters()
            if idx==0:
                prompter_next_basemodel_logits = forward_with_cache(
                    model=prompter,
                    cache_mode="None",
                    cache=cache,
                    message_ids=message_rep,
                    sfx_tar_ids=suffix_beams_ids,
                    attention_mask=full_attention_mask,
                    expand_factor=num_beams_in,
                )[:, -1:]
                prompter.enable_adapters()
                prompter_next_logits= forward_with_cache(
                    model=prompter,
                    cache_mode="None",
                    cache=cache,
                    message_ids=message_rep,
                    sfx_tar_ids=suffix_beams_ids,
                    attention_mask=full_attention_mask,
                    expand_factor=num_beams_in,
                )[:, -1:]
            else:
                prompter_next_basemodel_logits = forward_with_cache(
                    model=prompter,
                    cache_mode=self.cache_mode,
                    cache=cache,
                    message_ids=message_rep,
                    sfx_tar_ids=suffix_beams_ids,
                    attention_mask=full_attention_mask,
                    expand_factor=num_beams_in,
                )[:, -1:]
                prompter.enable_adapters()
                prompter_next_logits= forward_with_cache(
                    model=prompter,
                    cache_mode=self.cache_mode,
                    cache=cache,
                    message_ids=message_rep,
                    sfx_tar_ids=suffix_beams_ids,
                    attention_mask=full_attention_mask,
                    expand_factor=num_beams_in,
                )[:, -1:]

            if suffix_beams_ids.size(1) > 0 and self.repetition_penalty:  
                next_dist_logits_prompter = apply_repetition_penalty(
                    logits=prompter_next_logits.squeeze(1),
                    prev_ids=suffix_beams_ids,
                    penalty=self.repetition_penalty,
                ) 
                next_dist_logits_basemodel = apply_repetition_penalty(
                    logits=prompter_next_basemodel_logits.squeeze(1),
                    prev_ids=suffix_beams_ids,
                    penalty=self.repetition_penalty,
                )
            else:
                next_dist_logits_prompter = prompter_next_logits.squeeze(1)
                next_dist_logits_basemodel = prompter_next_basemodel_logits.squeeze(1)

            next_dist_logprobs_basemodel = torch.log_softmax(next_dist_logits_basemodel, dim=-1)
            
            # Sampling candidates
            num_chunks = self.num_chunks
            num_samples_per_beam = self.topk // (num_chunks * num_beams_in)
            
            all_next_token_candidate_ids = None 
            all_candidate_beam_scores = None 
            all_candidate_losses = None 

            for i in range(num_chunks):
                next_dist_logits = next_dist_logits_prompter.clone()
                if all_next_token_candidate_ids is not None:
                     # Mask already selected
                     previous_khot = torch.scatter(torch.zeros_like(next_dist_logits), 1, all_next_token_candidate_ids, 1)
                     next_dist_logits -= 1e10 * previous_khot

                probs = torch.softmax(next_dist_logits / self.generate_params["temperature"], dim=-1)
                next_token_candidate_ids = probs.multinomial(num_samples=num_samples_per_beam, replacement=False)
                
                # Prepare batch for target LLM evaluation
                current_bs, num_samples_actual = next_token_candidate_ids.shape # shape: [] 
                B_eval = current_bs * num_samples_actual
                
                all_beam_size =num_beams_in * num_samples_actual
                # Expand cache for target
                if cache_for_target is not None:
                    cache_for_target.set_expand(all_beam_size)
                
                # We need suffix ids to feed to target
                # If idx > 0, we have previous suffix parts
                if idx > 0:
                    suffix_part = suffix_beams_ids.unsqueeze(1).expand(current_bs, num_samples_actual, idx).reshape(B_eval, idx).to(target_llm.device)
                else: 
                    suffix_part = None
                
                candidate_ids = next_token_candidate_ids.reshape(B_eval, 1).to(target_llm.device)
                if suffix_part is not None:
                    suffix_for_target = torch.cat([suffix_part, candidate_ids], dim=1)
                else:
                    suffix_for_target = candidate_ids
                
                new_ids = torch.cat([suffix_for_target, 
                                    batch_target_ids.to(target_llm.device).unsqueeze(1)
                                    .expand(B, all_beam_size, -1).reshape(B_eval, tar_len)], dim=1)
                
                
                # Attention mask
                msg_len = batch_message_ids.shape[1]
                tar_len = batch_target_ids.shape[1]
                total_current_len = msg_len + suffix_for_target.shape[1] + tar_len
                
                m_mask_exp = target_message_mask.unsqueeze(1).expand(B, num_beams_in * num_samples_actual, -1).reshape(B_eval, -1)
                s_mask = torch.ones((B_eval, suffix_for_target.shape[1]), device=target_llm.device)
                t_mask_exp = batch_target_mask.to(target_llm.device).unsqueeze(1).expand(B, num_beams_in * num_samples_actual, -1).reshape(B_eval, -1)
                combined_mask = torch.cat([m_mask_exp, s_mask, t_mask_exp], dim=1)
                
                # Target Forward
                target_logits = forward_with_cache(
                    model=target_llm,
                    cache_mode=self.cache_mode,
                    cache=cache_for_target,
                    full_ids=new_ids,
                    message_ids=target_message_ids, # Used for non-cache mode or full rebuild check
                    sfx_tar_ids=new_ids,
                    attention_mask=combined_mask,
                    expand_factor=num_beams_in * num_samples_actual
                )

                if cache_for_target is not None:
                    cache_for_target.set_expand(None)

                pred_target = target_logits[:, -tar_len - 1: -1]
                target_ids_eval = batch_target_ids.to(target_llm.device).unsqueeze(1).expand(B, num_beams_in * num_samples_actual, -1).reshape(B_eval, -1)
                _, _, chunk_loss_flat = loss_seqs(pred_target, target_ids=target_ids_eval, reweight_loss=True)
                
                chunk_losses = chunk_loss_flat.view(current_bs, num_samples_actual).to(prompter.device)
                
                # Scoring
                #pdb.set_trace()
                loss_org = target_loss_batch.to(prompter.device)#.unsqueeze(1).expand(-1, num_beams_in).reshape(-1)
                loss_delta = chunk_losses - loss_org[:, None]
                
                selected_logprobs_basemodel = torch.gather(next_dist_logprobs_basemodel, dim=-1, index=next_token_candidate_ids)
                candidate_beam_scores = beam_scores[:, None] + selected_logprobs_basemodel - loss_delta * self.lambda_val

                if all_next_token_candidate_ids is None:
                    all_next_token_candidate_ids = next_token_candidate_ids
                    all_candidate_beam_scores = candidate_beam_scores
                    all_candidate_losses = chunk_losses
                else:
                    all_next_token_candidate_ids = torch.cat((all_next_token_candidate_ids, next_token_candidate_ids), dim=1)
                    all_candidate_beam_scores = torch.cat((all_candidate_beam_scores, candidate_beam_scores), dim=1)
                    all_candidate_losses = torch.cat((all_candidate_losses, chunk_losses), dim=1)

            # Select best beams
            ebs, total_num_samples = all_candidate_beam_scores.shape
            bs = B
            
            # Reshape to (B, num_beams * samples)
            candidate_beam_scores_reshaped = all_candidate_beam_scores.reshape(bs, num_beams_in * total_num_samples)
            
            # We want to sample num_beams_out
            # Softmax over scores
            # Tip: use re-normalization/shift for stability if needed, but here simple softmax
            # We use the trick from original: subtract 1e10 for max to prevent repetition if needed, but here we select top
            
            # Original code applies a "correction" then samples.
            candidate_beam_scores_top_ids = candidate_beam_scores_reshaped.argmax(dim=-1)
            candidate_beam_scores_onehot = torch.zeros_like(candidate_beam_scores_reshaped)
            candidate_beam_scores_onehot.scatter_(1, candidate_beam_scores_top_ids[:, None], 1)
            candidate_beam_scores_corrected = candidate_beam_scores_reshaped - 1e10 * candidate_beam_scores_onehot
            
            beam_probs = torch.softmax(candidate_beam_scores_corrected / self.beam_temperature, dim=-1)
            next_beam_indices = beam_probs.multinomial(num_samples=num_beams_out, replacement=False)
            
            # Add back the best one if we want to ensure greedy best is kept?
            # Original code:
            next_beam_indices = torch.cat(
                [candidate_beam_scores_top_ids[:, None], next_beam_indices[:, :-1]],
                dim=-1,
            )

            # Determine indices into flattened arrays
            next_beam_indices_expanded = (
                next_beam_indices
                + torch.arange(0, bs, device=prompter.device)[:, None]
                * num_beams_in * total_num_samples
            ).reshape(-1)

            next_token_candidate_ids_flat = all_next_token_candidate_ids.reshape(-1, 1)
            
            # Update suffix beams
            if suffix_beams_ids.size(1) == 0:
                suffix_beams_ids = next_token_candidate_ids_flat[next_beam_indices_expanded]
                suffix_beams_mask = torch.ones_like(suffix_beams_ids, dtype=torch.bool)
            else:
                beam_candidates_ids = suffix_beams_ids.unsqueeze(1).expand(ebs, total_num_samples, -1).reshape(ebs * total_num_samples, -1)
                beam_candidates_mask = suffix_beams_mask.unsqueeze(1).expand(ebs, total_num_samples, -1).reshape(ebs * total_num_samples, -1)
                
                beam_candidates_ids = torch.cat([beam_candidates_ids, next_token_candidate_ids_flat], dim=1)
                # mask update
                beam_candidates_mask = torch.cat([beam_candidates_mask, torch.ones((ebs*total_num_samples, 1), dtype=torch.bool, device=prompter.device)], dim=1)
                
                suffix_beams_ids = beam_candidates_ids[next_beam_indices_expanded]
                suffix_beams_mask = beam_candidates_mask[next_beam_indices_expanded]
                
            # Update scores and losses
            candidate_losses_reshaped = all_candidate_losses.reshape(bs, num_beams_in * total_num_samples)
            selected_losses = candidate_losses_reshaped.gather(dim=1, index=next_beam_indices)
            target_loss_batch = selected_losses.reshape(bs * num_beams_out) # Update for next iter
            
            selected_beam_scores = candidate_beam_scores_reshaped.gather(dim=1, index=next_beam_indices)
            beam_scores = selected_beam_scores.reshape(bs * num_beams_out)
            
        num_beams = self.num_beams
        B = batch_message_ids.shape[0]
        
        return target_loss_batch, suffix_beams_ids # Should return the optimized losses
    

    
    def _accumulate_data(self, prompter_tokenizer, message_ids, target_ids, suffix_ids, accumulator):
        for i in range(len(message_ids)):
            accumulator.append((
                prompter_tokenizer.decode(message_ids[i], skip_special_tokens=True),
                prompter_tokenizer.decode(target_ids[i], skip_special_tokens=True),
                prompter_tokenizer.decode(suffix_ids[i], skip_special_tokens=True),
                prompter_tokenizer.decode(torch.cat([message_ids[i], suffix_ids[i]], dim=0), skip_special_tokens=True)
            ))

    def _save_suffix_dataset(self, data):
        
        suffix_dataset_key = f"dataset_opt_{self.step_now}"
        dataset = dotdict(
            data=data,
            fields=["message", "target", "suffix", "full_instruct"],
            suffix_dataset_key=suffix_dataset_key,
        )
        dir=os.path.join(self.save_dir, "suffix_data")
        os.makedirs(dir, exist_ok=True)
        
        save_path = os.path.join(dir, f"{dataset.suffix_dataset_key}.json")

        export_data = []
        for row in dataset.data:
            item = dict(zip(dataset.fields, row))
            export_data.append(item)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
                print(f"Suffix dataset saved to {save_path}")
        except Exception as e:
            print(f"Error saving suffix dataset: {e}")
            
    def _should_eval(self):
         return (
            self.train_args["eval_every"] is not None
            and (self.epoch + 1) % self.train_args["eval_every"] == 0
            and (self.epoch + 1) < self.train_args["epochs"]
        )

    # def add_to_replay_buffer(self, message, message_mask, suffix, target_llm_loss, target_llm_loss_opt, target_llm_response):
    #     loss_batch = target_llm_loss
    #     loss_opt_batch = target_llm_loss_opt

    #     priority = (
    #         torch.relu(loss_batch - loss_opt_batch)
    #         * self.train_args["replay_buffer"]["priority_factor"]["loss_delta"]
    #     )
    #     if self.train_args["replay_buffer"]["priority_factor"]["jailbreaking"] > 0:
    #         _, target_llm_ar_opt_jailbroken_list = check_jailbroken(
    #             response_texts=target_llm_response,
    #             refusal_prefixes=self.test_prefixes,
    #             affirmative_prefixes=self.affirmative_prefixes,
    #         )
    #         jailbroken = torch.tensor(
    #             target_llm_ar_opt_jailbroken_list, device=loss_batch.device
    #         )
    #         priority += (
    #             jailbroken * self.train_args["replay_buffer"]["priority_factor"]["jailbreaking"]
    #         )
        
    #     priority = priority.cpu()
        
    #     # Flatten and add to buffer
    #     B = message.shape[0]
    #     # Prepare full message for finetuning context
    #     full_message = torch.cat([message, suffix], dim=1).detach()
    #     full_message_mask = torch.cat(
    #         [message_mask, torch.ones_like(suffix, dtype=torch.bool)], dim=1).detach()
            
    #     for i in range(B):
    #         if priority[i] > 0:
    #             self.replay_buffer.add([ 
    #                 full_message[i].cpu(),
    #                 full_message_mask[i].cpu(),
    #                 suffix[i].detach().cpu(),
    #                 priority[i]
    #             ])

    def add_to_replay_buffer(self, message, message_mask, suffix, target_llm_loss, target_llm_loss_opt, target_llm_response):

        loss_batch = target_llm_loss
        loss_opt_batch = target_llm_loss_opt

        priority = (
            torch.relu(loss_batch - loss_opt_batch)
            * self.train_args["replay_buffer"]["priority_factor"]["loss_delta"]
        )

        
        if self.train_args["replay_buffer"]["priority_factor"]["jailbreaking"] > 0:
            _, target_llm_ar_opt_jailbroken_list = check_jailbroken(
                response_texts=target_llm_response,
                refusal_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
            )
            jailbroken = torch.tensor(
                target_llm_ar_opt_jailbroken_list, device=loss_batch.device
            )
            priority += (
                jailbroken * self.train_args["replay_buffer"]["priority_factor"]["jailbreaking"]
            )

        priority = priority.cpu()
        valid_mask = priority > 0
        priority = torch.nan_to_num(priority, nan=0.0, posinf=1.0, neginf=0.0)
        priority = torch.clamp(priority, min=1e-6, max=10.0)
        
        num_valid = valid_mask.sum().item()

        if num_valid == 0:
            return 

        full_message = torch.cat([message, suffix], dim=1).detach()
        full_message_mask = torch.cat(
            [message_mask, torch.ones_like(suffix, dtype=torch.bool)], dim=1).detach()


        data_to_add = TensorDict({
            "full_message": full_message[valid_mask].cpu(),
            "full_message_mask": full_message_mask[valid_mask].cpu(),
            "suffix": suffix[valid_mask].detach().cpu(),
            "priority": priority[valid_mask],  
        }, batch_size=[num_valid]) 
        self.replay_buffer.extend(data_to_add)
        
    def finetune_prompter(self, prompter, sfx_len):
        if len(self.replay_buffer) < self.batch_size:
            return None

        num_updates = min(
            self.train_args["replay_buffer"]["num_updates"],
            len(self.replay_buffer) // self.batch_size,
        )

        prompter.train()
        
        for _ in range(num_updates):
            batch = self.replay_buffer.sample()
            batch = batch.to(prompter.device)
            is_weights = batch.get("_weight") # shape: [B]
            
            self.prompter_optimizer.zero_grad()
            pred_logits = prompter(
                input_ids=batch["full_message"], 
                attention_mask=batch["full_message_mask"],
                use_cache=False
            ).logits
            pred_suffix_logits = pred_logits[:, -sfx_len - 1 : -1].contiguous()

            _, _, per_sample_loss = loss_seqs(
                pred_suffix_logits, 
                target_ids=batch["suffix"]
            )
            loss = (per_sample_loss * is_weights.reshape(-1)).mean()
            if torch.isnan(per_sample_loss).any() or torch.isinf(per_sample_loss).any():
                print(f"Warning: Invalid loss detected: {per_sample_loss}")
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prompter.parameters(), 1.0)
            self.prompter_optimizer.step()
            
            new_priorities = per_sample_loss.detach().clone()
            new_priorities = torch.where(
                torch.isfinite(new_priorities),
                new_priorities,
                torch.ones_like(new_priorities) 
            )
            new_priorities = torch.clamp(new_priorities, min=1e-6, max=10.0)
            new_priorities = torch.nan_to_num(new_priorities, nan=0.0, posinf=1.0, neginf=0.0)
            new_priorities = torch.clamp(new_priorities, min=1e-6, max=10.0)
            self.replay_buffer.update_priority(batch["index"], new_priorities.float())

        return loss.item()
        # last_loss = None
        # for _ in range(num_updates):
        #     context, priority_batch = self.replay_buffer.sample(batch_size=self.batch_size)
            
        #     self.prompter_optimizer.zero_grad()
        #     pred_logits = prompter(
        #         input_ids=context["full_message"].to(prompter.device), 
        #         attention_mask=context["full_message_mask"].to(prompter.device),
        #         use_cache=False
        #     ).logits
        #     pred_suffix_logits=pred_logits[:, -sfx_len - 1 : -1].to(prompter.device)
        #     # context suffix includes message? context["suffix"] should be just suffix ids 
        #     # In add_to_replay_buffer we save just suffix. 
        #     suffix_final_loss, _, _ = loss_seqs(pred_suffix_logits, target_ids=context["suffix"].to(prompter.device))
            
        #     suffix_final_loss.backward()
        #     self.prompter_optimizer.step()
        #     last_loss = suffix_final_loss.item()
        # return last_loss
                
    def attack(self,
               model: transformers.PreTrainedModel,
               tokenizer: transformers.PreTrainedTokenizerBase,
               message: Sequence[str],
               target: Sequence[str],
               target_llm: transformers.PreTrainedModel,
               target_llm_tokenizer: transformers.PreTrainedTokenizerBase,
               save_dir: str = None,
               save_name: str = None,
               optimizer: torch.optim.Optimizer = None):
        
        if optimizer is not None:
             self.prompter_optimizer = optimizer
             
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        ids_mask = self._build_ids_and_mask(
            tokenizer=tokenizer,
            message=message,
            target=target,
            device=model.device,
            pad_id=pad_id,
        )
        
        self.device = model.device
        
        # Initialize replay buffer if not done 
        self.replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=self.train_args["replay_buffer"]["priority_alpha"],
            beta=1.0, 
            storage=LazyTensorStorage(max_size=self.train_args["replay_buffer"]["size"]),
            batch_size=self.batch_size,
            priority_key="priority",
        )

        self.train(
            prompter=model,
            prompter_tokenizer=tokenizer,
            message_ids=ids_mask["message_ids"],
            message_mask=ids_mask["message_mask"],
            target_ids=ids_mask["target_ids"],
            target_mask=ids_mask["target_mask"],
            target_llm=target_llm,
            target_llm_tokenizer=target_llm_tokenizer,
            save_dir=save_dir,
            save_name=save_name,
        )

    @torch.no_grad()
    def eval(self,
            prompter: transformers.PreTrainedModel,
            prompter_tokenizer: transformers.PreTrainedTokenizerBase,
            target_llm: transformers.PreTrainedModel,
            target_llm_tokenizer: transformers.PreTrainedTokenizerBase,
            message_ids: torch.Tensor,
            message_mask: torch.Tensor,
            target_ids: torch.Tensor,
            target_mask: torch.Tensor,
            sfx_len: int = None,
            cache=None):  
        
        if sfx_len is None:
            sfx_len = self.suffix_length

        prompter.eval()
        target_llm.eval()
        device = self.device
        B = len(message_ids)
        grad_bs = B if self.batch_size == -1 else self.batch_size
        
        target_message_mask = message_mask.to(target_llm.device)

        
        cache_for_target = cache
        
        total_loss = 0
        total_jailbroken = 0
        processed_samples = 0
        
        pbar = tqdm(range(0, B, grad_bs))
        pbar.set_description("Evaluating")
        
        suffix_data = []
        
        for ii, be in enumerate(pbar):
            ed = min(be + grad_bs, B)
            batch_size = ed - be

            batch_message_ids = message_ids[be:ed].to(prompter.device)
            batch_message_mask = message_mask[be:ed].to(prompter.device)
            batch_target_ids = target_ids[be:ed].to(prompter.device)
            batch_target_mask = target_mask[be:ed].to(prompter.device)
            
            # Prompter generation
            # Original uses cache=False for eval generation
            suffix_output = prompter.generate(
                input_ids=batch_message_ids,
                attention_mask=batch_message_mask,
                max_new_tokens=sfx_len,
                use_cache=False,
                **self.generate_params
            )
            
            suffix_ids = suffix_output[:, -sfx_len:]
            
            # Target evaluation
            # Move to target device
            suffix_ids_for_target = suffix_ids.to(target_llm.device)
            batch_target_ids_for_target = batch_target_ids.to(target_llm.device)
            batch_input_mask = torch.cat([
                batch_message_mask,
                torch.ones((batch_size, sfx_len), dtype=torch.bool, device=device),
                batch_target_mask
            ], dim=1)
            batch_input_mask_for_target = batch_input_mask.to(target_llm.device)
            batch_message_ids_for_target = batch_message_ids.to(target_llm.device)

            with torch.no_grad():
                cache_entry_target = (
                    cache_for_target[ii] if (cache_for_target and ii < len(cache_for_target)) else None
                )

                
                new_ids = torch.cat([batch_target_ids_for_target, suffix_ids], dim=1)
                
                target_pred_logits = forward_with_cache(
                    model=target_llm,
                    cache_mode=self.cache_mode,
                    cache=cache_entry_target,
                    message_ids=batch_message_ids, 
                    sfx_tar_ids=new_ids,
                    attention_mask=batch_input_mask_for_target,
                )

                tar_len = batch_target_ids.shape[1]
                pred_target = target_pred_logits[:, -tar_len - 1: -1]
                _, _, target_loss_batch = loss_seqs(
                    pred_target,
                    target_ids=batch_target_ids_for_target,
                    reweight_loss=True
                )
                
                target_llm_kwargs = {
                        "attention_mask": batch_input_mask_for_target[: , : -tar_len]
                }
                
                # if cache_entry_target is not None:
                #      # Reset cache state for generation if needed
                #     cache_entry_target.set_expand(None)
                #     target_llm_kwargs["input_ids"] = suffix_ids_for_target
                #     target_llm_kwargs["past_key_values"] = cache_entry_target
                #     target_llm_kwargs["use_cache"] = False
                # else:
                #     full_prompt_for_target = torch.cat(
                #         [batch_message_ids_for_target, suffix_ids_for_target, batch_target_ids_for_target], dim=1
                #     )
                #     target_llm_kwargs["input_ids"] = full_prompt_for_target[:, : -tar_len]
                #     target_llm_kwargs["use_cache"] = (self.cache_mode != "None")

                full_prompt_for_target = torch.cat(
                        [batch_message_ids_for_target, suffix_ids_for_target, batch_target_ids_for_target], dim=1
                    )
                target_llm_kwargs["input_ids"] = full_prompt_for_target[:, : -tar_len]
                target_llm_kwargs["use_cache"] = False
                generate_kwargs = target_llm_kwargs.copy()
                generate_kwargs.update({
                    "max_new_tokens": tar_len,
                    "min_new_tokens": tar_len,
                    **self.generate_params
                })
                
                target_response_ids = target_llm.generate(**generate_kwargs)
                target_response_ids = target_response_ids[:, -tar_len:]
                target_responses = target_llm_tokenizer.batch_decode(
                    target_response_ids,
                    skip_special_tokens=True
                )

                _, jailbroken_list = check_jailbroken(
                    response_texts=target_responses,
                    refusal_prefixes=self.test_prefixes,
                    affirmative_prefixes=self.affirmative_prefixes,
                )
                
                for i in range(batch_size):
                    suffix_data.append({
                        'message': prompter_tokenizer.decode(batch_message_ids[i], skip_special_tokens=True),
                        'target': target_llm_tokenizer.decode(batch_target_ids_for_target[i], skip_special_tokens=True),
                        'suffix': prompter_tokenizer.decode(suffix_ids[i], skip_special_tokens=True),
                        'response': target_responses[i],
                        'jailbroken': jailbroken_list[i],
                        'loss': target_loss_batch[i].item() if i < len(target_loss_batch) else 0
                    })

                total_loss += target_loss_batch.sum().item()
                total_jailbroken += sum(jailbroken_list)
                processed_samples += batch_size
                
                pbar.set_description(
                    f"Eval (local) | Loss: {total_loss / processed_samples:.2f} | "
                    f"JB: {total_jailbroken}/{processed_samples}"
                )
        
        eval_metrics = {
            'eval/avg_loss': total_loss / processed_samples,
            'eval/jailbreak_rate': total_jailbroken / processed_samples,
            'eval/total_jailbroken': total_jailbroken,
            'eval/total_samples': processed_samples,
            'step': self.step_now
        }
        
        print(f"\nEvaluation Results (step {self.step_now}):")
        print(f"  Average Loss: {eval_metrics['eval/avg_loss']:.2f}")
        print(f"  Jailbreak Rate: {eval_metrics['eval/jailbreak_rate']:.2%}")
        
        return eval_metrics

