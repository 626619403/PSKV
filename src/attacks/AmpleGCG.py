from __future__ import annotations

import argparse
import gc
import itertools
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset as HFDataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from types import SimpleNamespace

import utils
from utils import (
    AttackerBase,
    DatacollatorConfig,
    DatasetWrapper,
    Loader,
    build_datacollator,
    build_optimizer,
    build_scheduler,
    forward_with_cache,
    initialize_prefix_cache,
)


class _AmpleGCGTraining(AttackerBase):
    
    def __init__(
        self,
        suffix_length: int,
        steps: int,
        topk: int,
        search_width: int,
        batch_size: int = -1,
        width_bs: int = -1,
        kv_cache: str = "None",
        disable_tqdm: bool = False,
        **kwargs
    ):
        super().__init__(suffix_length)
        self.suffix_length = suffix_length
        self.steps = steps
        self.topk = topk
        self.search_width = search_width
        self.batch_size = batch_size
        self.width_bs = width_bs
        self.kv_cache = kv_cache
        self.cache_mode = kv_cache
        self.disable_tqdm = disable_tqdm
        self.kwargs = kwargs
        self.target_llm_name= kwargs.get("target_llm", None)
        self.target_llm_tokenizer_name = kwargs.get("target_llm_tokenizer", None)
        self.test_prefixes = kwargs.get("refusal_prefixes", [])
        self.affirmative_prefixes = kwargs.get("affirmative_prefixes", [])
        
        self.collected_suffixes_list = None
    


    def _compute_gradient_topk(
        self,
        model: transformers.PreTrainedModel,
        cache,
        embedding_layer: nn.Embedding,
        message_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        input_mask: torch.Tensor,
        advsfx_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        grad_bs: int,
        control_weight: float,
    ) -> torch.Tensor:
        B = advsfx_ids.shape[0]
        vocab_size = embedding_layer.weight.shape[0]
        tar_len = target_embeds.shape[1]
        topk_ids = []
        crit = nn.CrossEntropyLoss(reduction='mean')
        for idx, be in enumerate(range(0, B, grad_bs)):
            ed = min(be + grad_bs, B)
            adv_onehot = F.one_hot(advsfx_ids[be:ed], num_classes=vocab_size).to(embedding_layer.weight.dtype)
            adv_onehot.requires_grad_(True)

            adv_embeds = torch.matmul(adv_onehot, embedding_layer.weight)
            new_embeds = torch.cat([adv_embeds, target_embeds[be:ed]], dim=1)
            cache_entry = cache[idx] if cache else None
            logits = forward_with_cache(
                model=model,
                cache_mode=self.cache_mode,
                cache=cache_entry,
                message_embeds=message_embeds[be:ed],
                sfx_tar_embeds=new_embeds,
                attention_mask=input_mask[be:ed],
            )
            target_logits = logits[:, -(tar_len + 1) : -1]
            target_logps = target_logits.log_softmax(dim=-1)
            target_logps = torch.gather(target_logps, dim=-1, index=target_ids[be:ed].unsqueeze(-1)).squeeze(-1)
            target_loss = -(target_logps * target_mask[be:ed]).mean()

            control_logits = logits[:, -(tar_len + self.suffix_length) : -tar_len - 1]
            control_targets = advsfx_ids[be:ed, 1:]
            control_loss = crit(control_logits.reshape(-1, vocab_size), control_targets.reshape(-1))

            total_loss = target_loss + control_weight * control_loss
            gd = torch.autograd.grad(total_loss, adv_onehot)[0]
            topk_ids.append(gd.topk(self.topk, dim=-1, largest=False).indices)
            del gd, adv_onehot, adv_embeds, new_embeds, logits, target_logits, target_logps, control_logits, total_loss, target_loss, control_loss
        
        return torch.cat(topk_ids)

    def _sample_candidate_suffixes(
        self,
        advsfx_ids: torch.Tensor,
        topk_ids: torch.Tensor,
        width: int,
        device: torch.device,
    ):
        B = advsfx_ids.shape[0]
        sfx_len = self.suffix_length
        values = topk_ids.unsqueeze(0).expand(width, -1, -1, -1)
        token_choices = torch.randint(0, self.topk, (width, B, sfx_len, 1), device=device)
        values = torch.gather(values, dim=-1, index=token_choices).squeeze(-1)
        position_indices = torch.randint(0, sfx_len, (width, B, 1), device=device)
        values = torch.gather(values, dim=-1, index=position_indices)
        cand_ids = advsfx_ids.unsqueeze(0).expand(width, -1, -1).clone()
        cand_ids.scatter_(dim=-1, index=position_indices, src=values)
        return cand_ids, position_indices, values

    @torch.no_grad()
    def _score_candidate_suffixes(
        self,
        model: transformers.PreTrainedModel,
        cache,
        embedding_layer: nn.Embedding,
        cand_advsfx_ids: torch.Tensor, # [width, B, sfx_len]
        message_embeds: torch.Tensor,  
        target_embeds: torch.Tensor,  
        input_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        message_ids: torch.Tensor,     # [B, msg_len]
        tokenizer: transformers.PreTrainedTokenizerBase,
        cand_message_offset: torch.Tensor,
        cand_target_offset: torch.Tensor,
        msg_len: int,
        tar_len: int,
        grad_bs: int,
        qry_bs: int,
        L: int,
    ):
        width = cand_advsfx_ids.shape[0]
        B = cand_advsfx_ids.shape[1]
        sfx_len = self.suffix_length
        vocab_size = embedding_layer.weight.shape[0]
        scores = []
        crit = nn.CrossEntropyLoss(reduction='none')

        for outer, be in enumerate(range(0, width, qry_bs)):
            ed = min(be + qry_bs, width)


            scr_rows = []
            for idx, start in enumerate(range(0, B, grad_bs)):
                end = min(start + grad_bs, B)

                # message_ids: [B, msg_len] -> [grad_bs, msg_len] -> expand to [qry_bs, grad_bs, msg_len]
                # cand_advsfx_ids: [width, B, sfx_len] -> slice -> [qry_bs, grad_bs, sfx_len]
                # target_ids: [B, tar_len] -> [grad_bs, tar_len] -> expand to [qry_bs, grad_bs, tar_len]

                curr_msg_ids = message_ids[start:end].unsqueeze(0).expand(ed - be, -1, -1)

                curr_sfx_ids = cand_advsfx_ids[be:ed, start:end]

                curr_tar_ids = target_ids[start:end].unsqueeze(0).expand(ed - be, -1, -1)
                
                full_ids = torch.cat([curr_msg_ids, curr_sfx_ids, curr_tar_ids], dim=-1)
                # Shape: [(ed-be) * (end-start), L]
                full_ids = full_ids.reshape((ed - be) * (end - start), -1)

                cand_mask = input_mask[start:end].unsqueeze(0).expand(ed - be, -1, -1).reshape((ed - be) * (end - start), -1).contiguous()
                
                cache_entry = cache[idx] if cache else None

                message_ids_part = full_ids[:, :msg_len]
                sfx_tar_ids_part = full_ids[:, msg_len:]


                out_logits = forward_with_cache(
                    model=model,
                    cache_mode=self.cache_mode,
                    cache=cache_entry,

                    full_ids=full_ids,         
                    message_ids=message_ids_part,
                    sfx_tar_ids=sfx_tar_ids_part, 
                    attention_mask=cand_mask,
                    expand_factor=ed - be,
                )
                
                
                cand_target_ids = target_ids[start:end].unsqueeze(0).expand(ed - be, -1, -1)
                cand_target_mask = target_mask[start:end].unsqueeze(0).expand(ed - be, -1, -1)
                target_logits = out_logits[:, -(tar_len + 1) : -1]
                target_logits = target_logits.view(ed - be, end - start, *target_logits.shape[1:])
                target_logps = target_logits.log_softmax(dim=-1)
                target_logps = torch.gather(target_logps, dim=-1, index=cand_target_ids.unsqueeze(-1)).squeeze(-1)
                target_scr = -(target_logps * cand_target_mask).mean(dim=-1)

                control_logits = out_logits[:, -(tar_len + sfx_len) : -tar_len - 1]
                control_logits = control_logits.view(ed - be, end - start, sfx_len - 1, vocab_size)
                cand_control_ids = cand_advsfx_ids[be:ed, start:end, 1:]
                control_loss = crit(
                    control_logits.reshape(-1, vocab_size),
                    cand_control_ids.reshape(-1),
                ).view(ed - be, end - start, sfx_len - 1).mean(dim=-1)

                total_scr = target_scr + control_loss
                scr_rows.append(total_scr)

                del (full_ids, cand_mask, message_ids_part, 
                     sfx_tar_ids_part, out_logits, cand_target_ids, 
                     cand_target_mask, target_logits, target_logps, target_scr,
                     control_logits, cand_control_ids, control_loss, total_scr,
                     curr_msg_ids, curr_sfx_ids, curr_tar_ids)
            
            scores.append(torch.cat(scr_rows, dim=1))

        score_tensor = torch.cat(scores, dim=0).view(width, B)

        cand_ids = torch.cat([
            message_ids.unsqueeze(0).expand(width, -1, -1),
            cand_advsfx_ids,
            target_ids.unsqueeze(0).expand(width, -1, -1),
        ], dim=-1).view(width * B, -1)
        
        filter_mask = self._get_filter_mask(
            tokenizer,
            cand_ids,
            cand_message_offset,
            cand_target_offset,
        )
        return score_tensor, filter_mask
    
    def _overgenerate_suffixes(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        message_ids: torch.Tensor,
        message_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        device: Union[str, torch.device],
        save_dir: Optional[str] = None,
        save_name: Optional[str] = None,
        control_weight: float = 0.1,
        *args
    ):

        reqs_grad = []
        for pp in model.parameters():
            reqs_grad.append(pp.requires_grad)
            pp.requires_grad = False
        model.eval()
        max_collect_steps = 10

        
        B = len(message_ids)
        self.collected_suffixes_list = [[] for _ in range(B)]
        sfx_len = self.suffix_length
        topk = self.topk
        width = self.search_width
        grad_bs = B if self.batch_size == -1 else self.batch_size
        qry_bs = width if self.width_bs == -1 else self.width_bs
        vocab_size = model.vocab_size
        message_ids = message_ids.to(device)
        message_mask = message_mask.to(device)
        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)
        self.suffix_counts = torch.zeros(B, dtype=torch.long, device=device)
        embedding_layer = model.get_input_embeddings()

        message_embeds = embedding_layer(message_ids)
        msg_len = message_embeds.shape[1]
        target_embeds = embedding_layer(target_ids)
        tar_len = target_embeds.shape[1]
        message_offset = self._get_prefix_offset_from_mask(message_mask)
        target_offset = self._get_suffix_offset_from_mask(target_mask)
        L = msg_len + sfx_len + tar_len
        cand_message_offset = message_offset.unsqueeze(0).expand(width, -1).reshape(width*B).contiguous()
        cand_target_offset = target_offset.unsqueeze(0).expand(width, -1).reshape(width*B).contiguous()
        advsfx_ids = torch.tensor(tokenizer.encode(" x" * (sfx_len + 5), add_special_tokens=False), dtype=torch.int64, device=device)
        advsfx_ids = advsfx_ids[: sfx_len].unsqueeze(0).expand(B, -1).clone()
        
        
        cache = initialize_prefix_cache(
            model=model,
            message_embeds=message_embeds,
            message_mask=message_mask,
            cache_mode=self.cache_mode,
            grad_bs=grad_bs,
            dataset_size=B,
            search_width=width,
        )
        advsfx_mask = torch.ones((B, sfx_len), dtype=torch.int64, device=device)

        input_mask = torch.cat([message_mask, advsfx_mask, torch.ones_like(target_mask)], dim=1)
        pbar = tqdm(range(self.steps), disable=self.disable_tqdm, desc="AmpleGCG Steps")
        
        for step in pbar:
            topk_ids = self._compute_gradient_topk(
                model=model,
                cache=cache,
                embedding_layer=embedding_layer,
                message_embeds=message_embeds,
                target_embeds=target_embeds,
                input_mask=input_mask,
                advsfx_ids=advsfx_ids,
                target_ids=target_ids,
                target_mask=target_mask,
                grad_bs=grad_bs,
                control_weight=control_weight,
            ).detach()
            with torch.no_grad():
                cand_advsfx_ids, position_indices, sampled_values = self._sample_candidate_suffixes(
                    advsfx_ids=advsfx_ids,
                    topk_ids=topk_ids,
                    width=width,
                    device=device,
                )
                del topk_ids
            
                score_tensor, filter_mask = self._score_candidate_suffixes(
                    model=model,
                    cache=cache,
                    embedding_layer=embedding_layer,
                    cand_advsfx_ids=cand_advsfx_ids,
                    message_embeds=message_embeds,
                    target_embeds=target_embeds,
                    input_mask=input_mask,
                    target_ids=target_ids,
                    target_mask=target_mask,
                    message_ids=message_ids,
                    tokenizer=tokenizer,
                    cand_message_offset=cand_message_offset,
                    cand_target_offset=cand_target_offset,
                    msg_len=msg_len,
                    tar_len=tar_len,
                    grad_bs=grad_bs,
                    qry_bs=qry_bs,
                    L=L,
                )

                filter_mask_view = filter_mask.view(width, B)
                score = score_tensor + filter_mask_view * 1e6

                if step >= self.steps - max_collect_steps:
                    valid_mask = filter_mask_view == 0
                    if valid_mask.any():
                        valid_w_indices, valid_b_indices = torch.where(valid_mask)
                        batch_counts = torch.bincount(valid_b_indices, minlength=B)
                        sorted_b_indices, perm = torch.sort(valid_b_indices)
                        sorted_w_indices = valid_w_indices[perm]
                        all_valid_candidates = cand_advsfx_ids[sorted_w_indices, sorted_b_indices]
                        split_tensors = torch.split(all_valid_candidates, batch_counts.tolist())
                        final_cpu_tensors = [t.to('cpu') for t in split_tensors]
                        for b_idx, tensors_for_b in enumerate(final_cpu_tensors):
                            if tensors_for_b.numel() > 0:
                                self.collected_suffixes_list[b_idx].extend(list(tensors_for_b))
                    else:
                        raise ValueError(f"No valid candidates found in step {step}, all candidates filtered out.")
                
                    
                
                    best_idx = score.argmin(dim=0, keepdim=True)
                    fin_filter = (score.min(dim=0).values > 1e5).unsqueeze(-1)
                    advsfx_ids_old = advsfx_ids.clone()

                    upd_positions = torch.gather(position_indices.squeeze(-1), dim=0, index=best_idx).squeeze(0).unsqueeze(-1)
                    upd_values = torch.gather(sampled_values.squeeze(-1), dim=0, index=best_idx).squeeze(0).unsqueeze(-1)
                    advsfx_ids.scatter_(dim=-1, index=upd_positions, src=upd_values)

                    advsfx_ids = advsfx_ids * (~fin_filter) + advsfx_ids_old * fin_filter
                    del cand_advsfx_ids, position_indices, sampled_values, score_tensor, filter_mask
                    del score, best_idx, fin_filter, advsfx_ids_old, upd_positions, upd_values
                    if step % 5 == 0: 
                        gc.collect()
                        torch.cuda.empty_cache() 
                        
        for pp, rq_gd in zip(model.parameters(), reqs_grad):
            pp.requires_grad = rq_gd

        if hasattr(self, 'collected_suffixes_list') and self.collected_suffixes_list is not None:
            
            all_suffixes_list = self.collected_suffixes_list
            
            total_valid_suffixes = sum(len(b_list) for b_list in all_suffixes_list)
            
            print(f"Processing collected suffixes: {total_valid_suffixes} total.")

        else:
            all_suffixes_list = []
            B = len(advsfx_ids) 
            for b in range(B):
                batch_suffixes = [advsfx_ids[b].clone().to('cpu')] 
                all_suffixes_list.append(batch_suffixes)
            print(f"Warning: No suffixes collected, returning the final optimized suffix for each of the {B} batches.")
        
        
        save_data = {
                'all_suffixes': all_suffixes_list,
                'metadata': {
                    'total_batches': len(all_suffixes_list),
                    'timestamp': datetime.now().isoformat()
                }
            }
        
        torch_path = os.path.join(save_dir, f"{save_name}_all_suffixes.pt")
        torch.save(save_data, torch_path)
        print(f"Saved suffixes to: {torch_path}")
    
        return all_suffixes_list
        
    
    def filter_successful_suffixes(
        self,
        tokenizer,
        message_ids: torch.Tensor,
        suffixes: list,
        target_llm,          
        device,
        *,
        suffix_batch_size: int = 512,
        generation_batch_size: int = 512,
        judge_batch_size: int = 128,
    ):
        if isinstance(suffixes, list):
            suffixes = torch.stack(suffixes, dim=0).to(device)
        else:
            suffixes = suffixes.to(device)
            
        suffixes=torch.unique(suffixes,dim=0)

        start_idx = 0
        successful_tensor = torch.empty(0, suffixes.shape[-1], dtype=suffixes.dtype, device=device)
        
        message_ids = message_ids.to(device)
        total_suffixes = len(suffixes)
        
        print(f"Processing {total_suffixes - start_idx} remaining suffixes...")
        
        
        for batch_start in range(start_idx, total_suffixes, suffix_batch_size):
        
            batch_end = min(batch_start + suffix_batch_size, total_suffixes)
            batch_suffixes = suffixes[batch_start:batch_end]
            generated_texts = self._process_suffixes_and_generate(
                tokenizer=tokenizer,
                message_ids=message_ids,
                suffixes=batch_suffixes,
                target_llm=target_llm,
                device=device,
                generation_batch_size=generation_batch_size
            )
            
            if len(generated_texts) > 0:
                batch_successful_tensor = self._batch_judge(
                    target_tokenizer=tokenizer,  
                    message_ids=message_ids,
                    suffixes=batch_suffixes, 
                    generated_texts=generated_texts, 
                    judge_batch_size=judge_batch_size
                )
                
                if len(batch_successful_tensor) > 0:
                    successful_tensor = torch.cat([successful_tensor, batch_successful_tensor], dim=0)
                    del batch_successful_tensor
            del batch_suffixes, generated_texts
        print(f"Found {len(successful_tensor)} successful suffixes out of {total_suffixes} total")
        return successful_tensor 

    def _process_suffixes_and_generate(
        self,
        tokenizer,
        message_ids: torch.Tensor,
        suffixes: torch.Tensor,
        target_llm,
        device,
        generation_batch_size: int
    ) -> tuple[torch.Tensor, list]:

        print("Preprocessing suffixes...")
        message_ids = message_ids.to(device)
        suffixes = suffixes.to(device)
        message_expanded = message_ids.unsqueeze(0).expand(len(suffixes), -1)  # [num_suffixes, message_len]
        all_batch_ids = torch.cat([message_expanded, suffixes], dim=1)   # [num_suffixes, message_len + suffix_len]
        
        all_batch_mask = torch.ones_like(all_batch_ids).long()
        all_batch_ids = all_batch_ids.to(device)
        all_batch_mask = all_batch_mask.to(device)
        
        
        input_length = all_batch_mask.sum(dim=1)[0].item()  
        
        all_generated_texts = []
        
        print(f"Generating responses with batch size {generation_batch_size}...")
        for i in range(0, len(all_batch_ids), generation_batch_size):
            end_idx = min(i + generation_batch_size, len(all_batch_ids))
            
            batch_ids = all_batch_ids[i:end_idx]
            batch_mask = all_batch_mask[i:end_idx]
            
            with torch.no_grad():
                outputs = target_llm.generate(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    max_new_tokens=60,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=False
                )
                
                generated_parts = outputs[:, input_length:]
                
                generated_texts = tokenizer.batch_decode(generated_parts, skip_special_tokens=True)
                all_generated_texts.extend(generated_texts)
        
        
        
        return all_generated_texts
    
    def _batch_judge(self, target_tokenizer, message_ids, suffixes, generated_texts, judge_batch_size: int) -> torch.Tensor:

        _, jailbroken_list = utils.check_jailbroken(
                response_texts=generated_texts,
                refusal_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
            )
        
        jailbroken_mask = torch.tensor(jailbroken_list, dtype=torch.bool, device=suffixes.device)
        
        if jailbroken_mask.any():
            successful_count = jailbroken_mask.sum().item()
            print(f"✓ Found {successful_count} successful suffixes!")
            

            successful_indices = torch.where(jailbroken_mask)[0].tolist()
            

            for i, idx in enumerate(successful_indices[:3]):
                print(f"Generation: '{generated_texts[idx][:200]}...'")
                print("=" * 50)
                
            if len(successful_indices) > 3:
                print(f"... and {len(successful_indices) - 3} more successful suffixes")

            return suffixes[jailbroken_mask]
        else:
            return torch.empty(0, suffixes.shape[-1], dtype=suffixes.dtype, device=suffixes.device)
        
    @staticmethod
    def _pad_sequences_left(sequences, padding_value):
        from torch.nn.utils.rnn import pad_sequence
        
        reversed_seqs = [torch.flip(seq, [0]) for seq in sequences]
        padded = pad_sequence(reversed_seqs, batch_first=True, padding_value=padding_value)
        return torch.flip(padded, [1])

    @staticmethod
    def load_all_suffixes_tensors(pt_path: str, device: torch.device):
        save_data = torch.load(pt_path, map_location=device)
        return save_data['all_suffixes']
        
    @staticmethod
    def _repeat_mask(ids: torch.Tensor, repeat: int = 3) -> torch.Tensor:
        raise AttributeError("Aborted")
        if ids.ndim == 1:              
            ids = ids.unsqueeze(0)
        eq_next = ids[:, 1:] == ids[:, :-1]          # [N, L-1]
        if eq_next.size(1) < repeat - 1:
            return torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
        kernel = torch.ones(1, 1, repeat - 1, device=ids.device)
        
        import torch.nn.functional as F
        runs = F.conv1d(eq_next.unsqueeze(1).to(torch.float), kernel)
        return (runs.squeeze(1) >= repeat - 1).any(dim=1)


    @staticmethod
    def _contains_refusal(text: str) -> bool:
        return False
    
    def generate_training_suffixes_ids(
        self,
        model,
        tokenizer,
        message,  
        target,
        device,
        target_llm=None,
        target_llm_tokenizer=None,
        save_dir: Optional[str] = None,
        save_name: Optional[str] = None,
    ):
        pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

        message = copy.deepcopy(message)
        target = copy.deepcopy(target)

        ids_mask_dict = self._build_ids_and_mask(tokenizer, message, target, device=device, pad_id=pad_id)
        message_ids = ids_mask_dict["message_ids"].to(device)
        message_mask = ids_mask_dict["message_mask"].to(device)
        target_ids = ids_mask_dict["target_ids"].to(device)
        target_mask = ids_mask_dict["target_mask"].to(device)
        all_suffixes = None
        full_suffixes_path = os.path.join(save_dir, f"{save_name}_all_suffixes.pt")
        #if not os.path.exists(full_suffixes_path):
        all_suffixes = self._overgenerate_suffixes(
                model=model,
                tokenizer=tokenizer,
                message_ids=message_ids,
                message_mask=message_mask,
                target_ids=target_ids,
                target_mask=target_mask,
                device=device,
                save_dir=save_dir,
                save_name=save_name,
                control_weight=self.kwargs.get("control_weight", 0.1),
            )

        num_total_messages = len(message_ids) 
        all_indices = list(range(num_total_messages))

        checkpoint_dir = os.path.join(save_dir, "message_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)



        if target_llm is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        adv_suffix = []  


        pbar = tqdm(all_indices, 
                    desc=f"Filtering successful suffixes", )
        
        for msg_idx in pbar:
            message_checkpoint = os.path.join(checkpoint_dir, f"{save_name}_message_{msg_idx}.pt")

            if os.path.exists(message_checkpoint):
                try:
                    msg_data = torch.load(message_checkpoint, map_location=device)
                    adv_suffix[msg_idx] = msg_data['result_tensor']
                    continue
                except Exception as e:
                    print(f"WARN: failed to load message {msg_idx} checkpoint: {e}")

            msg_ids = message_ids[msg_idx].to(device) 
            suffixes = all_suffixes[msg_idx]
            
            if len(suffixes) > 0:
                if isinstance(suffixes, list):
                    suffixes = [s.to(device) for s in suffixes]
                else:
                    suffixes = suffixes.to(device)
                    
            if len(suffixes) == 0:
                suffix_result = torch.full((1, self.suffix_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
            else:
                filter_model = target_llm if target_llm is not None else model
                filter_tokenizer = target_llm_tokenizer if target_llm is not None else tokenizer

                succ_suffixes = self.filter_successful_suffixes(
                    filter_tokenizer, msg_ids, suffixes, filter_model, device,
                )

                if len(succ_suffixes) == 0:
                    suffix_result = torch.full((1, self.suffix_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
                else:
                    suffix_result = succ_suffixes
            
            adv_suffix[msg_idx] = suffix_result
            torch.save({'result_tensor': suffix_result}, message_checkpoint)

        return adv_suffix     
    

    def finetune_prompter_dataset(
        self,
        prompts: Sequence[str],
        suffix_results: List[Any],
        *,
        base_model_id: str,
        lora_cfg: Dict[str, Any],
        train_cfg: Dict[str, Any],
        num_epochs: int,
        logger: Optional[utils.logging.Logger] = None,
    ) -> None:
        if not suffix_results or type(suffix_results) is not List:
            raise ValueError("Suffix results should be a list for finetuning.")

        if logger:
            logger.info("Starting LoRA finetuning for AmpleGCG prompter.")

        model, tokenizer = utils.get_model(base_model_id)
        device = model.device

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("lora_r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_cfg.get("bias", "none"),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        suffix_entries = suffix_results
        if len(prompts) != len(suffix_entries):
            raise ValueError(
                f"Mismatch between prompts ({len(prompts)}) and suffix entries ({len(suffix_entries)})."
            )

        pairs: List[Dict[str, str]] = []
        for idx, prompt in enumerate(prompts):

            if idx not in suffix_entries:
                continue
                
            suffixes = suffix_entries[idx] 
            if suffixes.ndim == 1:
                suffixes = suffixes.unsqueeze(0)
            
            for i in range(len(suffixes)):
                s_tensor = suffixes[i]

                suffix_text = tokenizer.decode(s_tensor.tolist(), skip_special_tokens=True)
                
                if suffix_text.strip():
                    pairs.append({
                        "prompt": str(prompt),
                        "target": suffix_text,
                    })
        if not pairs:
            raise ValueError("Suffix dataset is empty; cannot perform finetuning.")

        # if len(pairs) > 5000:
        #     import random
        #     random.seed(42)
        #     pairs = random.sample(pairs, 5000)

        hf_dataset = HFDataset.from_list(pairs)
        dataset = DatasetWrapper(hf_dataset, prompt_name="prompt", target_name="target")

        dc_cfg = dict(train_cfg.get("datacollator_config", {}))
        override_collator = self.kwargs.get("datacollator")
        if override_collator:
            dc_cfg["name"] = override_collator
        datacollator = DatacollatorConfig(**dc_cfg)
        collator = build_datacollator(datacollator, tokenizer=tokenizer, is_adv=False)

        train_loader = Loader(
            dataset=dataset,
            batch_size=train_cfg.get("train_batch_size", 32),
            train=True,
            collate_fn=collator,
            num_workers=0,
        )

        optimizer = build_optimizer(parameters=model.parameters(), **train_cfg.get("optimizer_config", {}))
        scheduler = build_scheduler(optimizer=optimizer, **train_cfg.get("scheduler_config", {}))

        model.train()
        total_steps = train_cfg.get("train_steps")
        if total_steps is None:
            total_steps = len(train_loader) * max(1, num_epochs)

        step = 0
        data_iter = itertools.cycle(train_loader)
        while step < total_steps:
            batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}

            attention_mask = batch["input_mask"]
            labels = batch["input_ids"].clone()
            labels[attention_mask == 0] = -100

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            if logger and step % 200 == 0:
                logger.info("Finetune step %d/%d - loss %.4f", step, total_steps, loss.item())

        save_path = os.path.join(self.save_dir, self.llm_save_name + "_finetuned")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        if logger:
            logger.info("Saved finetuned prompter LoRA to %s", save_path)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AmpleGCG(_AmpleGCGTraining):

    def __init__(
        self,
        suffix_length: int,
        steps: Optional[int] = None,
        topk: Optional[int] = None,
        search_width: Optional[int] = None,
        batch_size: int = -1,
        width_bs: int = -1,
        kv_cache: str = "None",
        disable_tqdm: bool = False,
        num_groups: int = 4,
        group_size: int = 4,
        diversity_penalty: float = 0.5,
        diversity_type: str = 'ngram',
        ngram_size: int = 2,
        top_k: int = 16,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.95,
        **kwargs,
    ):
        resolved_steps = steps if steps is not None else kwargs.pop("steps", None)
        resolved_topk = topk if topk is not None else kwargs.pop("topk", None)
        resolved_search_width = search_width if search_width is not None else kwargs.pop("search_width", None)
        if resolved_steps is None or resolved_topk is None or resolved_search_width is None:
            raise ValueError("AmpleGCG requires 'steps', 'topk', and 'search_width' parameters.")
        if kv_cache == "None" and "kv_cache" in kwargs:
            kv_cache = kwargs.pop("kv_cache")

        super().__init__(
            suffix_length=suffix_length,
            steps=resolved_steps,
            topk=resolved_topk,
            search_width=resolved_search_width,
            batch_size=batch_size,
            width_bs=width_bs,
            kv_cache=kv_cache,
            disable_tqdm=disable_tqdm,
            **kwargs,
        )

        self.disable_tqdm = disable_tqdm
        self.num_groups = num_groups
        self.group_size = group_size
        self.num_beams = num_groups * group_size
        self.diversity_penalty = diversity_penalty
        self.diversity_type = diversity_type
        self.ngram_size = ngram_size
        self.generation_params = {
            'do_sample': do_sample,
            'temperature': temperature if do_sample else None,
            'top_k': top_k if do_sample else None,
            'top_p': top_p if do_sample else None,
        }
        self.generation_params = {k: v for k, v in self.generation_params.items() if v is not None}
        self.save_dir = kwargs.get("save_dir")
        self.llm_save_name = kwargs.get("llm_save_name") or kwargs.get("save_name")
        if self.save_dir is None or self.llm_save_name is None:
            raise ValueError("AmpleGCG requires 'save_dir' and 'llm_save_name' (or 'save_name').")
        self.load_path = os.path.join(self.save_dir, self.llm_save_name + "_finetuned")
        self.prompter_model_id = kwargs.get("prompter_model_id")
        self.target_llm_id = kwargs.get("target_llm") or kwargs.get("target_llm_id")
        self.dataset = kwargs.get("dataset")
        self.train_cfg_path = kwargs.get("train_cfg_path")
        self.lora_cfg_path = kwargs.get("lora_cfg_path")
        self.random_seed = kwargs.get("random_seed")
        self.auto_train = kwargs.get("auto_train", True)
        self.train_epochs = kwargs.get("epochs", kwargs.get("train_epochs", 10))
        self.test_prefixes = kwargs.get("refusal_prefixes", [])
        self.affirmative_prefixes = kwargs.get("affirmative_prefixes", [])
        self._prompter = None
        self.load_path = os.path.join(self.save_dir, self.llm_save_name + "_finetuned")
        self.test_prefixes = kwargs.get("refusal_prefixes", [])
        self.affirmative_prefixes = kwargs.get("affirmative_prefixes", [])
        self.prompter_model_id = kwargs.get("prompter_model_id")
        self.target_llm_id = kwargs.get("target_llm") or kwargs.get("target_llm_id")
        self.dataset = kwargs.get("dataset")
        self.train_cfg_path = kwargs.get("train_cfg_path")
        self.lora_cfg_path = kwargs.get("lora_cfg_path")
        self.random_seed = kwargs.get("random_seed")
        self.auto_train = kwargs.get("auto_train", True)
        self._prompter = None

    def _ensure_prompter(self, reference_model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer) -> transformers.PreTrainedModel:
        if self._prompter is not None:
            if self._prompter.device != reference_model.device:
                self._prompter.to(reference_model.device)
            return self._prompter

        base_model_id = self.prompter_model_id or getattr(reference_model.config, "_name_or_path", None)
        if base_model_id is None:
            raise ValueError(
                "AmpleGCG requires either 'prompter_model_id' or a reference model "
                "with config._name_or_path set to load the fine-tuned prompter."
            )

        if not os.path.exists(self.load_path):
            if not self.auto_train:
                raise FileNotFoundError(
                    f"Finetuned prompter not found at {self.load_path}. "
                    "Set 'auto_train'=True or provide a valid checkpoint."
                )
            self._train_prompter(base_model_id, reference_model, tokenizer)
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(
                    f"Finetuned prompter still missing after training attempt: {self.load_path}"
                )

        load_kwargs = {}
        if hasattr(reference_model, "dtype") and reference_model.dtype is not None:
            load_kwargs["dtype"] = reference_model.dtype

        prompter = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
        prompter = PeftModel.from_pretrained(prompter, self.load_path)
        prompter.to(reference_model.device)
        prompter.eval()
        self._prompter = prompter
        return self._prompter

    def _train_prompter(self, base_model_id: str, tar_model: transformers.PreTrainedModel, tar_tokenizer: transformers.PreTrainedTokenizer):
        required = {
            "train_cfg_path": self.train_cfg_path,
            "lora_cfg_path": self.lora_cfg_path,
            "dataset": self.dataset,
            "save_dir": self.save_dir,
            "llm_save_name": self.llm_save_name,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                "Missing configuration for AmpleGCG auto-train: " + ", ".join(missing)
            )

        os.makedirs(self.save_dir, exist_ok=True)
        logger = getattr(self, "logger", None)
        if logger is None:
            log_args = SimpleNamespace(save_dir=self.save_dir, save_name=self.llm_save_name)
            logger = utils.generic_init(log_args)

        lora_cfg = utils.load_yaml(self.lora_cfg_path)
        train_cfg = utils.load_yaml(self.train_cfg_path)

        raw_dataset = utils.get_dataset(self.dataset)
        prompts = [example["prompt"] for example in raw_dataset]
        targets = [example.get("target", "") for example in raw_dataset]

        prompter_model, prompter_tokenizer = utils.get_model(base_model_id)

        suffix_results = self.generate_training_suffixes_ids(
            model=prompter_model,
            tokenizer=prompter_tokenizer,
            message=prompts,
            target=targets,
            device=prompter_model.device,
            target_llm=tar_model,
            target_llm_tokenizer=tar_tokenizer,
            save_dir=self.save_dir,
            save_name=self.llm_save_name,
        )

        results_path = os.path.join(self.save_dir, f"{self.llm_save_name}_results.pt")
        torch.save(suffix_results, results_path)

        del prompter_model, prompter_tokenizer, target_llm, target_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.finetune_prompter_dataset(
            prompts=prompts,
            suffix_results=suffix_results,
            base_model_id=base_model_id,
            lora_cfg=lora_cfg,
            train_cfg=train_cfg,
            num_epochs=self.train_epochs,
            logger=logger,
        )

    def attack_embeds(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        message_ids: torch.Tensor,
        message_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        """
        Generates adversarial suffixes for a batch of messages using batched group beam search.
        """
        
        prompter = self._ensure_prompter(model, tokenizer)
        with torch.no_grad():
            advsfx_ids = self._generate_suffix_batch(
                prompter, tokenizer, message_ids, message_mask
            )
        return advsfx_ids

    def _generate_suffix_batch(self, model, tokenizer, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        model_device = next(model.parameters()).device
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        # Expand prompts to number of beams
        input_ids = prompt_ids.unsqueeze(1).expand(-1, self.num_beams, -1).reshape(batch_size * self.num_beams, prompt_len)

        attention_mask = prompt_mask.unsqueeze(1).expand(-1, self.num_beams, -1).reshape(batch_size * self.num_beams, prompt_len)
        
        

        # A tensor to track which beams are finished
        finished_beams = torch.zeros(batch_size, self.num_beams, device=model_device, dtype=torch.bool)
        group_size = self.group_size
        beam_scores = torch.zeros(batch_size, self.num_groups, group_size, device=model_device)
        cache = None
        if self.num_beams > 1:
            beam_scores[:, 1:] = -1e9
        for step in tqdm(range(self.suffix_length), disable=self.disable_tqdm, desc="Batched AmpleGCG"):
            
            if cache is not None:
                outputs = model(input_ids=input_ids[:, -1:], attention_mask=attention_mask, use_cache=True, past_key_values=cache)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            if tokenizer.eos_token_id is not None:
                next_token_logits[:, tokenizer.eos_token_id] = -float('inf')
            if tokenizer.pad_token_id is not None:
                next_token_logits[:, tokenizer.pad_token_id] = -float('inf')    
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :].view(batch_size, self.num_groups, group_size, -1)
            selected_tokens_this_step = [] 
    
            new_tokens_list = []
            new_scores_list = []
            parent_beam_indices_list = []
            # Apply sampling parameters
            for g in range(self.num_groups):
                
                group_logits = logits[:, g, :, :] # [batch, group_size, vocab]
                group_logits = self._apply_sampling_params(group_logits) # Apply Temp/TopK/TopP
                
                # Apply Diversity Penalty
                if g > 0 and self.diversity_penalty > 0:
                    for prev_tokens in selected_tokens_this_step:
                        # prev_tokens: [batch, group_size]
                        penalty_mask = torch.zeros_like(group_logits)
                        penalty_mask.scatter_add_(-1, prev_tokens.unsqueeze(-1), torch.ones_like(prev_tokens.unsqueeze(-1), dtype=group_logits.dtype))
                        group_logits -= penalty_mask * self.diversity_penalty
                
                group_log_probs = F.log_softmax(group_logits, dim=-1)
                
                # D. Beam Selection (Intra-group)
                # score = current_beam_score + token_prob
                # [batch, group_size, 1] + [batch, group_size, vocab]
                candidate_scores = beam_scores[:, g, :].unsqueeze(-1) + group_log_probs
                
                # Flatten to pick best in group
                # [batch, group_size * vocab]
                flat_candidate_scores = candidate_scores.view(batch_size, -1)
                #flat_candidate_indices = candidate_indices.flatten(start_dim=1)
                
                
                top_scores, top_indices = torch.topk(flat_candidate_scores, self.group_size, dim=-1)
                vocab_size = group_logits.shape[-1]
                token_ids = top_indices % vocab_size
                beam_offset_in_group = top_indices // vocab_size # range: 0 ~ group_size-1
                
                # Calculate Global Beam Index (0 ~ num_beams-1)
                # global_idx = group_start_idx + offset
                global_beam_idx = g * self.group_size + beam_offset_in_group
                
                new_scores_list.append(top_scores)
                new_tokens_list.append(token_ids)
                parent_beam_indices_list.append(global_beam_idx)
                
                selected_tokens_this_step.append(token_ids) # For diversity penalty in next groups

            # stack back to [batch, num_groups, group_size] then flatten to [batch, num_beams]
            beam_scores = torch.stack(new_scores_list, dim=1).view(batch_size, self.num_groups, self.group_size)
            
            # [batch, num_beams]
            next_tokens = torch.stack(new_tokens_list, dim=1).view(batch_size, -1)
            source_beam_indices = torch.stack(parent_beam_indices_list, dim=1).view(batch_size, -1)  # (batch_size, num_groups * group_size)
            batch_base_indices = torch.arange(batch_size, device=model_device).view(-1, 1) * self.num_beams
            flat_beam_indices = (batch_base_indices + source_beam_indices).view(-1)

            # 1. Reorder Input IDs (History)
            input_ids = input_ids[flat_beam_indices]
            
            # 2. Append New Tokens
            input_ids = torch.cat([input_ids, next_tokens.view(-1, 1)], dim=-1)
            
            # 3. Reorder Attention Mask & Update
            attention_mask = attention_mask[flat_beam_indices]
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens.view(-1, 1))], dim=-1)
            
            # 4. Reorder KV Cache

            cache = model._reorder_cache(cache, flat_beam_indices)


        flat_scores = beam_scores.view(batch_size, -1)
        best_indices = flat_scores.argmax(dim=-1) # [batch]
        
        # Retrieve best sequences
        # input_ids: [batch * num_beams, total_len]
        # We need to pick the specific row corresponding to the best beam for each batch item
        best_global_indices = torch.arange(batch_size, device=model_device) * self.num_beams + best_indices
        final_sequences = input_ids[best_global_indices]
        
        # Trim prompt
        generated_suffixes = final_sequences[:, prompt_len:]
        
        return generated_suffixes

def _apply_sampling_params(self, logits: torch.Tensor) -> torch.Tensor:
        if 'temperature' in self.generation_params:
            logits /= self.generation_params['temperature']
        if 'top_k' in self.generation_params:
            top_k = self.generation_params['top_k']
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            full_logits = torch.full_like(logits, float('-inf'))
            full_logits.scatter_(-1, top_k_indices, top_k_logits)
            logits = full_logits
        if 'top_p' in self.generation_params:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.generation_params['top_p']
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        return logits
    

