import torch
import numpy as np
from typing import Union, List, Optional
from tqdm import tqdm
import pdb
import asyncio
try:
    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
    from sglang import Engine
except ImportError:
    print("Error: sglang not installed.")
# monitor.peak_system_reserved_mb
from utils import AttackerBase

class BEAST_SGLang(AttackerBase):
    """
    SGLang accelerated implementation of BEAST attack.
    Leverages RadixAttention for efficient KV cache reuse during beam search.
    """
    def __init__(
        self,
        suffix_length: int,
        beam_size: int,
        search_width: int,
        gpu_memory_utilization: float = 0.9,
        tp_size: int = 1,
        **kwargs
    ):
        super().__init__(suffix_length)
        self.beam_size = beam_size
        self.search_width = search_width
        self.tp_size = tp_size
        self.gpu_memory_utilization = gpu_memory_utilization
        

    def attack_embeds(
            self,
            model, 
            tokenizer,
            message_ids: torch.Tensor,
            message_mask: torch.Tensor,
            target_ids: torch.Tensor,
            target_mask: torch.Tensor,
            device: Union[str, torch.device] = 'cuda',
        ) -> torch.Tensor:

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        batch_size = message_ids.shape[0]
        self.engine = Engine(
            model_path=model,
            tp_size=self.tp_size,
            mem_fraction_static=self.gpu_memory_utilization,
            trust_remote_code=True
        )
        del tokenizer
        
        msg_ids_batch = []
        tgt_ids_batch = []
        
        for i in range(batch_size):
            m_len = int(message_mask[i].sum().item())
            t_len = int(target_mask[i].sum().item())
            # Slicing
            msg_ids_batch.append(message_ids[i].tolist()[-m_len:])
            tgt_ids_batch.append(target_ids[i].tolist()[:t_len])

        sfx_len = self.suffix_length
        beam_size = self.beam_size
        width = self.search_width


        
        init_prompts = []
        init_map_to_batch = [] 

        for b_idx, msg_ids in enumerate(msg_ids_batch):
            for _ in range(beam_size):
                init_prompts.append(msg_ids)
                init_map_to_batch.append(b_idx)

        init_params = {"temperature": 0.0, "max_new_tokens": 1}
        
        # Batch Call
        init_outputs = self.engine.generate(input_ids=init_prompts, sampling_params=init_params, return_logprob=True,
                top_logprobs_num=width)

        # current_beams[b_idx] = List[Beam]
        current_beams = [[] for _ in range(batch_size)]
        
        for i, out in enumerate(init_outputs):
            b_idx = init_map_to_batch[i]
            if 'meta_info' in out and 'output_token_logprobs' in out['meta_info']:
                #pdb.set_trace()
                token_logprobs = out['meta_info']['output_token_logprobs'][0]
                
                if isinstance(token_logprobs, (list, tuple)) and len(token_logprobs) >= 2:
                    gen_id = token_logprobs[1] 
            else:
                 gen_id = out['token_ids'][-1] # Fallback

            current_beams[b_idx].append([gen_id])
            
        pbar = tqdm(range(1, sfx_len), disable=False, desc="BEAST SGLang Batch")
        
        for step in pbar:
            
            # === A. Expand Step ===
            base_prompts = []
            map_info = []

            for b_idx in range(batch_size):
                base_msg = msg_ids_batch[b_idx]
                for beam_idx, beam in enumerate(current_beams[b_idx]):
                    full_prefix = base_msg + beam
                    base_prompts.append(full_prefix)
                    map_info.append((b_idx, beam_idx))
            
            expand_params = {"temperature": 0.0, "max_new_tokens": 1}
            expand_outputs = self.engine.generate(input_ids=base_prompts, sampling_params=expand_params, return_logprob=True,
                top_logprobs_num=width)

            batch_candidates = [[] for _ in range(batch_size)]

            for i, out in enumerate(expand_outputs):
                b_idx, beam_idx = map_info[i]
                
                top_tokens = []
                if 'meta_info' in out and 'output_top_logprobs' in out['meta_info']:

                    top_k_list = out['meta_info']['output_top_logprobs'][0]
                    
                    if top_k_list:
                        for item in top_k_list:
                            # item  (logprob, token_id, text)
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                top_tokens.append(item[1]) 
                
                if top_tokens:
                    base_beam = current_beams[b_idx][beam_idx]
                    for token_id in top_tokens:
                        batch_candidates[b_idx].append(base_beam + [token_id])
                else:
                    # Fallback
                    new_token = out['token_ids'][-1]
                    base_beam = current_beams[b_idx][beam_idx]
                    batch_candidates[b_idx].append(base_beam + [new_token])

            # === B. Score Step ===
            score_prompts = []
            score_map_to_batch = []

            for b_idx in range(batch_size):
                base_msg = msg_ids_batch[b_idx]
                tgt_ids = tgt_ids_batch[b_idx]
                for cand in batch_candidates[b_idx]:
                    score_prompts.append(base_msg + cand + tgt_ids)
                    score_map_to_batch.append(b_idx)

            score_params = {
                "max_new_tokens": 0,
                "temperature": 0.0,
                
            }

            score_outputs = self.engine.generate(input_ids=score_prompts, sampling_params=score_params, return_logprob=True,
                top_logprobs_num=width)

            # scores_per_batch[b_idx] = List[float_scores]
            scores_per_batch = [[] for _ in range(batch_size)]

            for i, out in enumerate(score_outputs):
                b_idx = score_map_to_batch[i]
                tgt_len = len(tgt_ids_batch[b_idx])

                # SGLang Logprob Extraction Logic (Copied & Adapted)
                logprobs_data = None
                if 'input_token_logprobs' in out:
                    logprobs_data = out['input_token_logprobs']
                elif 'meta_info' in out and 'input_token_logprobs' in out['meta_info']:
                    logprobs_data = out['meta_info']['input_token_logprobs']
                
                if logprobs_data is None:
                    scores_per_batch[b_idx].append(100.0)
                    continue

                target_logprobs = logprobs_data[-tgt_len:]
                
                current_loss = 0.0
                valid_cnt = 0 
                
                for item in target_logprobs:
                    val = None
                    if isinstance(item, (tuple, list)):
                        val = item[0]
                    elif isinstance(item, dict):
                        val = list(item.values())[0]
                    else:
                        val = item
                    
                    if val is not None:
                        current_loss -= val # -log(p)
                        valid_cnt += 1
                    else:
                        current_loss += 100.0 # Penalty
                
                scores_per_batch[b_idx].append(current_loss / max(valid_cnt, 1))

            # === C. Selection (TopK) ===
            new_beams = []
            for b_idx in range(batch_size):
                scores_tensor = torch.tensor(scores_per_batch[b_idx])
                
                k = min(beam_size, len(batch_candidates[b_idx]))
                topk_indices = torch.topk(scores_tensor, k=k, largest=False).indices
                
                selected = [batch_candidates[b_idx][idx] for idx in topk_indices.tolist()]
                new_beams.append(selected)

            current_beams = new_beams

        # --- 4. Final Return ---
        final_ids = []
        for b_idx in range(batch_size):
            final_ids.append(current_beams[b_idx][0])
            
        return torch.tensor(final_ids, device=device)
