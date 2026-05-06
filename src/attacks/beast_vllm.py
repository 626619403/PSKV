import torch
import numpy as np
from typing import Union, List, Optional
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vllm not installed.")

from utils import AttackerBase

class BEAST_VLLM(AttackerBase):
    """
    vLLM accelerated implementation of BEAST attack.
    Uses 'enable_prefix_caching=True' to speed up the beam search tree exploration.
    """
    """Install vLLM separately before using this attacker."""
    def __init__(
        self,
        suffix_length: int,
        beam_size: int,
        search_width: int,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        super().__init__(suffix_length)
        self.beam_size = beam_size
        self.search_width = search_width
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        

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
        """
        Executes the BEAST attack using vLLM.
        Note: Supports batch_size=1 (single prompt) efficiently. 
        For larger batches, vLLM handles it internally, but the logic below iterates per sample 
        or assumes inputs are prepared for vLLM's batching.
        """
        
        
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_prefix_caching=True, 
            trust_remote_code=True,
            enforce_eager=False
        )
        del tokenizer
        
        self.tokenizer = self.llm.get_tokenizer()
        if message_ids.shape[0] > 1:
            print("Warning: BEAST_VLLM current implementation optimizes for Batch=1. Processing only first element.")
            
        batch_size = message_ids.shape[0]
        
        msg_ids_batch = []
        tgt_ids_batch = []
        
        for i in range(batch_size):
            m_len = int(message_mask[i].sum().item())
            t_len = int(target_mask[i].sum().item())
            msg_ids_batch.append(message_ids[i].tolist()[-m_len:])
            tgt_ids_batch.append(target_ids[i].tolist()[:t_len])

        sfx_len = self.suffix_length
        beam_size = self.beam_size
        width = self.search_width

        sfx_len = self.suffix_length
        beam_size = self.beam_size
        width = self.search_width
        
        
        sampling_params_init = SamplingParams(
            n=beam_size, temperature=1.0, max_tokens=1, logprobs=0
        )
        
        
        outputs = self.llm.generate(
            prompt_token_ids=msg_ids_batch,
            sampling_params=sampling_params_init,
            use_tqdm=False
        )

        current_beams = [] 
        for i in range(batch_size):
            batch_beams = []

            for output_seq in outputs[i].outputs:
                batch_beams.append(list(output_seq.token_ids))
            current_beams.append(batch_beams)
            
        # --- 2. Iterative Beam Search ---
        pbar = tqdm(range(1, sfx_len), disable=False, desc="BEAST vLLM")
        
        for step in pbar:

            flatten_prompts_ids = []
            map_idx_to_batch = []   
            map_idx_to_beam_in_batch = []

            for b_idx in range(batch_size):
                base_msg = msg_ids_batch[b_idx]
                for beam_idx, beam in enumerate(current_beams[b_idx]):
                    flatten_prompts_ids.append(base_msg + beam)
                    map_idx_to_batch.append(b_idx)
                    map_idx_to_beam_in_batch.append(beam_idx)

            sampling_params_expand = SamplingParams(
                n=width,
                temperature=1.0,
                max_tokens=1
            )
            
            expand_outputs = self.llm.generate(
                prompt_token_ids=flatten_prompts_ids,
                sampling_params=sampling_params_expand,
                use_tqdm=False
            )
            

            batch_candidates = [[] for _ in range(batch_size)]
            
            for i, output in enumerate(expand_outputs):
                b_idx = map_idx_to_batch[i]
                beam_idx_in_batch = map_idx_to_beam_in_batch[i]
                
                base_beam = current_beams[b_idx][beam_idx_in_batch]
                
                for sample in output.outputs:
                    new_cand = base_beam + list(sample.token_ids)
                    batch_candidates[b_idx].append(new_cand)
            
            
            score_prompts_ids = []
            score_map_to_batch = []
            
            for b_idx in range(batch_size):
                tgt = tgt_ids_batch[b_idx]
                msg = msg_ids_batch[b_idx]
                for cand in batch_candidates[b_idx]:
                    score_prompts_ids.append(msg + cand + tgt)
                    score_map_to_batch.append(b_idx)
            
            sampling_params_score = SamplingParams(
                max_tokens=1, prompt_logprobs=1, temperature=0.0
            )
            
            score_outputs = self.llm.generate(
                prompt_token_ids=score_prompts_ids,
                sampling_params=sampling_params_score,
                use_tqdm=False
            )
            
            # --- Calculate Loss & Group by Batch ---
            # scores_per_batch: [Batch_0_Scores, Batch_1_Scores, ...]
            scores_per_batch = [[] for _ in range(batch_size)]

            for i, output in enumerate(score_outputs):
                b_idx = score_map_to_batch[i]
                tgt_len = len(tgt_ids_batch[b_idx])
            
                full_ids = output.prompt_token_ids
                logprobs_list = output.prompt_logprobs
                
                current_loss_sum = 0.0
                valid_cnt = 0
                

                start_idx = len(full_ids) - tgt_len
                
                for j in range(tgt_len):
                    pos = start_idx + j
                    target_token_id = full_ids[pos]
                    

                    token_logprobs = logprobs_list[pos] 
                    
                    if token_logprobs and target_token_id in token_logprobs:

                        val = token_logprobs[target_token_id]
                        if hasattr(val, 'logprob'):
                            current_loss_sum -= val.logprob # Loss = -log(p)
                        else:
                            current_loss_sum -= val
                        valid_cnt += 1
                    else:
                        current_loss_sum += 100.0 # Penalty
                
                avg_loss = self._calc_loss_vllm(output, tgt_len, tgt_ids_batch[b_idx])
                scores_per_batch[b_idx].append(avg_loss)

            # --- Select Top-k Beams for Each Batch ---
            # scores_per_batch: [Batch_0_Scores, Batch_1_Scores, ...]
            new_beams = []
            for b_idx in range(batch_size):
                scores_tensor = torch.tensor(scores_per_batch[b_idx])
                topk_indices = torch.topk(scores_tensor, k=beam_size, largest=False).indices
                
                selected = [batch_candidates[b_idx][j] for j in topk_indices.tolist()]
                new_beams.append(selected)
            
            current_beams = new_beams
            
        final_ids = []
        for b_idx in range(batch_size):
            final_ids.append(current_beams[b_idx][0])
            
        return torch.tensor(final_ids, device=device)
    
    def _calc_loss_vllm(self, output, tgt_len, tgt_ids):

        full_ids = output.prompt_token_ids
        logprobs_list = output.prompt_logprobs
        current_loss_sum = 0.0
        valid_cnt = 0
        start_idx = len(full_ids) - tgt_len
        
        for j in range(tgt_len):
            pos = start_idx + j
            target_token_id = full_ids[pos]
            token_logprobs = logprobs_list[pos]
            if token_logprobs and target_token_id in token_logprobs:
                 val = token_logprobs[target_token_id]
                 if hasattr(val, 'logprob'): current_loss_sum -= val.logprob
                 else: current_loss_sum -= val
                 valid_cnt += 1
            else:
                 current_loss_sum += 100.0
        return current_loss_sum / max(valid_cnt, 1)
