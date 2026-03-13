from typing import Union, Sequence, Dict, Tuple, Optional
from tqdm import tqdm
import copy
import torch, transformers

from utils import (
    ExpandablePrefixCache,
    AttackerBase,
    initialize_prefix_cache,
    forward_with_cache,
)


class BEAST(AttackerBase):
    """
    Reference:
        [Fast Adversarial Attacks on Language Models In One GPU Minute](https://arxiv.org/pdf/2402.15570)
    """
    def __init__(
        self,
        suffix_length: int,
        beam_size: int,
        search_width: int,
        width_bs: int = -1,
        kv_cache: str = "None",
        disable_tqdm: bool = False,
        **kwargs
    ):
        super().__init__(suffix_length)

        self.beam_size = beam_size
        self.search_width = search_width
        self.width_bs = width_bs
        self.kv_cache = kv_cache
        self.disable_tqdm = disable_tqdm

    @torch.no_grad()
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

        model.eval()

        B = len(message_ids)
        vocab_size=len(tokenizer)
        sfx_len = self.suffix_length
        beam_size = self.beam_size
        width = self.search_width
        width_bs = (beam_size * width) if self.width_bs == -1 else self.width_bs

        msg_len = message_ids.shape[1]
        message_offset = self._get_prefix_offset_from_mask(message_mask)

        tar_len = target_ids.shape[1]
        target_offset = self._get_suffix_offset_from_mask(target_mask)
        embed_layers = model.get_input_embeddings()
        
        L = msg_len + sfx_len + tar_len

        outputs = model(input_ids=message_ids, attention_mask=message_mask, use_cache=True)
        probs = outputs["logits"][:,-1, :vocab_size].softmax(-1)
        advsfx_ids = torch.multinomial(probs, num_samples=beam_size).permute([1, 0]).unsqueeze(-1).contiguous() # shape: [beam_size, B, 1]

        cache = initialize_prefix_cache(model=model, search_width=beam_size * width, 
                        raw_cache=outputs["past_key_values"], 
                        cache_mode=self.kv_cache)

        del outputs

        pbar = tqdm(range(1, sfx_len, 1), disable=self.disable_tqdm)

        for step in pbar:
            
            input_mask = torch.cat([
                message_mask.unsqueeze(0).expand(beam_size, -1, -1),
                torch.ones_like(advsfx_ids),
            ], dim=-1).view(beam_size * B, -1).contiguous() # shape: [beam_size * B, msg_len + step]

            last_logit = forward_with_cache(
                model=model, cache_mode=self.kv_cache, cache=cache,
                message_ids=message_ids.unsqueeze(0).expand(beam_size, -1, -1).reshape(beam_size * B, -1),
                sfx_tar_ids=advsfx_ids.view(beam_size * B, -1),
                attention_mask=input_mask,expand_factor=beam_size,
            )[:, -1, :vocab_size]
            
            last_prob = last_logit.softmax(dim=-1) # shape: [beam_size * B, vocab_size]

            last_ids = torch.multinomial(last_prob, num_samples=width).permute([1, 0]).unsqueeze(-1) # shape: [width, beam_size * B, 1]
            last_ids = last_ids.view(width, beam_size, B, 1)

            cand_advsfx_ids = torch.cat([
                advsfx_ids.unsqueeze(0).expand(width, -1, -1, -1),
                last_ids,
            ], dim=-1).view(width * beam_size, B, step+1) # shape: [width * beam_size, B, step+1]

            width_full = width * beam_size
            score = []
            for be in range(0, width_full, width_bs):
                ed = min(be + width_bs, width_full)

                cand_ids = torch.cat([
                    message_ids.unsqueeze(0).expand(ed-be, -1, -1),
                    cand_advsfx_ids[be:ed],
                    target_ids.unsqueeze(0).expand(ed-be, -1, -1),
                ], dim=-1).view((ed-be) * B, -1).contiguous() # shape: [(ed-be) * B, length]

                cand_mask = torch.cat([
                    message_mask.unsqueeze(0).expand(ed-be, -1, -1),
                    torch.ones((ed-be, B, (step+1) + tar_len), dtype=torch.int64, device=device),
                ], dim=-1).view((ed-be) * B, -1).contiguous() # shape: [(ed-be) * B, msg_len + (step+1) + tar_len]

                out_logits = forward_with_cache(
                        model, self.kv_cache, cache, cand_mask,
                        sfx_tar_ids=cand_ids[:, msg_len :],
                        full_ids=cand_ids,
                        expand_factor=ed-be
                    )

                out_logits = out_logits[:, -(tar_len+1) : -1]
                out_logps = out_logits.log_softmax(dim=-1) # shape: [(ed-be)*B, tar_len, vocab_size]

                cand_target_ids = target_ids.unsqueeze(0).expand(ed-be, -1, -1).reshape((ed-be)*B, -1) # shape: [(ed-be)*B, tar_len]
                cand_target_mask = target_mask.unsqueeze(0).expand(ed-be, -1, -1).reshape((ed-be)*B, -1)

                out_logps = torch.gather(out_logps, dim=-1, index=cand_target_ids.unsqueeze(-1)).squeeze(-1)

                scr = -(out_logps * cand_target_mask).mean(dim=-1)
                cand_msg_offset = message_offset.unsqueeze(0).expand(ed-be, -1).reshape(-1)
                cand_tar_offset = target_offset.unsqueeze(0).expand(ed-be, -1).reshape(-1)
                fltr_msk = self._get_filter_mask(tokenizer, cand_ids, cand_msg_offset, cand_tar_offset)
                del cand_ids, cand_msg_offset, cand_tar_offset, cand_target_ids, cand_target_mask, out_logps, out_logits
                scr = (scr + fltr_msk * 1e6) # shape: [(ed-be)*B,]
                score.append(scr)

            score = torch.cat(score).view(width_full, B)
            score, indices = score.topk(beam_size, dim=0, largest=False) # shape: [beam_size, B]
            indices = indices.unsqueeze(-1).expand(-1, -1, step+1) # shape: [beam_size, B, step+1]

            advsfx_ids = torch.gather(cand_advsfx_ids, dim=0, index=indices) # shape: [beam_size, B, step+1]


        best_ind = score.argmin(dim=0, keepdim=True).unsqueeze(-1).expand(-1, -1, sfx_len) # shape: [1, B, sfx_len]
        advsfx_ids = torch.gather(advsfx_ids, dim=0, index=best_ind).squeeze(0) # shape: [B, sfx_len]

        return advsfx_ids
