from typing import Union, Sequence, Dict, List, Tuple, Optional
from tqdm import tqdm
import copy
import torch, transformers

from utils import (
    initialize_prefix_cache,
    AttackerBase,
    forward_with_cache,
)


class GCQ(AttackerBase):
    """
    A "batch inputs" implementation of the GCQ Jailbreaking Attack.

    Reference:
        [Query-Based Adversarial Prompt Generation](https://arxiv.org/pdf/2402.12329)
    """
    def __init__(
        self,
        suffix_length: int,
        steps: int,
        search_width: int,
        top_bq: int,
        beam_size: int = -1,
        width_bs: int = -1,
        kv_cache: str = "None",
        disable_tqdm: bool = False,
        **kwargs
    ):
        super().__init__(suffix_length)

        self.steps = steps
        self.search_width = search_width
        self.top_bq = top_bq
        self.beam_size = beam_size
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
        # prepare parameters
        vocab_size = len(tokenizer)
        B = len(message_ids)

        sfx_len = self.suffix_length
        width = self.search_width
        top_bq = self.top_bq
        beam_size = self.beam_size
        width_bs = (beam_size * width) if self.width_bs == -1 else self.width_bs

        msg_len = message_ids.shape[1]
        message_offset = self._get_prefix_offset_from_mask(message_mask) # shape: [B,]

        tar_len = target_ids.shape[1]
        target_offset = self._get_suffix_offset_from_mask(target_mask)

        L = msg_len + sfx_len + tar_len

        advsfx_ids = torch.tensor(tokenizer.encode(" x" * (sfx_len + 5), add_special_tokens=False), dtype=torch.int64, device=device)
        advsfx_ids = advsfx_ids[: sfx_len].unsqueeze(0).expand(B, -1).clone() # shape: [B, sfx_len]

        advsfx_mask = torch.ones((B, sfx_len), dtype=torch.int64, device=device)

        # mask that will be used when calculating gradients
        input_mask = torch.cat([message_mask, advsfx_mask, torch.ones_like(target_mask)], dim=1) # shape: [B, L]
        
        outputs = model(
            input_ids=torch.cat([message_ids, advsfx_ids, target_ids], dim=-1),
            attention_mask=torch.cat([message_mask, advsfx_mask, target_mask], dim=-1),
        )
        out_logits = outputs["logits"][:, -(tar_len+1) : -1]
        out_logps = out_logits.log_softmax(dim=-1)
        out_logps = torch.gather(out_logps, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        score = -(out_logps * target_mask).mean(dim=-1) # shape: [B, tar_len] -> [B,]
        score = score.unsqueeze(0).expand(beam_size, B).contiguous() # shape: [beam_size, B]
        msg_outputs = model(
            input_ids=message_ids,
            attention_mask=message_mask,
        )
        cache = initialize_prefix_cache(
            raw_cache=msg_outputs["past_key_values"],
            search_width=self.search_width,
            cache_mode=self.kv_cache,
        )
        del msg_outputs

        advsfx_ids = advsfx_ids.unsqueeze(0).expand(beam_size, -1, -1).contiguous()
        advsfx_mask = advsfx_mask.unsqueeze(0).expand(beam_size, -1, -1).contiguous()

        pbar = tqdm(range(self.steps), disable=self.disable_tqdm)

        # for step in range(self.steps):
        for step in pbar:

            ind = score.argmin(dim=0, keepdim=True).unsqueeze(-1).expand(-1, -1, sfx_len) # shape: [1, B, sfx_len]
            p_sfx_ids = torch.gather(advsfx_ids, dim=0, index=ind).expand(width, -1, -1).contiguous() # shape: [width, B, sfx_len]

            rand_indices = torch.randint(0, sfx_len, (width, B, 1), device=device)
            rand_values = torch.randint(0, vocab_size, (width, B, 1), device=device)
            p_sfx_ids.scatter_(dim=-1, index=rand_indices, src=rand_values)

            p_score = []
            for ii, be in enumerate(range(0, width, width_bs)):
                ed = min(be + width_bs, width)

                p_cand_ids = torch.cat([
                    message_ids.unsqueeze(0).expand(ed-be, -1, -1), # shape: [width, B, msg_len]
                    p_sfx_ids[be:ed],
                    target_ids.unsqueeze(0).expand(ed-be, -1, -1),
                ], dim=-1).view((ed-be) * B, -1) # shape: [(ed-be) * B, L]

                cand_mask = input_mask.unsqueeze(0).expand(ed-be, -1, -1).reshape((ed-be) * B, -1)

                out_logits = forward_with_cache(
                        model, cache_mode = self.kv_cache, cache=cache, attention_mask=cand_mask, full_ids=p_cand_ids,
                        sfx_tar_ids=p_cand_ids[:, (msg_len) : ], expand_factor=ed-be)[:, -(tar_len+1) : -1]
                    
                out_logps = out_logits.log_softmax(dim=-1).view((ed-be), B, tar_len, -1)
                tar_ids = target_ids.unsqueeze(0).expand(ed-be, -1, -1) # shape: [(ed-be), B, tar_len]
                tar_mask = target_mask.unsqueeze(0).expand(ed-be, -1, -1) # shape: [(ed-be), B, tar_len]
                out_logps = torch.gather(
                    out_logps, dim=-1, index=tar_ids.unsqueeze(-1),
                ).squeeze(-1) # shape: [(ed-be), B, tar_len]
                p_s = -(out_logps * tar_mask).mean(dim=-1) # shape: [(ed-be), B]

                cand_msg_offset = message_offset.unsqueeze(0).expand(ed-be, -1).reshape(-1) # shape: [(ed-be) * B,]
                cand_tar_offset = target_offset.unsqueeze(0).expand(ed-be, -1).reshape(-1) # shape: [(ed-be) * B,]
                p_fk = self._get_filter_mask(tokenizer, p_cand_ids, cand_msg_offset, cand_tar_offset).view(ed-be, B)
                p_s = (p_s + p_fk * 1e6)

                p_score.append(p_s)

            p_score = torch.cat(p_score, dim=0) # shape: [width, B]

            top_bq_indices = p_score.topk(top_bq, dim=0, largest=False).indices # shape: [top_bq, B]

            p_sfx_ids = torch.gather(
                p_sfx_ids, dim=0, index=top_bq_indices.unsqueeze(-1).expand(-1, -1, sfx_len),
            ) # shape: [top_bq, B, sfx_len]
            p_score = torch.gather(p_score, dim=0, index=top_bq_indices) # shape: [top_bq, B]

            cat_sfx_ids = torch.cat([advsfx_ids, p_sfx_ids], dim=0) # shape: [beam_size + top_bq, B, sfx_len]
            cat_score = torch.cat([score, p_score], dim=0) # shape: [beam_size + top_bq, B]

            top_cat_indices = cat_score.topk(beam_size, dim=0, largest=False).indices # shape: [beam_size, B]
            advsfx_ids = torch.gather(
                cat_sfx_ids, dim=0, index=top_cat_indices.unsqueeze(-1).expand(-1, -1, sfx_len),
            ) # shape: [beam_size, B, sfx_len]
            score = torch.gather(cat_score, dim=0, index=top_cat_indices) # shape: [beam_size, B]

            # print(f"score.min(dim=0) = {score.min(dim=0)}")

        best_ind = score.argmin(dim=0, keepdim=True).unsqueeze(-1).expand(-1, -1, sfx_len) # shape: [1, B, sfx_len]
        advsfx_ids = torch.gather(advsfx_ids, dim=0, index=best_ind).squeeze(0) # shape: [B, sfx_len]
        # print(f"message_ids[0] = {message_ids[0].tolist()}")
        # print(f"advsfx_ids[0] = {advsfx_ids[0].tolist()}")
        # print(f"target_ids[0] = {target_ids[0].tolist()}")

        return advsfx_ids
