from typing import Union, Sequence, Dict, List, Tuple
from tqdm import tqdm
import copy
import torch, transformers
import torch.cuda.nvtx as nvtx  # NVTX markers

from utils import (
    initialize_prefix_cache,
    AttackerBase,
    forward_with_cache,
)


class GCG_MEM(AttackerBase):
    """
    A "batch inputs" implementation of the GCG Jailbreaking Attack.

    Reference:
        [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043)
    """
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
        mem_tracker=None,
        **kwargs
    ):
        super().__init__(suffix_length)

        self.steps = steps
        self.topk = topk
        self.search_width = search_width
        self.batch_size = batch_size
        self.width_bs = width_bs
        self.kv_cache = kv_cache
        self.disable_tqdm = disable_tqdm
        self.num = 0
        self.mem_tracker = mem_tracker  # FIX: actually store the tracker

    def _set_phase(self, phase: str):
        """Helper to set phase on mem_tracker if available."""
        if self.mem_tracker is not None:
            self.mem_tracker.set_phase(phase)

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

        nvtx.range_push("GCG::attack_embeds")
        self._set_phase("attack_embeds")

        # ============================================================
        # Phase: Initialization
        # ============================================================
        nvtx.range_push("Phase::Initialization")
        self._set_phase("Initialization")

        # close model autograd for potential speed up
        reqs_grad = []
        for pp in model.parameters():
            reqs_grad.append(pp.requires_grad)
            pp.requires_grad = False
        model.eval()

        # prepare parameters
        vocab_size = model.vocab_size
        embedding_layer = model.get_input_embeddings()
        self.num += 1
        B = len(message_ids)

        topk = self.topk
        sfx_len = self.suffix_length
        width = self.search_width
        grad_bs = B if self.batch_size == -1 else self.batch_size
        qry_bs = width if self.width_bs == -1 else self.width_bs

        nvtx.range_push("Init::EmbeddingLookup")
        self._set_phase("Init::EmbeddingLookup")
        message_embeds = embedding_layer(message_ids)
        msg_len = message_embeds.shape[1]
        message_offset = self._get_prefix_offset_from_mask(message_mask)

        target_embeds = embedding_layer(target_ids)
        tar_len = target_embeds.shape[1]
        target_offset = self._get_suffix_offset_from_mask(target_mask)
        nvtx.range_pop()  # Init::EmbeddingLookup

        L = msg_len + sfx_len + tar_len

        # build ids and mask for advsfx
        advsfx_ids = torch.tensor(
            tokenizer.encode(" x" * (sfx_len + 5), add_special_tokens=False),
            dtype=torch.int64, device=device
        )
        advsfx_ids = advsfx_ids[:sfx_len].unsqueeze(0).expand(B, -1).clone()
        advsfx_mask = torch.ones((B, sfx_len), dtype=torch.int64, device=device)

        # mask that will be used when calculating gradients
        input_mask = torch.cat(
            [message_mask, advsfx_mask, torch.ones_like(target_mask)], dim=1
        )  # shape: [B, L]

        cand_message_offset = message_offset.unsqueeze(0).expand(width, -1).reshape(width * B).contiguous()
        cand_target_offset = target_offset.unsqueeze(0).expand(width, -1).reshape(width * B).contiguous()

        nvtx.range_push("Init::KVCache")
        self._set_phase("Init::KVCache")
        cache = None
        if self.kv_cache != "None":
            cache = initialize_prefix_cache(
                model=model, search_width=self.search_width,
                message_embeds=message_embeds, message_mask=message_mask,
                cache_mode=self.kv_cache, grad_bs=grad_bs, dataset_size=B
            )
        nvtx.range_pop()  # Init::KVCache

        nvtx.range_pop()  # Phase::Initialization

        # ============================================================
        # Main optimization loop
        # ============================================================
        pbar = tqdm(range(self.steps), unit="step", leave=True)

        for step in pbar:
            nvtx.range_push(f"Step::{step}")
            self._set_phase(f"Step::{step}")

            # --------------------------------------------------------
            # Step 1: Gradient computation -> top-k token selection
            # --------------------------------------------------------
            nvtx.range_push("Phase::GradientTopK")
            self._set_phase("GradientTopK")
            topk_ids = []
            for ii, be in enumerate(range(0, B, grad_bs)):
                ed = min(be + grad_bs, B)
                nvtx.range_push(f"GradBatch::{ii}")
                self._set_phase(f"GradBatch::{ii}")

                nvtx.range_push("Grad::OneHotEmbed")
                self._set_phase("Grad::OneHotEmbed")
                adv_onehot = torch.nn.functional.one_hot(
                    advsfx_ids[be:ed], num_classes=vocab_size
                ).to(embedding_layer.weight.dtype)
                adv_onehot.requires_grad = True
                adv_embeds = torch.matmul(adv_onehot, embedding_layer.weight.data)
                nvtx.range_pop()  # Grad::OneHotEmbed

                inp_mask = input_mask[be:ed]
                inp_embeds = torch.cat([adv_embeds, target_embeds[be:ed]], dim=1)

                nvtx.range_push("Grad::Forward")
                self._set_phase("Grad::Forward")
                out_logits = forward_with_cache(
                    model=model, cache_mode=self.kv_cache,
                    cache=cache[ii] if cache is not None else None,
                    message_embeds=message_embeds[be:ed],
                    sfx_tar_embeds=inp_embeds,
                    attention_mask=inp_mask
                )[:, -(tar_len + 1):-1]
                nvtx.range_pop()  # Grad::Forward

                nvtx.range_push("Grad::LossBackward")
                self._set_phase("Grad::LossBackward")
                out_logps = out_logits.log_softmax(dim=-1)
                out_logps = torch.gather(
                    out_logps, dim=-1, index=target_ids[be:ed].unsqueeze(-1)
                ).squeeze(-1)
                loss = -(out_logps * target_mask[be:ed]).mean()
                gd = torch.autograd.grad(loss, adv_onehot)[0]
                nvtx.range_pop()  # Grad::LossBackward

                nvtx.range_push("Grad::TopKSelection")
                self._set_phase("Grad::TopKSelection")
                topk_ids.append(gd.topk(topk, dim=-1, largest=False).indices)
                nvtx.range_pop()  # Grad::TopKSelection

                nvtx.range_pop()  # GradBatch
            topk_ids = torch.cat(topk_ids)  # shape: [B, sfx_len, topk]
            nvtx.range_pop()  # Phase::GradientTopK
            torch.cuda.empty_cache()

            # --------------------------------------------------------
            # Step 2: Candidate sampling
            # --------------------------------------------------------
            nvtx.range_push("Phase::CandidateSampling")
            self._set_phase("CandidateSampling")
            values = topk_ids.unsqueeze(0).expand(width, -1, -1, -1)
            indices = torch.randint(0, topk, (width, B, sfx_len, 1), device=device)
            values = torch.gather(values, dim=-1, index=indices).squeeze(-1)
            indices = torch.randint(0, sfx_len, (width, B, 1), device=device)
            values = torch.gather(values, dim=-1, index=indices)

            cand_advsfx_ids = advsfx_ids.unsqueeze(0).expand(width, -1, -1).clone()
            cand_advsfx_ids.scatter_(dim=-1, index=indices, src=values)
            nvtx.range_pop()  # Phase::CandidateSampling

            # --------------------------------------------------------
            # Step 3: Candidate scoring
            # --------------------------------------------------------
            nvtx.range_push("Phase::CandidateScoring")
            self._set_phase("CandidateScoring")

            score = []
            for ii, be in enumerate(range(0, width, qry_bs)):
                ed = min(be + qry_bs, width)
                nvtx.range_push(f"ScoreBatch_w::{ii}")
                self._set_phase(f"ScoreBatch_w::{ii}")

                nvtx.range_push("Score::EmbedLookup")
                self._set_phase("Score::EmbedLookup")
                cand_advsfx_embeds = embedding_layer(cand_advsfx_ids[be:ed])
                nvtx.range_pop()  # Score::EmbedLookup

                scr_rows = []

                for kk, be2 in enumerate(range(0, B, grad_bs)):
                    ed2 = min(be2 + grad_bs, B)
                    nvtx.range_push(f"ScoreBatch_b::{kk}")
                    self._set_phase(f"ScoreBatch_b::{kk}")

                    nvtx.range_push("Score::BuildCandEmbeds")
                    self._set_phase("Score::BuildCandEmbeds")
                    cand_embeds = torch.cat([
                        message_embeds[be2:ed2].unsqueeze(0).expand(ed - be, -1, -1, -1),
                        cand_advsfx_embeds[:, be2:ed2],
                        target_embeds[be2:ed2].unsqueeze(0).expand(ed - be, -1, -1, -1),
                    ], dim=-2).view((ed - be) * (ed2 - be2), L, -1)

                    cand_mask = input_mask[be2:ed2].unsqueeze(0).expand(
                        ed - be, -1, -1
                    ).reshape((ed - be) * (ed2 - be2), -1).contiguous()
                    nvtx.range_pop()  # Score::BuildCandEmbeds

                    nvtx.range_push("Score::Forward")
                    self._set_phase("Score::Forward")
                    out_logits = forward_with_cache(
                        model=model, cache_mode=self.kv_cache,
                        cache=cache[kk] if cache is not None else None,
                        sfx_tar_embeds=cand_embeds[:, msg_len:],
                        full_embeds=cand_embeds,
                        attention_mask=cand_mask,
                        expand_factor=ed - be
                    )[:, -(tar_len + 1):-1]
                    nvtx.range_pop()  # Score::Forward

                    nvtx.range_push("Score::LossCompute")
                    self._set_phase("Score::LossCompute")
                    cand_target_ids = target_ids[be2:ed2].unsqueeze(0).expand(ed - be, -1, -1)
                    cand_target_mask = target_mask[be2:ed2].unsqueeze(0).expand(ed - be, -1, -1)

                    out_logits = out_logits.view(ed - be, ed2 - be2, *out_logits.shape[1:])
                    out_logps = out_logits.log_softmax(dim=-1)
                    out_logps = torch.gather(
                        out_logps, dim=-1, index=cand_target_ids.unsqueeze(-1)
                    ).squeeze(-1)

                    scr = -(out_logps * cand_target_mask).mean(dim=-1)
                    scr_rows.append(scr)
                    nvtx.range_pop()  # Score::LossCompute

                    nvtx.range_pop()  # ScoreBatch_b

                scr_rows = torch.cat(scr_rows, dim=1)
                score.append(scr_rows)

                nvtx.range_pop()  # ScoreBatch_w

            score = torch.cat(score, dim=0).view(width * B)
            nvtx.range_pop()  # Phase::CandidateScoring

            torch.cuda.empty_cache()

            # --------------------------------------------------------
            # Step 4: Filtering & Update
            # --------------------------------------------------------
            nvtx.range_push("Phase::FilterAndUpdate")
            self._set_phase("FilterAndUpdate")

            nvtx.range_push("Update::Filter")
            self._set_phase("Update::Filter")
            cand_ids = torch.cat([
                message_ids.unsqueeze(0).expand(width, -1, -1),
                cand_advsfx_ids,
                target_ids.unsqueeze(0).expand(width, -1, -1),
            ], dim=-1)
            cand_ids = cand_ids.view(width * B, L)
            fltr_msk = self._get_filter_mask(
                tokenizer, cand_ids, cand_message_offset, cand_target_offset
            )
            nvtx.range_pop()  # Update::Filter

            nvtx.range_push("Update::SelectBest")
            self._set_phase("Update::SelectBest")
            score = (score + fltr_msk * 1e6).view(width, B)
            ind = score.argmin(dim=0, keepdim=True)

            fin_fltr_msk = (score.min(dim=0).values > 1e5).unsqueeze(-1)
            advsfx_ids_old = advsfx_ids.clone()

            upd_indices = torch.gather(
                indices.squeeze(-1), dim=0, index=ind
            ).squeeze(0).unsqueeze(-1)
            upd_values = torch.gather(
                values.squeeze(-1), dim=0, index=ind
            ).squeeze(0).unsqueeze(-1)
            advsfx_ids.scatter_(dim=-1, index=upd_indices, src=upd_values)
            advsfx_ids = advsfx_ids * (~fin_fltr_msk) + advsfx_ids_old * fin_fltr_msk
            nvtx.range_pop()  # Update::SelectBest

            nvtx.range_pop()  # Phase::FilterAndUpdate

            nvtx.range_pop()  # Step

        # re-open model autograd
        for pp, rq_gd in zip(model.parameters(), reqs_grad):
            pp.requires_grad = rq_gd

        nvtx.range_pop()  # GCG::attack_embeds

        return advsfx_ids