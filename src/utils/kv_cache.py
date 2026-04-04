from __future__ import annotations

from typing import Optional, Union, Sequence, Dict, Any, Tuple, List
import copy
import transformers, torch
import pdb
from transformers.cache_utils import DynamicCache, DynamicLayer

def normalize_cache_mode(mode: Optional[Union[str, bool]]) -> str:
    #Do not use this unless you want to change the cache mode format for whole project.
    """
    Convert various user-facing cache configuration values into a canonical string.

    Args:
        mode: Value provided by configuration/CLI. Accepts strings ("ours", "normal",
            "none", case-insensitive) or booleans (True -> "normal", False -> "none").

    Returns:
        One of {"ours", "normal", "none"}.
    """
    if isinstance(mode, bool):
        return "normal" if mode else "none"
    if mode is None:
        return "none"
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized in {"ours", "normal", "none"}:
            return normalized
        if normalized in {"true", "enabled"}:
            return "normal"
        if normalized in {"false", "disabled"}:
            return "none"
    raise ValueError(f"Unknown KV-cache mode: {mode!r}")

class BaseCacheLayer(DynamicLayer):
    """
    Base class for PSKV Layer.
    """
    def __init__(self):
        super().__init__()
        self.expand_num = None
        self.clip = None
        self.pseudo_cache_len = None
        self.prevent_update = True
        self.is_initialized = False

    def initialize_layer(self, key_states: torch.Tensor, value_states: torch.Tensor):
        raise NotImplementedError("Subclasses must implement initialize_layer")

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
        raise NotImplementedError("Subclasses must implement update")

    def get_seq_length(self) -> int:
        """Helper to get current sequence length safely"""
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]


class BaseCache(DynamicCache):
    """ 
    Base class for KV-cache used in PSKV attacks (Compatible with Transformers >= 4.36).
    Acts as a controller for a list of BaseCacheLayer objects.
    """
    def __init__(self):
        super().__init__()

    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states.
        Delegates to the specific layer.
        """
        if layer_idx < len(self.layers):

            layer = self.layers[layer_idx]
            if getattr(layer, "pseudo_cache_len", None) is not None:
                return layer.pseudo_cache_len
            return layer.get_seq_length()
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None
    
    def set_expand(self, expand_num: int = None):
        for layer in self.layers:
            if hasattr(layer, "expand_num"):
                layer.expand_num = expand_num

    def set_pseudo_cache_len(self, pseudo_cache_len: int = None):
        for layer in self.layers:
            if hasattr(layer, "pseudo_cache_len"):
                layer.pseudo_cache_len = pseudo_cache_len

    def set_clip(self, clip_len: int = None):
        for layer in self.layers:
            if hasattr(layer, "clip"):
                layer.clip = clip_len
    
    def set_prevent_update(self, prevent: bool):
        """Enable/Disable read-only mode for prefix caching"""
        for layer in self.layers:
            if hasattr(layer, "prevent_update"):
                layer.prevent_update = prevent


    def initialize_layer(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        """
        Initializes a specific layer. 
        Note: Subclasses (StandardCache) usually implement mechanisms (like __getitem__) 
        to auto-create layers if they don't exist.
        """
        self[layer_idx].initialize_layer(key_states, value_states)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """
        Implementation of Beam Search reordering.
        Iterates over all initialized layers and reorders keys/values.
        """
        for layer in self.layers:
            if not layer.is_initialized:
                continue
                
            device = layer.keys.device
            beam_idx = beam_idx.to(device)
            
            if layer.keys.shape[0] == beam_idx.shape[0]:
                layer.keys = layer.keys.index_select(0, beam_idx)
                layer.values = layer.values.index_select(0, beam_idx)
            else:
                pass

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        """
        Return (kv_length, kv_offset) for attention mask preparation.
        Compatible with Transformers dynamic mask logic.
        """
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        
        layer = self.layers[layer_idx]

        kv_length = layer.get_seq_length() + cache_position.shape[0]
        return kv_length, 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self[layer_idx]

        k, v = layer.update(key_states, value_states, cache_kwargs)
        
        return k, v
    
    def __getitem__(self, layer_idx: int) -> BaseCacheLayer:
        """
        Retrieve the layer object. 
        Subclasses should override this if they support lazy initialization of specific layer types.
        """
        if layer_idx < len(self.layers):
            return self.layers[layer_idx]
        raise KeyError(f"Layer index {layer_idx} out of range (current layers: {len(self.layers)})")

    def __iter__(self):
        """
        Iterate over layers. 
        Transformers expects this to yield Layer objects in recent versions, 
        or (key, value) tuples in older legacy paths. 
        Since we inherit DynamicCache, we rely on its behavior or yield layers.
        """
        for layer in self.layers:
            yield layer

    def __len__(self):
        return len(self.layers)
    

class StandardCache(BaseCache):
    
    def __init__(self, search_width: int = 1):
        super().__init__()
        self.search_width = search_width
        self.layers: List[StandardLayer] = [] 

    def __getitem__(self, layer_idx: int) -> StandardLayer:
        if layer_idx < len(self.layers):
            return self.layers[layer_idx]
        else:
            for i in range(len(self.layers), layer_idx + 1):
                self.layers.append(StandardLayer(search_width=self.search_width))
            return self.layers[layer_idx]
        
class ExpandablePrefixCache(BaseCache):
    
    def __init__(self):
        super().__init__()
        self.layers: List[ExpandablePrefixLayer] = []
    
    def __getitem__(self, layer_idx: int) -> ExpandablePrefixLayer:
        if layer_idx < len(self.layers):
            return self.layers[layer_idx]
        else:
            for i in range(len(self.layers), layer_idx + 1):
                self.layers.append(ExpandablePrefixLayer())
            return self.layers[layer_idx]
            
class StandardLayer(BaseCacheLayer):
    def __init__(self, search_width: int = 1):
        super().__init__()
        self.search_width = search_width
        self.expand_num = None
        self.clip = None
        self.pseudo_cache_len = None
        
        self.prevent_update = True 

    def initialize_layer(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        
        self.keys = (
            key_states.unsqueeze(0)
            .expand(self.search_width, -1, -1, -1, -1)
            .reshape(-1, *key_states.shape[1:])
            .contiguous()
        )
        self.values = (
            value_states.unsqueeze(0)
            .expand(self.search_width, -1, -1, -1, -1)
            .reshape(-1, *value_states.shape[1:])
            .contiguous()
        )
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.keys = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
            self.values = torch.tensor([], dtype=value_states.dtype, device=value_states.device)
            self.is_initialized = True

        keys = self.keys
        values = self.values
        
        B = keys.shape[0] // self.search_width if self.search_width else keys.shape[0]
        if self.pseudo_cache_len is not None:
            keys = keys[:, :, : self.pseudo_cache_len, :]
            values = values[:, :, : self.pseudo_cache_len, :]
            
        if self.expand_num is None or self.expand_num == 1:
            keys = keys[:B]; values = values[:B]

        elif self.expand_num < self.search_width:
            end_idx = self.expand_num * B
            keys = keys[:end_idx]; values = values[:end_idx]
            
        out_keys = torch.cat([keys, key_states], dim=-2)
        out_values = torch.cat([values, value_states], dim=-2)


        if not self.prevent_update:
            self.keys = out_keys
            self.values = out_values
            
        if (self.clip is not None) and (self.expand_num is None):
            self.keys = self.keys[:, :, : self.clip, :].contiguous()
            self.values = self.values[:, :, : self.clip, :].contiguous()

        return out_keys, out_values
        
class ExpandablePrefixLayer(BaseCacheLayer):
    def __init__(self):
        super().__init__()
        self.expand_num = None
        self.clip = None
        self.pseudo_cache_len = None
        self.prevent_update = True # [NEW]

    def initialize_layer(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = key_states.clone()
        self.values = value_states.clone()
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.keys = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
            self.values = torch.tensor([], dtype=value_states.dtype, device=value_states.device)
            self.is_initialized = True

        keys = self.keys
        values = self.values

        if self.pseudo_cache_len is not None:
            keys = keys[:, :, : self.pseudo_cache_len, :]
            values = values[:, :, : self.pseudo_cache_len, :]
        if self.expand_num is not None:
            newB = self.expand_num * keys.shape[0]
            if newB == key_states.shape[0]:
                keys = keys.unsqueeze(0).expand(self.expand_num, *(len(keys.shape)*[-1,]))
                keys = keys.reshape(newB, *keys.shape[2:])
                values = values.unsqueeze(0).expand(self.expand_num, *(len(values.shape)*[-1,]))
                values = values.reshape(newB, *values.shape[2:])

        out_keys = torch.cat([keys, key_states], dim=-2)
        out_values = torch.cat([values, value_states], dim=-2)
            
        if (self.clip is not None) and (self.expand_num is None):
            self.keys = self.keys[:, :, : self.clip, :]
            self.values = self.values[:, :, : self.clip, :]
        
        return out_keys, out_values

def initialize_prefix_cache(
    model: Optional[transformers.PreTrainedModel] = None,
    message_embeds: Optional[torch.Tensor] = None,
    message_ids: Optional[torch.Tensor] = None,
    message_mask: Optional[torch.Tensor] = None,
    cache_mode: Optional[str]= "None",
    dataset_size: Optional[int] = None,
    grad_bs: Optional[str] = None,
    search_width: Optional[int] = None,
    raw_cache: Optional[Any] = None,
):
    if cache_mode == "None":
        return None
    if raw_cache is None:
        if message_ids is not None:
            outputs = model(
                input_ids=message_ids,
                attention_mask=message_mask,
                use_cache=True,
            )
        elif message_embeds is not None:
            outputs = model(
                inputs_embeds=message_embeds,
                attention_mask=message_mask,
                use_cache=True,
            )
        else:
            raise ValueError("At least one of message_ids or message_embeds must be provided.")
        raw_cache = outputs.past_key_values

    if cache_mode == "Ours":
        cache = [ExpandablePrefixCache() for ii in range(0, dataset_size, grad_bs)] if grad_bs is not None else ExpandablePrefixCache()
    else:
        cache = [StandardCache(search_width=search_width) for ii in range(0, dataset_size, grad_bs)] if grad_bs is not None else StandardCache(search_width=search_width)

    # [FIXED ITERATION LOGIC]
    
    num_layers = len(raw_cache)
    for idx in range(num_layers):
        item = raw_cache[idx]
        if isinstance(item, (tuple, list)):
            key_states, value_states = item
        elif hasattr(item, "keys") and hasattr(item, "values"): 
            key_states, value_states = item.keys, item.values
        else:
             key_states, value_states = item

        if grad_bs is not None:
            for cache_idx, be in enumerate(range(0, dataset_size, grad_bs)):
                ed = min(be + grad_bs, dataset_size)
                cache[cache_idx].initialize_layer(key_states[be:ed], value_states[be:ed], idx)
        else:
            cache.initialize_layer(key_states, value_states, idx)

    if isinstance(cache, list):
        for c in cache:
            if hasattr(c, "set_prevent_update"):
                c.set_prevent_update(True)
    elif hasattr(cache, "set_prevent_update"):
        cache.set_prevent_update(True)

    return cache

def forward_with_cache(
    model: transformers.PreTrainedModel,
    cache_mode: str,
    cache: Any, # BaseCache or List[BaseCache]
    attention_mask: torch.Tensor,
    sfx_tar_embeds: Optional[torch.Tensor]= None,
    message_embeds: Optional[torch.Tensor]= None,
    sfx_tar_ids: Optional[torch.Tensor]= None,
    message_ids: Optional[torch.Tensor]= None,
    expand_factor: Optional[int] = None,
    clip_len: Optional[int] = None,
    pseudo_cache_len: Optional[int] = None,
    full_embeds: Optional[torch.Tensor] = None,
    full_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''Forward pass with cache handling. Input embeddings': [B, grad_bs, L, D]'''
    
    if sfx_tar_embeds is not None:
        current_input = sfx_tar_embeds
        seq_len = sfx_tar_embeds.shape[1]
    elif sfx_tar_ids is not None:
        current_input = sfx_tar_ids
        seq_len = sfx_tar_ids.shape[1]
    else:
        current_input = None
        seq_len = 0 

    if current_input is not None:
        bsz = current_input.shape[0]
        device = current_input.device
    else:
        bsz = full_ids.shape[0] if full_ids is not None else full_embeds.shape[0]
        device = model.device

    if cache_mode != "None":
        cache_list = cache if isinstance(cache, list) else [cache]
        if message_embeds is None and message_ids is None:
            if full_embeds is not None:
                message_embeds = full_embeds[:, :full_embeds.shape[1]-sfx_tar_embeds.shape[1], :]
            elif full_ids is not None:
                message_ids = full_ids[:, :full_ids.shape[1]-sfx_tar_ids.shape[1]]
        for c in cache_list:
            c.set_expand(expand_factor)
            c.set_clip(clip_len)
            c.set_pseudo_cache_len(pseudo_cache_len)
            c.set_prevent_update(True) 

        input_cache = cache 
        
        if message_embeds is not None:
            logits = model(
                inputs_embeds=sfx_tar_embeds,
                attention_mask=attention_mask,
                past_key_values=input_cache,
                use_cache=True,
            )["logits"]
        else:
            logits = model(
                input_ids=sfx_tar_ids,
                attention_mask=attention_mask,
                past_key_values=input_cache,
                use_cache=True,
            )["logits"]

        for c in cache_list:
            c.set_expand(None)
            c.set_clip(None)
            c.set_pseudo_cache_len(None)

            c.set_prevent_update(True) 
            
        return logits
    
    if full_embeds is None and message_embeds is not None:
        full_embeds = torch.cat([message_embeds, sfx_tar_embeds], dim=1)
    elif full_ids is None and message_ids is not None:
        full_ids = torch.cat([message_ids, sfx_tar_ids], dim=1)
        
    if full_embeds is not None:
        out_logits = model(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )["logits"]
    else:
        out_logits = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )["logits"]
        
    return out_logits  

def cache_enabled(mode: Optional[Union[str, bool]]) -> bool:
    """Return ``True`` if the cache mode requires prefix reuse."""
    return normalize_cache_mode(mode) != "None"

class AttackerBase:
    """
    A "batch inputs" implementation of the GCG Jailbreaking Attack.

    Reference:
        [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043)
    """
    def __init__(self, suffix_length: int):
        self.suffix_length = suffix_length

    def attack(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        message: Union[str, Sequence[Dict], Sequence[Union[str, Sequence[Dict]]]],
        target: Union[str, Sequence[str]],
        device: Union[str, torch.device],
        **attack_kwargs,
    ):
        if (
            isinstance(message, str)
            or (isinstance(message, Sequence) and isinstance(message[0], Dict))
        ):
            message = [message,]
        if isinstance(target, str):
            target = [target,]
        assert len(message) == len(target)

        if isinstance(message[0], Sequence) and isinstance(message[0][0], Dict):
            raise NotImplementedError("Chatting-inputs-processing still has bugs")

        pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

        message = copy.deepcopy(message)
        target = copy.deepcopy(target)

        ids_mask_dict = self._build_ids_and_mask(tokenizer, message, target, device=device, pad_id=pad_id)
        advsfx_ids = self.attack_embeds(model, tokenizer, device=device, **ids_mask_dict, **attack_kwargs)

        adv_message = []
        adv_suffix = []
        for msg, a_ids in zip(message, advsfx_ids):
            a_text = tokenizer.decode(a_ids)
            if isinstance(msg, str):
                msg += a_text
            else:
                msg[-1]["content"] += a_text
            adv_message.append(msg)
            adv_suffix.append(a_text)

        return {
            "adv_message": adv_message,
            "adv_suffix": adv_suffix,
        }

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
        raise NotImplementedError(f"Please implement attack_embeds() method for {self.__class__}")

    """TODO: Check systemPrompt2 mask"""
    @staticmethod
    def _build_ids_and_mask(
        tokenizer: transformers.PreTrainedTokenizerBase,
        message: Sequence[Union[str, Sequence[Dict]]],
        target: Sequence[str],
        device: Union[str, torch.device],
        pad_id: int,
    ) -> Dict[str, Union[None, torch.Tensor]]:

        placeholder = "{advsfx}"
        before_list, after_list = [], []
        for x in message:
            if isinstance(x, str):
                before = tokenizer.encode(x, add_special_tokens=True)
                after = []
            else:
                if "content" not in x[-1].keys():
                    x[-1]["content"] = ""
                old_content = x[-1]["content"]
                x[-1]["content"] += placeholder
                text = tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
                x[-1]["content"] = old_content
                before, after = text.split(placeholder)
                before = tokenizer.encode(before, add_special_tokens=False)
                after = tokenizer.encode(after, add_special_tokens=False)

            before_list.append(before)
            after_list.append(after)

        B = len(message)
        before_len = max(len(ids) for ids in before_list)
        after_len = max(len(ids) for ids in after_list)

        message_ids = torch.full((B, before_len), fill_value=pad_id, dtype=torch.int64, device=device)
        message_mask = torch.zeros_like(message_ids)
        for ii, ids in enumerate(before_list):
            message_ids[ii, -len(ids) : ] = torch.tensor(ids, device=device)
            message_mask[ii, -len(ids) : ] = 1

        target_list = [tokenizer.encode(text, add_special_tokens=False) for text in target]
        tar_len = max(len(ids) for ids in target_list)
        target_ids = torch.full((B, after_len+tar_len), fill_value=pad_id, dtype=torch.int64, device=device)
        target_mask = torch.zeros_like(target_ids)
        for ii, (aft_ids, tar_ids) in enumerate(zip(after_list, target_list)):
            aft_ids = torch.tensor(aft_ids, dtype=torch.int64, device=device)
            tar_ids = torch.tensor(tar_ids, dtype=torch.int64, device=device)
            ids = torch.cat([aft_ids, tar_ids])
            target_ids[ii, : len(ids)] = ids
            target_mask[ii, len(aft_ids) : len(ids)] = 1

        return {
            "message_ids"  : message_ids,
            "message_mask" : message_mask,
            "target_ids"   : target_ids,
            "target_mask"  : target_mask,
        }

    @staticmethod
    def _get_prefix_offset_from_mask(mask: torch.Tensor):
        # mask should be in the shape [..., L]
        L = mask.shape[-1]
        filled_mask = (mask.cumsum(dim=-1) > 0)
        offset = L - filled_mask.sum(dim=-1)
        return offset

    @staticmethod
    def _get_suffix_offset_from_mask(mask: torch.Tensor):
        # mask should be in the shape [..., L]
        L = mask.shape[-1]
        filled_mask = ((mask.sum(dim=-1, keepdim=True) - mask.cumsum(dim=-1) + mask) > 0)
        offset = L - filled_mask.sum(dim=-1)
        return offset

    @staticmethod
    def _get_filter_mask(
        tokenizer: transformers.PreTrainedTokenizerBase,
        ids: torch.Tensor,
        prefix_offset: torch.Tensor,
        suffix_offset: torch.Tensor,
    ):
        B, L = len(ids), ids.shape[-1]
        device = ids.device
        re_ids = ids.clone()
        for i in range(B):
            pfx_off = prefix_offset[i].item()
            sfx_off = suffix_offset[i].item()

            txt = tokenizer.decode(ids[i, pfx_off+1 : L-sfx_off])
            rid = torch.tensor(tokenizer.encode(txt, add_special_tokens=False), device=device)
            if pfx_off+1 + len(rid) > L:
                continue

            re_ids[i, pfx_off+1 : L-sfx_off] = 0
            re_ids[i, pfx_off+1 : pfx_off+1 + len(rid)] = rid

        mask = ((ids != re_ids).sum(dim=-1) != 0)

        return mask

    @torch.no_grad()
    @staticmethod
    def _verify_loss(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        message: Union[str, Sequence[Dict]],
        target: str,
        device: Union[str, torch.device],
    ):
        if isinstance(message, str):
            m_ids = tokenizer(message, add_special_tokens=True).input_ids
        else:
            m_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True)
        t_ids = tokenizer(target, add_special_tokens=False).input_ids

        m_ids = torch.tensor(m_ids, device=device)
        t_ids = torch.tensor(t_ids, device=device)
        ids = torch.cat([m_ids, t_ids])
        out_logits = model(input_ids=ids.unsqueeze(0)).logits.squeeze(0)

        out_logits = out_logits[-(len(t_ids)+1) : -1]
        out_logps = out_logits.log_softmax(-1)
        out_logps = torch.gather(out_logps, dim=-1, index=t_ids.unsqueeze(-1)).squeeze(-1)
        loss = -out_logps.mean()

        return loss.item()
