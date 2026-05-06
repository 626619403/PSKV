from .gcg import GCG
from .gcq import GCQ
from .gcg_memory_test import GCG_MEM
from .beast import BEAST
from .autodan_zhu import AutoDANZhu

from .AmpleGCG import AmpleGCG
from .advPrompter import AdvPrompterOpt
from .beast_sglang import BEAST_SGLang
from .beast_vllm import BEAST_VLLM

__attacker_zoo__ = {
    "gcg"              : GCG,
    "gcq"              : GCQ,
    "beast"            : BEAST,
    "autodan-zhu"      : AutoDANZhu,
    "ample-gcg"        : AmpleGCG,
    "adv-prompter"     : AdvPrompterOpt,
    "beast_sglang"     : BEAST_SGLang,
    "beast_vllm"       : BEAST_VLLM,
    "gcg_mem"          : GCG_MEM,
}

def build_attacker(name: str, **kwargs):
    return __attacker_zoo__[name](**kwargs)
