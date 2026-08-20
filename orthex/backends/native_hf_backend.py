from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orthex.backends.base import Backend
from orthex.backends.registry import register


@register("native_hf")
class NativeHFBackend(Backend):
    """Forward hooks directly on AutoModelForCausalLM, single-device
    placement. `device_map="auto"` is deliberately avoided: it routes the
    model through accelerate's dispatch hooks, which can interfere with our
    own forward_pre_hook/forward_hook registration and leaves parameters as
    meta/offloaded tensors, breaking the in-place `.data` mutation used by
    ablation. All v1 target models comfortably fit on one GPU."""

    def __init__(self, model_id: str, dtype: str = "bfloat16"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype, device_map={"": device},
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding makes the last real token always sit at index -1,
        # so capture/generation code never needs per-sequence index math.
        self.tokenizer.padding_side = "left"
        self.device = device
