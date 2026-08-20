from __future__ import annotations

from orthex.architectures.registry import register


@register("qwen3_5")
class Qwen3_5Adapter:
    """Qwen/Qwen3.5-* checkpoints are multimodal, but transformers registers
    a text-only Qwen3_5ForCausalLM under MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    ("VLM compatibility") -- AutoModelForCausalLM.from_pretrained resolves
    to it directly, ignoring the vision tower. Structure is then flat, same
    as Llama/Qwen2/Qwen3/Gemma2: model.model.embed_tokens / model.model.
    layers[i]. The wrinkle is the hybrid decoder layer (confirmed against
    installed modeling_qwen3_5.py): only every 4th layer (full_attention_
    interval) has self_attn.o_proj; the rest use a Qwen3_5GatedDeltaNet
    linear-attention block whose linear_attn.out_proj plays the identical
    "writes to the residual stream" role -- same shape convention
    ([hidden_size, intermediate], out_features=hidden_size), same
    orthogonalization formula applies."""

    def embed_module(self, model):
        return model.model.embed_tokens

    def decoder_layers(self, model):
        return list(model.model.layers)

    def layer_write_modules(self, model):
        modules = []
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                attn_out = layer.self_attn.o_proj
            elif hasattr(layer, "linear_attn"):
                attn_out = layer.linear_attn.out_proj
            else:
                raise AttributeError(f"Qwen3.5 decoder layer has neither self_attn nor linear_attn: {layer}")
            modules.append((attn_out, layer.mlp.down_proj))
        return modules
