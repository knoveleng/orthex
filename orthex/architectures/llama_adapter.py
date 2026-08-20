from __future__ import annotations

from orthex.architectures.registry import register


class GenericDecoderAdapter:
    """Llama, Qwen2.x, plain Qwen3, and Gemma2 all share identical module
    paths -- model.model.embed_tokens / model.model.layers[i].self_attn.
    o_proj / model.model.layers[i].mlp.down_proj -- confirmed directly
    against each family's installed modeling_*.py source. One adapter class
    covers all four; registered once per name below so `export.
    architecture_adapter` stays a self-documenting, per-family config value."""

    def embed_module(self, model):
        return model.model.embed_tokens

    def decoder_layers(self, model):
        return list(model.model.layers)

    def layer_write_modules(self, model):
        return [(layer.self_attn.o_proj, layer.mlp.down_proj) for layer in model.model.layers]


register("llama")(GenericDecoderAdapter)
register("qwen2")(GenericDecoderAdapter)
register("qwen3")(GenericDecoderAdapter)
register("gemma2")(GenericDecoderAdapter)
