from __future__ import annotations

from orthex.architectures.registry import register


@register("gemma3")
class Gemma3Adapter:
    """Gemma3ForConditionalGeneration is multimodal: model.model is a
    Gemma3Model wrapping .language_model (a Gemma3TextModel), one level
    deeper than gemma2's plain model.model.*. Confirmed directly against
    installed modeling_gemma3.py. Only the text decoder is ever touched --
    the vision_tower/multi_modal_projector are untouched and irrelevant for
    text-only refusal removal."""

    def embed_module(self, model):
        return model.model.language_model.embed_tokens

    def decoder_layers(self, model):
        return list(model.model.language_model.layers)

    def layer_write_modules(self, model):
        return [
            (layer.self_attn.o_proj, layer.mlp.down_proj)
            for layer in model.model.language_model.layers
        ]
