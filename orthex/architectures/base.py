from __future__ import annotations

from typing import Protocol

from torch import nn


class ArchitectureAdapter(Protocol):
    """Locates module paths per model family -- given a loaded HF model,
    returns the embedding module and the per-layer (attn_out, mlp_out)
    modules that ablation targets. A locator, not a converter: no format
    conversion is needed since we mutate HF weights in place."""

    def embed_module(self, model) -> nn.Embedding: ...

    def decoder_layers(self, model) -> list[nn.Module]: ...

    def layer_write_modules(self, model) -> list[tuple[nn.Linear, nn.Linear]]:
        """[(o_proj, down_proj) per layer] -- the two per-layer matrices
        whose *output* is added to the residual stream."""
        ...
