from __future__ import annotations

import torch

from orthex.ablation.registry import register


def _orthogonalize_embedding_like(W: torch.Tensor, d: torch.Tensor) -> None:
    """In-place. W: [N, H] where each ROW *is* a residual-stream vector
    directly (e.g. embed_tokens.weight: row i is token i's embedding,
    written straight into the residual stream -- residual axis = axis 1).
    Removes component d from every row: W' = W - (W @ d) dT = W (I - d dT).
    """
    orig_dtype = W.dtype
    W32 = W.to(torch.float32)
    d32 = d.to(dtype=torch.float32, device=W.device)
    proj = W32 @ d32  # [N]
    W32 -= torch.outer(proj, d32)
    W.copy_(W32.to(orig_dtype))


def _orthogonalize_linear_output_like(W: torch.Tensor, d: torch.Tensor) -> None:
    """In-place. W: [out_features, in_features] (nn.Linear storage
    convention, y = x @ W.T) where out_features IS the residual-stream axis
    -- axis 0 -- because this matrix's *output* gets added to the residual
    stream (o_proj, down_proj). This is the mirror image of the embedding
    case: removes d from the row space so dT @ (W @ x) == 0 for any x:
    W' = W - d (dT W) = (I - d dT) W.
    """
    orig_dtype = W.dtype
    W32 = W.to(torch.float32)
    d32 = d.to(dtype=torch.float32, device=W.device)
    proj = d32 @ W32  # [in_features]
    W32 -= torch.outer(d32, proj)
    W.copy_(W32.to(orig_dtype))


def orthogonalize(model, adapter, direction: torch.Tensor, targets: list[str]) -> None:
    # Epsilon-guarded like directions/mean_difference.py's normalization --
    # a degenerate zero-norm direction (e.g. positive/negative activations
    # happening to coincide) must not become a 0/0 NaN that silently
    # corrupts every touched weight.
    d = direction / (direction.norm() + 1e-8)

    if "embed" in targets:
        embed = adapter.embed_module(model)
        _orthogonalize_embedding_like(embed.weight.data, d)
        # Every v1 target model ties embed_tokens/lm_head (confirmed:
        # tie_word_embeddings=True on all of them), meaning lm_head.weight
        # IS embed.weight -- the same nn.Parameter. It is therefore already
        # ablated by the line above. Do NOT also run the linear-output-style
        # formula against lm_head: that is a different-orientation
        # projection and would corrupt the just-ablated tensor. An untied
        # model would leave lm_head un-ablated (no `unembed` ablation
        # target exists in the config schema) -- a known v1 scope limit.

    for o_proj, down_proj in adapter.layer_write_modules(model):
        if "attn_out" in targets:
            _orthogonalize_linear_output_like(o_proj.weight.data, d)
        if "mlp_out" in targets:
            _orthogonalize_linear_output_like(down_proj.weight.data, d)


@register("weight_orthogonalization")
class WeightOrthogonalizationStrategy:
    def apply(self, model, adapter, direction: torch.Tensor, targets: list[str]) -> None:
        orthogonalize(model, adapter, direction, targets)
