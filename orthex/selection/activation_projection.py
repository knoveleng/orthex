from __future__ import annotations

from contextlib import contextmanager

import torch

# Eval-time-only tool: projects a candidate direction out of the LIVE
# activation at generation time, so candidates can be cheaply compared
# without mutating any weights. This is explicitly NOT a shipping ablation
# mode. The math (h' = h - (h.d)d) is the same projection used by
# weight_orthogonalizer, just applied per-token at inference time instead
# of baked into the weights.


def _project_out(h: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    d = d.to(dtype=h.dtype, device=h.device)
    coeff = (h @ d).unsqueeze(-1)
    return h - coeff * d


@contextmanager
def project_direction(layer, site: str, direction: torch.Tensor):
    if site == "resid_pre":

        def pre_hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            hidden = _project_out(hidden, direction)
            if args:
                return (hidden,) + args[1:], kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hidden
            return args, kwargs

        handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    elif site == "resid_post":

        def post_hook(module, args, output):
            if isinstance(output, tuple):
                return (_project_out(output[0], direction),) + output[1:]
            return _project_out(output, direction)

        handle = layer.register_forward_hook(post_hook)
    else:
        raise ValueError(f"Unsupported site for activation projection: {site!r}")

    try:
        yield
    finally:
        handle.remove()
