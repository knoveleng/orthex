from __future__ import annotations

import torch

from orthex.capture.types import LayerSite
from orthex.generation import format_prompt


class _HookedCapture:
    """resid_pre = forward_pre_hook on a DecoderLayer (its input);
    resid_post = forward_hook on a DecoderLayer (its output). Plain
    nn.Module hooks only see a module's input/output, not intermediate
    values inside its forward() -- resid_mid is intentionally unsupported."""

    def __init__(self, layers, sites: list[str], position: int):
        self.layers = layers
        self.sites = sites
        self.position = position
        self.handles = []
        self.batch_pre: dict[int, torch.Tensor] = {}
        self.batch_post: dict[int, torch.Tensor] = {}

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            if "resid_pre" in self.sites:
                self.handles.append(
                    layer.register_forward_pre_hook(self._pre_hook(i), with_kwargs=True)
                )
            if "resid_post" in self.sites:
                self.handles.append(layer.register_forward_hook(self._post_hook(i)))
        return self

    def __exit__(self, *exc_info):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _pre_hook(self, i: int):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            self.batch_pre[i] = hidden[:, self.position, :].detach()

        return hook

    def _post_hook(self, i: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.batch_post[i] = hidden[:, self.position, :].detach()

        return hook


def capture_activations(
    model,
    tokenizer,
    layers,
    prompts: list[str],
    sites: list[str],
    position: int = -1,
    batch_size: int = 16,
) -> dict[LayerSite, torch.Tensor]:
    """Runs `prompts` through `model` in batches, capturing the activation at
    `position` for every DecoderLayer in `layers` at each requested site.
    Requires tokenizer.padding_side == "left" so `position=-1` always lands
    on the last real (non-pad) token for every sequence in a batch. Prompts
    are wrapped in the chat template (add_generation_prompt=True) so that
    last-token position is the same pre-generation position used by
    generation.py's generate_completions -- otherwise the extracted
    direction would come from an unrelated position in the raw prompt."""
    accum: dict[LayerSite, list[torch.Tensor]] = {}
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = [format_prompt(tokenizer, p) for p in prompts[start : start + batch_size]]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False
            ).to(device)
            with _HookedCapture(layers, sites, position) as cap:
                model(**enc)
            for i in range(len(layers)):
                if "resid_pre" in sites:
                    key = LayerSite(i, "resid_pre")
                    accum.setdefault(key, []).append(cap.batch_pre[i].to("cpu", torch.float32))
                if "resid_post" in sites:
                    key = LayerSite(i, "resid_post")
                    accum.setdefault(key, []).append(cap.batch_post[i].to("cpu", torch.float32))
    return {key: torch.cat(tensors, dim=0) for key, tensors in accum.items()}
