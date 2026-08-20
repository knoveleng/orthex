from __future__ import annotations


def _fmt(x: float | None, decimals: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{decimals}f}"


def render(result: dict) -> str:
    """Renders a standard HF model-card README (YAML frontmatter + body)
    from a pipeline.run() result dict. Pure function of `result` -- no
    model/tokenizer access -- so it's testable without loading anything."""
    model_id = result["model_id"]
    adapter = result["architecture_adapter"]
    candidate = result["selected_candidate"]
    refusal = result["evaluation"]["refusal_rate"]
    perplexity = result["evaluation"]["perplexity"]

    return f"""---
base_model: {model_id}
tags:
  - abliterated
  - orthex
---

# {model_id} (abliterated)

This model is a weight-level orthogonalized ("abliterated") version of [`{model_id}`](https://huggingface.co/{model_id}), produced with [orthex](https://github.com/knoveleng/orthex) — an implementation of the technique from [Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717) (NeurIPS 2024).

## What was done

- Architecture adapter: `{adapter}`
- Selected candidate: layer {candidate["layer"]}, site `{candidate["site"]}`
- Ablation targets: `embed_tokens`, and every layer's `attn_out` and `mlp_out` — orthogonalized in place in the weights (not a runtime hook; this checkpoint behaves this way standalone, with no orthex dependency at inference time)

## Evaluation

| metric | pre | post | delta |
|---|---|---|---|
| refusal rate | {_fmt(refusal["pre"])} | {_fmt(refusal["post"])} | {_fmt(refusal["delta"])} |
| perplexity | {_fmt(perplexity["pre"])} | {_fmt(perplexity["post"])} | {_fmt(perplexity["delta"])} |

See `evaluation_report.json` in this repo for the full per-prompt breakdown (`refusal_samples`) and the ranked candidate list considered during selection (`selection_report`).

## Responsible use

This model has had refusal behavior removed and may comply with requests the base model would normally decline. It is intended for red-teaming, robustness research, and model-behavior analysis. Usage remains subject to the base model's original license and usage policy — this repo does not grant any additional rights beyond what `{model_id}`'s license allows.

## License

Not set automatically — inherits obligations from the base model [`{model_id}`](https://huggingface.co/{model_id})'s license; set this field explicitly before publishing.
"""
