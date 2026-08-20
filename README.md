# orthex

**Weight-level orthogonalization (abliteration) toolkit for Hugging Face causal LMs.**

Implements the Arditi et al. "refusal direction" technique directly against `AutoModelForCausalLM` weights — no TransformerLens conversion step, no intermediate hooked-model format.


## Features

- **Config-driven pipeline**, one CLI, one config file — no per-model boilerplate.
- **Registry-based extensibility**: new data source, direction strategy, selection scorer, or model architecture is a new file + one registry line.
- **Weight-level, not activation-level** — the ablated model is a real, standalone checkpoint (`save_pretrained`/`push_to_hub`), not a runtime hook that needs to travel with the inference code.
- **Non-interactive candidate selection** with a full audit trail (`selection_report.ranked`) and an escape hatch (`selection.force_layer`) to override the auto-pick.
- **Per-prompt evaluation samples** (`refusal_samples`) alongside the aggregate refusal-rate/perplexity numbers, so results are auditable, not just a summary statistic.

## Requirements

- Python ≥ 3.10
- `torch`, `transformers` (≥ 4.57; the `qwen3_5` architecture adapter specifically needs a version that registers the `qwen3_5` model type — 5.x as of this writing), `datasets`, `accelerate`
- A CUDA GPU in practice — the `native_hf` backend loads on a single device, no CPU-offload/sharding support

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

With `pip`:

```bash
pip install -e .
```

## Quickstart

```bash
orthex --config configs/default.yaml \
  --set model.id=google/gemma-2-2b-it --set model.architecture_adapter=gemma2
```

Writes `out/gemma-2-2b-it/` — the abliterated checkpoint (`save_pretrained` output) plus `evaluation_report.json` (refusal rate + perplexity, pre/post, with per-prompt response samples).

`configs/default.yaml` is the only config file. Target a different model, or tune any other field, from the command line via the generic `--set key.path=value` flag rather than maintaining a yaml per model:

```bash
orthex --config configs/default.yaml \
  --set model.id=Qwen/Qwen2.5-3B-Instruct --set model.architecture_adapter=qwen2 \
  --set data.train.positive.params.n_samples=64 \
  --set selection.eval_top_n=10
```

See [docs/usage.md](docs/usage.md) for the full config schema, CLI reference, and how to extend orthex with a new model family, data source, or scorer.

## Supported architectures

| Architecture | Family | Example | Notes |
|---|---|---|---|
| `llama` | Llama | `meta-llama/Llama-3.2-3B-Instruct` | |
| `qwen2` | Qwen2.x | `Qwen/Qwen2.5-3B-Instruct` | |
| `qwen3` | Qwen3 (dense) | `Qwen/Qwen3-4B` | |
| `qwen3_5` | Qwen3.5 | `Qwen/Qwen3.5-4B` | hybrid linear/full-attention decoder; needs `transformers>=5.x` |
| `gemma2` | Gemma 2 | `google/gemma-2-2b-it` | |
| `gemma3` | Gemma 3 | `google/gemma-3-4b-it` | multimodal wrapper; text decoder nested under `model.language_model` |

`llama`, `qwen2`, `qwen3`, and `gemma2` share one adapter implementation (`GenericDecoderAdapter`) — their module paths are byte-for-byte identical (`model.model.embed_tokens` / `model.model.layers[i].self_attn.o_proj`), so supporting all four cost one class registered under four names, not four files. `qwen3_5` and `gemma3` needed dedicated adapters for their structural differences (see [docs/results.md](docs/results.md) for what those differences are).

Adding a new family is a new file under `orthex/architectures/` + one registry line — see [docs/usage.md](docs/usage.md#extending-orthex).

## How it works

Given a **positive** prompt set (elicits some target behavior — e.g. harmful requests) and a **negative** prompt set (baseline — e.g. harmless instructions), orthex:

1. captures residual-stream activations at every layer,
2. computes a per-layer direction (`positive_mean − negative_mean`),
3. scores candidate directions by generating completions with the direction temporarily projected out at inference time,
4. permanently orthogonalizes the selected direction out of `embed_tokens` and every layer's `attn_out`/`mlp_out`,
5. reports the refusal-rate and perplexity delta, pre vs. post ablation.

Terminology is `positive`/`negative`, not `harmful`/`harmless` — the same pipeline covers any elicited-behavior-vs-baseline target, refusal removal is just the default config.

## Responsible use

orthex exists to support red-teaming, robustness research, and model-behavior analysis — not to help anyone bypass safety measures on a model they aren't authorized to modify or deploy. Its default configuration specifically removes refusal behavior from safety-trained chat models. Only run it against models you have the right to modify, and don't publish or deploy an abliterated model in a way that violates that model's original license or your organization's policies. Knovel Engineering is not responsible for downstream misuse of models produced with this tool.

## Testing

```bash
pytest        # or: uv run pytest
```

Unit tests use tiny random-init models (constructed from config, no downloads) and mocked registries — no network access or GPU required. Real-model runs are only exercised manually via the CLI.

## Docs

- [docs/usage.md](docs/usage.md) — config schema, CLI reference, extending orthex with new providers/adapters/scorers
- [docs/results.md](docs/results.md) — measured refusal-rate/perplexity results across the validated model set, with links to the published checkpoints

## Acknowledgements

orthex implements the technique introduced in [Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717) (NeurIPS 2024). See their [reference implementation](https://github.com/andyrdt/refusal_direction) for the original code and experiments.

## Citation

If you use orthex in your work, please cite:

```bibtex
@software{orthex,
  title  = {orthex: Weight-Level Orthogonalization Toolkit for HF Causal LMs},
  author = {{Knovel Engineering}},
  year   = {2026},
  url    = {https://github.com/knoveleng/orthex}
}
```

## License

[MIT](LICENSE) © 2026 Knovel Engineering
