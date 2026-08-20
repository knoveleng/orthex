# Usage

## Config schema

`configs/default.yaml` is the only config file in the repo. Per-run fields (which model, which architecture adapter) are overridden from the command line — see [CLI reference](#cli-reference) — rather than maintained as separate yaml files.

```yaml
model:
  id: meta-llama/Llama-3.2-3B-Instruct
  architecture_adapter: llama   # which family this model belongs to -- lives here (not under
  # export) because it's used throughout the pipeline (capture/selection/ablation), not just
  # at export time
  backend: native_hf
  dtype: bfloat16

data:
  train:
    positive:
      provider: local_json
      params: {id: data/harmful_behaviors_train.json, n_samples: 256}
    negative:
      provider: local_json
      params: {id: data/harmless_alpaca_train.json, n_samples: 256}
  test:
    # positive only -- refusal_rate and perplexity are both measured on
    # this same held-out prompt set, pre vs. post ablation
    provider: local_json
    params: {id: data/harmbench_test.json, n_samples: 32}

capture:
  sites: [resid_pre, resid_post]
  position: -1

directions:
  strategy: mean_difference
  layer_range: [1, -1]

selection:
  strategy: refusal_phrase_heuristic
  eval_top_n: 20
  blacklist: []          # empty -> the built-in 15-phrase default
  sites: [resid_pre]
  max_new_tokens: 32
  generation_batch_size: 16
  force_layer: null       # set to skip auto-pick and use this exact candidate
  force_site: null

ablation:
  strategy: weight_orthogonalization
  targets: [embed, attn_out, mlp_out]

export:
  push_to_hub: false
  repo_id: null            # required if push_to_hub: true
  push_kwargs: {}          # forwarded to model/tokenizer.push_to_hub()
  output_dir: ./out

evaluation:
  refusal_rate: true
  perplexity_delta: true
  benchmark_subset: null   # unimplemented in v1
```

### `data.*.params`

Each data source is `{provider, params, name?}`. `params` is passed straight through as `**kwargs` to the provider's constructor:

| provider | params | notes |
|---|---|---|
| `hf_dataset` | `id`, `split="train"`, `text_field="text"`, `n_samples=256`, **`**kwargs`** | extra kwargs (`name` for a dataset sub-config, `trust_remote_code`, `revision`, `streaming`, ...) are forwarded straight to `datasets.load_dataset()` |
| `local_json` | `id`, `n_samples=32`, `text_field=None` | reads a JSON array of objects (HarmBench/AdvBench-style); `text_field=None` auto-detects among `prompt`/`text`/`instruction`/`goal`/`input`, set it explicitly if a file has more than one plausible field |
| `local_jsonl` | `id`, `n_samples=256`, `text_field=None` | same, one JSON object per line |

`local_json`/`local_jsonl` do **not** accept unknown params — a typo (e.g. `n_smaples`) fails fast with the accepted-params list. `hf_dataset` is the deliberate exception (see table above), trading that fail-fast guarantee for not having to hand-enumerate every `load_dataset()` option.

`data.train` needs both `positive` and `negative` (to compute the direction). `data.test` is a single source, positive only.

`configs/default.yaml` defaults to `local_json` for all three sources, reading the local mirror under `data/` (see `data/README.md` for provenance and how to regenerate it from the original HF datasets) — no network access needed at run time. `hf_dataset` remains available and is a straightforward swap if you'd rather point directly at the Hub.

### `export`

`repo_id` is required when `push_to_hub: true` — this is enforced at config-load time (`ExportConfig.__post_init__`), not mid-pipeline. `push_kwargs` is forwarded as `**kwargs` to both `model.push_to_hub()` and `tokenizer.push_to_hub()` (e.g. `{private: true, commit_message: "..."}`).

## CLI reference

```
orthex --config CONFIG [--config CONFIG ...]
       [--set key.path=value ...]
       [--output-dir DIR]
```

- `--config` — repeatable; later files deep-merge onto earlier ones. In practice there's one file (`configs/default.yaml`).
- `--set key.path=value` — the one mechanism for overriding any field, repeatable. Value is YAML-parsed (`--set export.push_to_hub=true` becomes a real bool). There are deliberately no named flags (`--model-id` etc.) for individual fields — `--set` reaches every field at any depth uniformly, so there's a single override mechanism to remember rather than named sugar for a few fields plus a fallback for everything else.
- `--output-dir` — defaults to `<export.output_dir>/<model-id-basename>`.

Examples:

```bash
# Run against a new model
orthex --config configs/default.yaml \
  --set model.id=Qwen/Qwen3-4B --set model.architecture_adapter=qwen3

# Pick a specific candidate instead of auto-select (e.g. reviewing
# selection_report.ranked from a prior run and choosing a gentler layer)
orthex --config configs/default.yaml \
  --set model.id=google/gemma-3-4b-it --set model.architecture_adapter=gemma3 \
  --set selection.force_layer=31 --set selection.force_site=resid_pre

# Push to the Hub
orthex --config configs/default.yaml \
  --set model.id=google/gemma-2-2b-it --set model.architecture_adapter=gemma2 \
  --set export.push_to_hub=true --set export.repo_id=your-org/gemma-2-2b-it-abliterated
```

`scripts/run_all_models.sh` drives the full validated model set sequentially (one GPU, one model at a time); `scripts/summarize_run.py` globs `out/*/evaluation_report.json` into a Markdown table.

## Output format

`out/<model>/` holds the `save_pretrained` checkpoint plus:

- `README.md` — a standard HF model card (YAML frontmatter + summary), generated from the run so the local repo isn't left undocumented even before a `push_to_hub`. `license` is deliberately left for you to set — see [`orthex/model_card.py`](../orthex/model_card.py).
- `evaluation_report.json`:

```json
{
  "model_id": "...",
  "architecture_adapter": "...",
  "selected_candidate": {"layer": 15, "site": "resid_pre", "refusal_rate": 0.19},
  "selection_report": {"ranked": [...], "selected": {...}},
  "evaluation": {
    "refusal_rate": {"pre": 0.78, "post": 0.19, "delta": -0.59},
    "perplexity": {"pre": 18.49, "post": 21.78, "delta": 3.29}
  },
  "refusal_samples": {
    "pre": [{"prompt": "...", "response": "...", "is_refusal": true}, ...],
    "post": [...]
  }
}
```

`refusal_samples` holds one record per held-out test prompt, pre- and post-ablation, so you can spot-check what the model actually generated rather than trusting the aggregate rate alone. `selection_report.ranked` holds every candidate evaluated during the (non-interactive) auto-pick, in case a different one looks preferable after the fact — see `selection.force_layer` above.

## Extending orthex

Every layer with more than one plausible implementation is a registry: `orthex/{data,backends,directions,selection/scorers,ablation,architectures}/registry.py`. Adding a new one is a new file + one registry line, nothing else changes:

```python
# orthex/architectures/my_family_adapter.py
from orthex.architectures.registry import register

@register("my_family")
class MyFamilyAdapter:
    def embed_module(self, model): ...
    def decoder_layers(self, model): ...
    def layer_write_modules(self, model): ...   # [(attn_out, mlp_out), ...] per layer
```

Then register the import in `orthex/pipeline.py`'s side-effect import block (`from orthex.architectures import ..., my_family_adapter  # noqa: F401`) and reference it as `model.architecture_adapter: my_family` / `--set model.architecture_adapter=my_family`.

The same pattern applies to a new `DataProvider` (`orthex/data/base.py`), `DirectionStrategy` (`orthex/directions/base.py`), selection `SelectionScorer` (`orthex/selection/scorers/base.py`), or `AblationStrategy` (`orthex/ablation/base.py`) — implement the Protocol, register under a name, import it once for the side effect.

Add a `test_registry_has_all_N_<things>()` sanity test alongside the new registration (see `tests/test_registries.py`) so a typo in the registered name gets caught immediately.
