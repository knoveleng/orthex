# data/

Local mirror of the prompt sets `configs/default.yaml` uses by default, so a run doesn't depend on network access to the Hub or on a path outside this repo. All three files are JSON arrays of `{"text": ...}` / `{"prompt": ...}` objects, read via the `local_json` provider.

| file | source | rows |
|---|---|---|
| `harmful_behaviors_train.json` | `mlabonne/harmful_behaviors`, `train` split | 416 |
| `harmless_alpaca_train.json` | `mlabonne/harmless_alpaca`, `train` split | 25058 |
| `harmbench_test.json` | HarmBench-style held-out eval set | 240 |

To regenerate the two dataset-derived files from their original HF datasets:

```python
import json
from datasets import load_dataset

def dump(dataset_id, split, out_path):
    ds = load_dataset(dataset_id, split=split)
    with open(out_path, "w") as f:
        json.dump([{"text": row["text"]} for row in ds], f, indent=2)

dump("mlabonne/harmful_behaviors", "train", "data/harmful_behaviors_train.json")
dump("mlabonne/harmless_alpaca", "train", "data/harmless_alpaca_train.json")
```

`harmbench_test.json` is a static eval set — not derived from a versioned upstream split, so there's nothing to regenerate it from; replace it directly if you want a different held-out set.
