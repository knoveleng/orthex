#!/usr/bin/env python3
"""One-off: push the already-ablated models in out/ to the knoveleng org on
the Hub, public, under a -Uncensored/-uncensored suffix matching the base
model's own casing style. Uploads the local save_pretrained + README.md +
evaluation_report.json directory as-is -- no re-loading into memory."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

MODELS = {
    "Llama-3.2-3B-Instruct": "knoveleng/Llama-3.2-3B-Instruct-Uncensored",
    "Qwen2.5-3B-Instruct": "knoveleng/Qwen2.5-3B-Instruct-Uncensored",
    "Qwen3-4B": "knoveleng/Qwen3-4B-Uncensored",
    "Qwen3.5-4B": "knoveleng/Qwen3.5-4B-Uncensored",
    "gemma-2-2b-it": "knoveleng/gemma-2-2b-it-uncensored",
    "gemma-3-4b-it": "knoveleng/gemma-3-4b-it-uncensored",
}


def main() -> None:
    api = HfApi()
    for local_name, repo_id in MODELS.items():
        local_dir = Path("out") / local_name
        print(f"PUSH_START {local_name} -> {repo_id}")
        api.create_repo(repo_id, private=False, exist_ok=True)
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            commit_message="Upload abliterated model (orthex)",
        )
        print(f"PUSH_DONE {local_name} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
