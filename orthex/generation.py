from __future__ import annotations

import torch


def format_prompt(tokenizer, prompt: str) -> str:
    """Wraps a raw instruction in the model's chat template with the
    assistant-turn cue appended (add_generation_prompt=True). Used by BOTH
    capture (direction extraction) and generation (selection scoring,
    pre/post evaluation) so the "last token" position always lands right
    before where the model would start generating -- the position that
    actually reflects its refusal decision. Capturing on raw, un-templated
    text would extract a direction at an unrelated position and produce a
    much weaker (or model-family-dependent, inconsistently effective)
    ablation.

    enable_thinking=False is passed unconditionally: Qwen3/Qwen3.5-family
    templates default to thinking mode ON (opening an unclosed `<think>`
    block) when the kwarg is omitted, which both slows generation a lot and
    means capture/scoring happen at a reasoning-preamble position rather
    than the model's actual answer-start position. Templates that don't
    define `enable_thinking` (Llama, Gemma, Qwen2.5) silently ignore the
    extra kwarg -- confirmed harmless, not just assumed."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


@torch.no_grad()
def generate_completions(
    model, tokenizer, prompts: list[str], max_new_tokens: int = 64, batch_size: int = 16
) -> list[str]:
    """Batched greedy generation through the instruct chat template (all v1
    target models are -Instruct/-it checkpoints). Returns only the newly
    generated continuation per prompt. Shared by selection/candidate_scorer
    (eval-time activation-projection scoring) and evaluation/metrics
    (pre/post refusal-rate measurement)."""
    device = next(model.parameters()).device
    completions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = [format_prompt(tokenizer, p) for p in prompts[start : start + batch_size]]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        completions.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return completions
