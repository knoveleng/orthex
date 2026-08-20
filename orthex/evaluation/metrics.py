from __future__ import annotations

import torch

from orthex.generation import generate_completions


def refusal_rate(
    model, tokenizer, prompts: list[str], scorer, max_new_tokens: int = 64, batch_size: int = 16
) -> tuple[float, list[dict]]:
    """Returns (rate, samples) where samples is one {prompt, response,
    is_refusal} record per prompt -- kept alongside the aggregate rate in
    the evaluation report so a human can spot-check what actually changed,
    not just the summary number."""
    responses = generate_completions(model, tokenizer, prompts, max_new_tokens, batch_size)
    rate = scorer.score(responses)
    is_refusal = getattr(scorer, "is_refusal", None)
    samples = [
        {"prompt": p, "response": r, **({"is_refusal": is_refusal(r)} if is_refusal else {})}
        for p, r in zip(prompts, responses)
    ]
    return rate, samples


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], stride: int = 512, max_length: int | None = None) -> float:
    """Standard HF sliding-window NLL/perplexity over concatenated `texts`."""
    device = next(model.parameters()).device
    max_length = max_length or getattr(model.config, "max_position_embeddings", 2048)
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100
        out = model(ids, labels=target_ids)
        num_valid = int((target_ids != -100).sum().item())
        nll_sum += out.loss.item() * num_valid
        n_tokens += num_valid
        prev_end = end
        if end == seq_len:
            break
    return float(torch.exp(torch.tensor(nll_sum / n_tokens)))
