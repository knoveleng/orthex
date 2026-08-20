from __future__ import annotations

from orthex.evaluation import metrics


def evaluate(model, tokenizer, pos_test: list[str], scorer, eval_config) -> dict:
    """Both refusal rate and perplexity are measured on the same held-out
    positive (behavior-eliciting) test prompts, pre vs. post ablation --
    an apples-to-apples comparison on one prompt set rather than splitting
    the two metrics across unrelated corpora. `benchmark_subset` (e.g.
    MMLU-lite) stays unimplemented for v1."""
    report: dict = {}
    if eval_config.refusal_rate:
        rate, samples = metrics.refusal_rate(model, tokenizer, pos_test, scorer)
        report["refusal_rate"] = rate
        report["refusal_samples"] = samples
    if eval_config.perplexity_delta:
        report["perplexity"] = metrics.perplexity(model, tokenizer, pos_test)
    return report
