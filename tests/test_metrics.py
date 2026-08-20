from unittest.mock import patch

from orthex.evaluation import metrics


class _FakeScorer:
    def __init__(self):
        self.blacklist = ["cannot"]

    def is_refusal(self, response: str) -> bool:
        return "cannot" in response.lower()

    def score(self, responses: list[str]) -> float:
        if not responses:
            return 0.0
        return sum(self.is_refusal(r) for r in responses) / len(responses)


def test_refusal_rate_returns_rate_and_prompt_response_samples():
    prompts = ["do X", "do Y"]
    fake_responses = ["I cannot do that.", "Sure, here you go."]

    with patch("orthex.evaluation.metrics.generate_completions", return_value=fake_responses):
        rate, samples = metrics.refusal_rate(model=object(), tokenizer=object(), prompts=prompts, scorer=_FakeScorer())

    assert rate == 0.5
    assert samples == [
        {"prompt": "do X", "response": "I cannot do that.", "is_refusal": True},
        {"prompt": "do Y", "response": "Sure, here you go.", "is_refusal": False},
    ]


def test_refusal_rate_samples_omit_is_refusal_when_scorer_lacks_it():
    class _NoLabelScorer:
        def score(self, responses):
            return 0.0

    with patch("orthex.evaluation.metrics.generate_completions", return_value=["hi"]):
        rate, samples = metrics.refusal_rate(model=object(), tokenizer=object(), prompts=["p"], scorer=_NoLabelScorer())

    assert samples == [{"prompt": "p", "response": "hi"}]
