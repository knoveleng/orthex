import pytest

from orthex.selection.scorers.refusal_phrase_heuristic import RefusalPhraseHeuristicScorer


@pytest.mark.parametrize(
    "response,expected_is_refusal",
    [
        ("I'm sorry, I cannot help with that.", True),
        ("I CANNOT do that.", True),  # case-insensitive
        ("As an AI, I must decline.", True),
        ("Here is how you do it: step 1...", False),
        ("I don't think that's a good idea, but here goes.", True),  # substring match
    ],
)
def test_default_blacklist_matches(response, expected_is_refusal):
    scorer = RefusalPhraseHeuristicScorer()
    rate = scorer.score([response])
    assert rate == (1.0 if expected_is_refusal else 0.0)


def test_score_is_fraction_of_batch():
    scorer = RefusalPhraseHeuristicScorer()
    responses = ["I cannot help.", "Sure, here you go.", "I can't do that.", "Absolutely, here's how."]
    assert scorer.score(responses) == 0.5


def test_score_empty_batch_is_zero():
    assert RefusalPhraseHeuristicScorer().score([]) == 0.0


def test_custom_blacklist_overrides_default():
    scorer = RefusalPhraseHeuristicScorer(["nope"])
    assert scorer.score(["I cannot help."]) == 0.0
    assert scorer.score(["nope, not doing that"]) == 1.0


def test_is_refusal_matches_score_labels():
    scorer = RefusalPhraseHeuristicScorer()
    assert scorer.is_refusal("I'm sorry, I cannot help.") is True
    assert scorer.is_refusal("Sure, here's how.") is False
