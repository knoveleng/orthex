from orthex.model_card import render

_RESULT = {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "architecture_adapter": "llama",
    "selected_candidate": {"layer": 15, "site": "resid_pre", "refusal_rate": 0.19},
    "evaluation": {
        "refusal_rate": {"pre": 0.78, "post": 0.19, "delta": -0.59},
        "perplexity": {"pre": 18.49, "post": 21.78, "delta": 3.29},
    },
}


def test_render_includes_frontmatter_with_base_model():
    card = render(_RESULT)
    assert card.startswith("---\n")
    assert "base_model: meta-llama/Llama-3.2-3B-Instruct" in card
    assert "- abliterated" in card
    assert "- orthex" in card


def test_render_includes_architecture_and_selected_candidate():
    card = render(_RESULT)
    assert "`llama`" in card
    assert "layer 15, site `resid_pre`" in card


def test_render_includes_evaluation_table_values():
    card = render(_RESULT)
    assert "0.780" in card
    assert "0.190" in card
    assert "-0.590" in card
    assert "18.490" in card
    assert "21.780" in card
    assert "3.290" in card


def test_render_includes_responsible_use_and_license_sections():
    card = render(_RESULT)
    assert "## Responsible use" in card
    assert "## License" in card


def test_render_handles_missing_metrics_gracefully():
    result = dict(_RESULT)
    result["evaluation"] = {
        "refusal_rate": {"pre": None, "post": None, "delta": None},
        "perplexity": {"pre": None, "post": None, "delta": None},
    }
    card = render(result)
    assert "n/a" in card
