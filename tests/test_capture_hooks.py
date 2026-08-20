import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from orthex.capture.activation_cache import capture_activations
from orthex.capture.types import LayerSite


@pytest.fixture(scope="module")
def tiny_llama():
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def test_capture_activations_shapes_and_count(tiny_llama):
    model, tokenizer = tiny_llama
    layers = list(model.model.layers)
    prompts = ["hello world", "a", "the quick brown fox jumps"]

    cache = capture_activations(model, tokenizer, layers, prompts, sites=["resid_pre", "resid_post"], batch_size=2)

    assert set(cache) == {LayerSite(i, site) for i in range(2) for site in ("resid_pre", "resid_post")}
    for tensor in cache.values():
        assert tensor.shape == (3, 8)


def test_resid_pre_and_resid_post_differ(tiny_llama):
    model, tokenizer = tiny_llama
    layers = list(model.model.layers)
    cache = capture_activations(model, tokenizer, layers, ["hello"], sites=["resid_pre", "resid_post"], batch_size=1)
    pre = cache[LayerSite(0, "resid_pre")]
    post = cache[LayerSite(0, "resid_post")]
    assert not torch.allclose(pre, post)


def test_hooks_are_removed_after_capture(tiny_llama):
    model, tokenizer = tiny_llama
    layers = list(model.model.layers)
    capture_activations(model, tokenizer, layers, ["hello"], sites=["resid_pre", "resid_post"], batch_size=1)
    for layer in layers:
        assert len(layer._forward_hooks) == 0
        assert len(layer._forward_pre_hooks) == 0
