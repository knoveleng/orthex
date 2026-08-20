from transformers import AutoModelForCausalLM, Gemma2Config, LlamaConfig, Qwen2Config

from orthex.architectures.gemma3_adapter import Gemma3Adapter
from orthex.architectures.llama_adapter import GenericDecoderAdapter
from orthex.architectures.qwen3_5_adapter import Qwen3_5Adapter


def _tiny_llama_family(config_cls, **extra):
    config = config_cls(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        **extra,
    )
    return AutoModelForCausalLM.from_config(config)


def test_generic_decoder_adapter_on_llama():
    model = _tiny_llama_family(LlamaConfig)
    adapter = GenericDecoderAdapter()
    assert adapter.embed_module(model) is model.model.embed_tokens
    assert len(adapter.decoder_layers(model)) == 3
    write_modules = adapter.layer_write_modules(model)
    assert len(write_modules) == 3
    for layer, (o_proj, down_proj) in zip(model.model.layers, write_modules):
        assert o_proj is layer.self_attn.o_proj
        assert down_proj is layer.mlp.down_proj


def test_generic_decoder_adapter_on_qwen2():
    model = _tiny_llama_family(Qwen2Config)
    adapter = GenericDecoderAdapter()
    assert adapter.embed_module(model) is model.model.embed_tokens
    assert len(adapter.decoder_layers(model)) == 3


def test_generic_decoder_adapter_on_gemma2():
    model = _tiny_llama_family(Gemma2Config)
    adapter = GenericDecoderAdapter()
    assert adapter.embed_module(model) is model.model.embed_tokens
    assert len(adapter.decoder_layers(model)) == 3


class _StubDecoderLayer:
    def __init__(self, hidden, intermediate):
        import torch.nn as nn

        self.self_attn = type("A", (), {"o_proj": nn.Linear(hidden, hidden, bias=False)})()
        self.mlp = type("M", (), {"down_proj": nn.Linear(intermediate, hidden, bias=False)})()


class _StubLanguageModel:
    def __init__(self, hidden, intermediate, n_layers):
        import torch.nn as nn

        self.embed_tokens = nn.Embedding(16, hidden)
        self.layers = [_StubDecoderLayer(hidden, intermediate) for _ in range(n_layers)]


class _StubGemma3Model:
    def __init__(self, hidden, intermediate, n_layers):
        self.language_model = _StubLanguageModel(hidden, intermediate, n_layers)


class _StubGemma3ForConditionalGeneration:
    def __init__(self, hidden=8, intermediate=16, n_layers=3):
        self.model = _StubGemma3Model(hidden, intermediate, n_layers)


class _Qwen3_5FullAttnLayer:
    def __init__(self, hidden, intermediate):
        import torch.nn as nn

        self.self_attn = type("A", (), {"o_proj": nn.Linear(hidden, hidden, bias=False)})()
        self.mlp = type("M", (), {"down_proj": nn.Linear(intermediate, hidden, bias=False)})()


class _Qwen3_5LinearAttnLayer:
    def __init__(self, hidden, intermediate):
        import torch.nn as nn

        self.linear_attn = type("L", (), {"out_proj": nn.Linear(hidden, hidden, bias=False)})()
        self.mlp = type("M", (), {"down_proj": nn.Linear(intermediate, hidden, bias=False)})()


class _Qwen3_5TextModel:
    def __init__(self, hidden, intermediate, layer_types):
        import torch.nn as nn

        self.embed_tokens = nn.Embedding(16, hidden)
        self.layers = [
            _Qwen3_5FullAttnLayer(hidden, intermediate)
            if t == "full_attention"
            else _Qwen3_5LinearAttnLayer(hidden, intermediate)
            for t in layer_types
        ]


class _Qwen3_5ForCausalLMStub:
    def __init__(self, hidden=8, intermediate=16, layer_types=("linear_attention",) * 3 + ("full_attention",)):
        self.model = _Qwen3_5TextModel(hidden, intermediate, layer_types)


def test_qwen3_5_adapter_handles_hybrid_attention_layers():
    # Real Qwen3_5ForCausalLM requires the full gated-delta-net/conv1d
    # machinery to construct even from a minimal config; a hand-built stub
    # with the confirmed attribute names (self_attn.o_proj OR linear_attn.
    # out_proj, per layer_types) is sufficient to test locator logic.
    model = _Qwen3_5ForCausalLMStub(layer_types=("linear_attention", "linear_attention", "full_attention", "linear_attention"))
    adapter = Qwen3_5Adapter()

    assert adapter.embed_module(model) is model.model.embed_tokens
    assert len(adapter.decoder_layers(model)) == 4

    write_modules = adapter.layer_write_modules(model)
    assert write_modules[0][0] is model.model.layers[0].linear_attn.out_proj
    assert write_modules[1][0] is model.model.layers[1].linear_attn.out_proj
    assert write_modules[2][0] is model.model.layers[2].self_attn.o_proj
    assert write_modules[3][0] is model.model.layers[3].linear_attn.out_proj
    for layer, (_, down_proj) in zip(model.model.layers, write_modules):
        assert down_proj is layer.mlp.down_proj


def test_gemma3_adapter_reaches_nested_language_model():
    # Real Gemma3ForConditionalGeneration requires a vision tower checkpoint
    # to construct even from a config; a hand-built stub with the exact
    # attribute names (confirmed against installed modeling_gemma3.py) is
    # sufficient to test the adapter's *locator* logic.
    model = _StubGemma3ForConditionalGeneration(n_layers=3)
    adapter = Gemma3Adapter()

    assert adapter.embed_module(model) is model.model.language_model.embed_tokens
    assert len(adapter.decoder_layers(model)) == 3
    write_modules = adapter.layer_write_modules(model)
    for layer, (o_proj, down_proj) in zip(model.model.language_model.layers, write_modules):
        assert o_proj is layer.self_attn.o_proj
        assert down_proj is layer.mlp.down_proj
