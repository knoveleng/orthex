import torch
from torch import nn

from orthex.ablation.weight_orthogonalizer import (
    _orthogonalize_embedding_like,
    _orthogonalize_linear_output_like,
    orthogonalize,
)


def _unit_direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(dim, generator=g)
    return d / d.norm()


def test_orthogonalize_embedding_like_removes_direction_from_every_row():
    torch.manual_seed(0)
    embed = nn.Embedding(8, 4)
    d = _unit_direction(4, seed=1)

    _orthogonalize_embedding_like(embed.weight.data, d)

    proj = embed.weight.data.to(torch.float32) @ d.to(torch.float32)
    assert proj.abs().max().item() < 1e-5


def test_orthogonalize_linear_output_like_removes_direction_from_every_possible_output():
    torch.manual_seed(0)
    linear = nn.Linear(4, 6, bias=False)  # out_features=6 plays the residual-stream role
    d = _unit_direction(6, seed=2)

    _orthogonalize_linear_output_like(linear.weight.data, d)

    # Behavioral check: for ANY input x, the output must have ~zero component along d.
    x = torch.randn(100, 4)
    y = x @ linear.weight.data.to(torch.float32).T
    proj = y @ d.to(torch.float32)
    assert proj.abs().max().item() < 1e-4


def test_orthogonalize_preserves_dtype():
    embed = nn.Embedding(8, 4)
    embed.weight.data = embed.weight.data.to(torch.bfloat16)
    d = _unit_direction(4, seed=3)
    _orthogonalize_embedding_like(embed.weight.data, d)
    assert embed.weight.data.dtype == torch.bfloat16


class _FakeLayer:
    def __init__(self, hidden: int, intermediate: int):
        self.self_attn = type("A", (), {"o_proj": nn.Linear(hidden, hidden, bias=False)})()
        self.mlp = type("M", (), {"down_proj": nn.Linear(intermediate, hidden, bias=False)})()


class _FakeModel:
    def __init__(self, hidden: int, intermediate: int, n_layers: int, tied: bool):
        self.embed_tokens = nn.Embedding(16, hidden)
        self.layers = [_FakeLayer(hidden, intermediate) for _ in range(n_layers)]
        self._lm_head = nn.Linear(hidden, 16, bias=False)
        if tied:
            self._lm_head.weight = self.embed_tokens.weight
        self.config = type("C", (), {"tie_word_embeddings": tied})()

    def get_output_embeddings(self):
        return self._lm_head


class _FakeAdapter:
    def embed_module(self, model):
        return model.embed_tokens

    def layer_write_modules(self, model):
        return [(layer.self_attn.o_proj, layer.mlp.down_proj) for layer in model.layers]


def test_orthogonalize_with_tied_embeddings_mutates_shared_tensor_once():
    torch.manual_seed(0)
    model = _FakeModel(hidden=4, intermediate=6, n_layers=2, tied=True)
    d = _unit_direction(4, seed=4)
    adapter = _FakeAdapter()

    orthogonalize(model, adapter, d, targets=["embed", "attn_out", "mlp_out"])

    assert model.embed_tokens.weight is model.get_output_embeddings().weight
    proj = model.embed_tokens.weight.data.to(torch.float32) @ d.to(torch.float32)
    assert proj.abs().max().item() < 1e-5
    for layer in model.layers:
        y = torch.randn(20, 4) @ layer.self_attn.o_proj.weight.data.to(torch.float32).T
        assert (y @ d.to(torch.float32)).abs().max().item() < 1e-4
