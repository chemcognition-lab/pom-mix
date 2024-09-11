"""Layers and models for Chemix.

We seeks three types of symmetries for layers and models of mixtures and sets of mixtures:
1. Permutation Equivariance (PE) for mixture layers, that is to say, a permutation on a input mixture should return a permuted output (mixture tensors). 
2. Permutation Invariance (PI) for mixture aggregators, that is to say, the order of compounds in a mixture should not affect the output (mixture embeddings).
2. Permutation Invariance (PI) for the pairs of mixtures, that is to say, the order of mixtures in a set should not affect the output (similary score for example).

"""

import enum
import functools
import sys
import warnings

import torch
from torch import nn

from . import types
from .maths import masked_max, masked_mean, masked_min, masked_variance

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

UNK_TOKEN = -999


class AggEnum(StrEnum):
    """Basic str enum for molecule aggregators."""

    mean = enum.auto()
    pna = enum.auto()
    attn = enum.auto()


class RegressorEnum(StrEnum):
    """Basic str enum for regressors."""

    minmax = enum.auto()
    pna = enum.auto()
    sum = enum.auto()
    scaled_cosine = enum.auto()


class ActivationEnum(StrEnum):
    """Basic str enum for activation functions."""

    sigmoid = enum.auto()
    hardtanh = enum.auto()


ACTIVATION_MAP = {
    ActivationEnum.sigmoid: nn.Sigmoid,
    ActivationEnum.hardtanh: functools.partial(nn.Hardtanh, min_val=0.0, max_val=1.0),
}


def compute_key_padding_mask(x, unk_token=UNK_TOKEN):
    return (x == unk_token).all(dim=2)


class MLP(nn.Module):
    """Basic MLP with dropout and GELU activation."""

    def __init__(
        self,
        hidden_dim: int,
        add_linear_last: bool,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(nn.LazyLinear(hidden_dim))
            self.layers.append(nn.GELU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(p=dropout_rate))
        if add_linear_last:
            self.layers.append(nn.LazyLinear(hidden_dim))

    def forward(self, x: types.Tensor) -> types.Tensor:
        output = self.layers(x)
        return output


class AddNorm(nn.Module):
    """Residual connection with layer normalization and dropout."""

    def __init__(self, embed_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1: types.Tensor, x2: types.Tensor) -> types.Tensor:
        return self.norm(x1 + self.dropout(x2))


class MolecularAttention(nn.Module):
    """Molecule-wise PE attention for a mixture."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        add_mlp: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.self_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.addnorm1 = AddNorm(embed_dim, dropout_rate)
        self.add_mlp = add_mlp
        if self.add_mlp:
            self.ffn = MLP(
                embed_dim, num_layers=1, add_linear_last=True, dropout_rate=0.0
            )
            self.addnorm2 = AddNorm(embed_dim, dropout_rate)

    def forward(
        self, x: types.MixTensor, key_padding_mask: types.MaskTensor
    ) -> types.MixTensor:
        attn_x, _ = self.self_attn_layer(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.addnorm1(x, attn_x)
        if self.add_mlp:
            out = self.addnorm2(out, self.ffn(out))
        return out


class MeanAggregation(nn.Module):
    """Simple mean aggregation with masking."""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: types.MixTensor, key_padding_mask: types.MaskTensor
    ) -> types.EmbTensor:
        global_emb = masked_mean(x, ~key_padding_mask)
        global_emb = global_emb.squeeze(1)
        return global_emb


class AttentionAggregation(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.cross_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.mean_agg = MeanAggregation()

    def forward(
        self, x: types.MixTensor, key_padding_mask: types.MaskTensor
    ) -> types.EmbTensor:
        avg_emb = self.mean_agg(x, key_padding_mask).unsqueeze(1)
        global_emb, _ = self.cross_attn_layer(
            query=avg_emb,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        global_emb = global_emb.squeeze(1)
        return global_emb


class PrincipalNeighborhoodAggregation(nn.Module):
    """PN-style (mean, var, min, max) aggregation."""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: types.MixTensor, key_padding_mask: types.MaskTensor
    ) -> types.EmbTensor:
        mean = masked_mean(x, ~key_padding_mask).unsqueeze(1)
        var = masked_variance(x, ~key_padding_mask).unsqueeze(1)
        minimum = masked_min(x, ~key_padding_mask).unsqueeze(1)
        maximum = masked_max(x, ~key_padding_mask).unsqueeze(1)
        global_emb = torch.cat((mean, var, minimum, maximum), dim=-1)
        global_emb = global_emb.squeeze(1)
        return global_emb


class MixtureBlock(nn.Module):
    """Stack of layers to processes mixtures of molecules with a final aggregation operation."""

    def __init__(
        self,
        mol_aggregation: nn.Module,
        num_layers: int = 1,
        **mol_attn_args,
    ):
        super().__init__()

        layers = (
            [MolecularAttention(**mol_attn_args) for _ in range(num_layers)]
            if num_layers > 0
            else [nn.Identity()]
        )
        self.mol_attn_layers = nn.ModuleList(layers)
        self.mol_aggregation = mol_aggregation
        self.ffn = MLP(
            hidden_dim=mol_attn_args["embed_dim"],
            dropout_rate=mol_attn_args["dropout_rate"],
            add_linear_last=False,
        )

    def forward(
        self, x: types.MixTensor, key_padding_mask: types.MaskTensor
    ) -> types.EmbTensor:
        for layer in self.mol_attn_layers:
            if isinstance(layer, nn.Identity):
                x = layer(x)
            else:
                x = layer(x, key_padding_mask)

        global_emb = self.mol_aggregation(x, key_padding_mask)
        global_emb = self.ffn(global_emb)
        return global_emb


class SumRegressor(nn.Module):
    """Mixture pair PI similarity regressor."""

    def __init__(self, output_dim: int, act: ActivationEnum):
        super().__init__()
        self.layer = nn.LazyLinear(output_dim)
        self.activation = ACTIVATION_MAP[act]()

    def forward(self, x: types.ManyEmbTensor) -> types.PredictionTensor:
        x = x.sum(-1)
        output = self.layer(x)
        output = self.activation(output)
        return output


class CatRegressor(nn.Module):
    """Concat embedding and regresss similarity, no PI"""

    def __init__(self, output_dim: int, act: ActivationEnum):
        super().__init__()
        self.layer = nn.LazyLinear(output_dim)
        self.activation = ACTIVATION_MAP[act]()

    def forward(self, x: types.ManyEmbTensor) -> types.PredictionTensor:
        x = torch.unbind(x, -1)
        x = torch.cat(x, 1)
        output = self.layer(x)
        output = self.activation(output)
        return output


class MinMaxRegressor(nn.Module):
    """Mixture pair PI similarity regressor (mix and max).

    Rationale is that a pair has 2-degrees of freedom, so we need at least two
    permutation invariant (PI) operations to capture all the information.
    Min and max work, but could be sum or avg.
    """

    def __init__(self, output_dim: int, act: ActivationEnum):
        super().__init__()
        self.layer = nn.LazyLinear(output_dim)
        self.activation = ACTIVATION_MAP[act]()

    def forward(self, x: types.ManyEmbTensor) -> types.PredictionTensor:
        x_min, _ = torch.min(x, dim=-1)
        x_max, _ = torch.max(x, dim=-1)
        x_cat = torch.cat([x_min, x_max], dim=-1)
        output = self.layer(x_cat)
        output = self.activation(output)
        return output


class PNARegressor(nn.Module):
    """Mixture pair PI similarity regressor (avg and mix and max).

    Rationale is that a pair has 2-degrees of freedom, so we need at least two
    permutation invariant (PI) operations to capture all the information.
    Min and max work, but could be sum or avg.
    """

    def __init__(self, output_dim: int, act: ActivationEnum):
        super().__init__()
        self.layer = nn.LazyLinear(output_dim)
        self.activation = ACTIVATION_MAP[act]()

    def forward(self, x: types.ManyEmbTensor) -> types.PredictionTensor:
        x_avg = torch.mean(x, dim=-1)
        x_min, _ = torch.min(x, dim=-1)
        x_max, _ = torch.max(x, dim=-1)
        x_cat = torch.cat([x_min, x_avg, x_max], dim=-1)
        output = self.layer(x_cat)
        output = self.activation(output)
        return output


class CosineRegressor(nn.Module):
    """Cosine similarity as regressor, is PI."""

    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        x1, x2 = x.unbind(dim=-1)
        sim = self.cosine_similarity(x1, x2).unsqueeze(-1)
        return 1.0 - sim


class ScaledCosineRegressor(nn.Module):
    """Use scaled cosine similarity as similarity regressor, is PI.

    We add a scaling layer since we have observed that cosine similarity
    will not always match the target range of the similarity score.
    Sigmoid is to restrict the output to [0, 1].
    """

    def __init__(self, out_dim: int, act: ActivationEnum):
        super().__init__()
        self.cosine_distance = CosineRegressor()
        self.scaler = nn.Linear(out_dim, out_dim)
        self.activation = ACTIVATION_MAP[act]()

    def forward(self, x):
        cos_dist = self.cosine_distance(x)
        return self.activation(self.scaler(cos_dist))


class Chemix(nn.Module):

    def __init__(
        self,
        input_net: nn.Module,
        mixture_net: nn.Module,
        regressor: nn.Module,
    ):
        super(Chemix, self).__init__()
        self.input_net = input_net
        self.mixture_net = mixture_net
        self.regressor = regressor
        self.unk_token = UNK_TOKEN

    def embed(self, x: types.ManyMixTensor) -> types.ManyEmbTensor:
        x_all = []
        for mix in torch.unbind(x, dim=-1):
            key_padding_mask = compute_key_padding_mask(mix, self.unk_token)
            mix = self.input_net(mix)
            emb_mix = self.mixture_net(mix, key_padding_mask)
            x_all.append(emb_mix)

        final_emb = torch.stack(x_all, dim=-1)
        return final_emb

    def forward(self, x: types.ManyMixTensor) -> types.PredictionTensor:
        emb = self.embed(x)
        pred = self.regressor(emb)
        return pred


def build_chemix(config):

    project_input = nn.Linear(config.pom_input.embed_dim, config.mixture_net.embed_dim)

    mol_aggregation_methods = {
        AggEnum.mean: MeanAggregation(),
        AggEnum.pna: PrincipalNeighborhoodAggregation(),
        AggEnum.attn: AttentionAggregation(
            embed_dim=config.attn_aggregation.embed_dim,
            num_heads=config.attn_num_heads,
            dropout_rate=config.dropout_rate,
        ),
    }

    mixture_net = MixtureBlock(
        num_layers=config.mixture_net.num_layers,
        embed_dim=config.mixture_net.embed_dim,
        num_heads=config.attn_num_heads,
        add_mlp=config.mixture_net.add_mlp,
        dropout_rate=config.dropout_rate,
        mol_aggregation=mol_aggregation_methods[config.mol_aggregation],
    )

    # Here for backwards support, remove once all modes have the
    if "activation" not in config.regressor:
        warnings.warn("chemix.regressor.activation not found, using default `sigmoid`")
        config.regressor.activation = "sigmoid"

    output_dim = config.regressor.output_dim
    act = config.regressor.activation
    regressor_type = {
        RegressorEnum.sum: SumRegressor(output_dim, act),
        RegressorEnum.minmax: MinMaxRegressor(output_dim, act),
        RegressorEnum.pna: PNARegressor(output_dim, act),
        RegressorEnum.scaled_cosine: ScaledCosineRegressor(output_dim, act),
    }

    chemix = Chemix(
        input_net=project_input,
        regressor=regressor_type[config.regressor.type],
        mixture_net=mixture_net,
    )

    return chemix


def test_molecule_permutation_equivariance(model, x, key_padding):
    """Tests permutation invariance of a model on the second axis of the input."""
    _, n_mols, _ = x.shape
    perm_indices = torch.randperm(n_mols)
    permuted_x = x[:, perm_indices, :]
    with torch.no_grad():
        output = model(x, key_padding)
        expected_output = output[:, perm_indices, :]
        permuted_key_padding = compute_key_padding_mask(permuted_x)
        actual_output = model(permuted_x, permuted_key_padding)

    distance = torch.norm(expected_output - actual_output).mean().item()
    assert torch.allclose(
        expected_output, actual_output
    ), f"Model ({model.__class__.__name__}) is not permutation equivariant, {distance:=}"
    return


def test_molecule_permutation_invariance(model, x, key_padding):
    """Tests permutation invariance of a model on the second axis of the input."""

    _, n_mols, _ = x.shape
    perm_indices = torch.randperm(n_mols)
    permuted_x = x[:, perm_indices, :]
    with torch.no_grad():
        expected_output = model(x, key_padding)
        permuted_key_padding = compute_key_padding_mask(permuted_x)
        actual_output = model(permuted_x, permuted_key_padding)
    distance = torch.norm(expected_output - actual_output).mean().item()
    assert torch.allclose(
        expected_output,
        actual_output,
    ), f"Model ({model.__class__.__name__}) is not permutation invariant, {distance:=}"
    return


def _rand_mixtures(batch_size: int, max_mix: int, embed_dim: int):
    lens = torch.randint(1, max_mix, (batch_size,))
    mixes = []
    for length in lens:
        x_temp = torch.rand((max_mix, embed_dim))
        x_temp[length:, :] = UNK_TOKEN
        mixes.append(x_temp)
    x = torch.stack(mixes)
    return x, lens


if __name__ == "__main__":
    print("Creating test data")
    batch_size, max_mix, embed_dim = 3, 5, 10
    input_shape = (batch_size, max_mix, embed_dim)
    x1, lens = _rand_mixtures(batch_size, max_mix, embed_dim)
    x2 = _rand_mixtures(batch_size, max_mix, embed_dim)
    x12 = torch.cat([x1, x2], dim=-1)
    x21 = torch.cat([x2, x1], dim=-1)
    print("Testing utilities")
    key_padding1 = ~compute_key_padding_mask(x1)
    key_padding2 = ~compute_key_padding_mask(x2)

    assert torch.allclose(
        lens, (~key_padding1).sum(1)
    ), "Padding mask does not coincide"
    assert torch.allclose(
        lens, (~key_padding2).sum(1)
    ), "Padding mask does not coincide"

    print("Testing molecule permutation equivariance")
    m = MolecularAttention(embed_dim)
    test_molecule_permutation_equivariance(m, x1, key_padding1)
    test_molecule_permutation_equivariance(m, x2, key_padding2)
    print("Testing molecule permutation invariance")

    for agg in [
        MeanAggregation,
        PrincipalNeighborhoodAggregation,
        lambda: AttentionAggregation(embed_dim),
    ]:
        m = agg()
        test_molecule_permutation_invariance(m, x1, key_padding1)
        test_molecule_permutation_invariance(m, x2, key_padding2)

    print("Testing prediction permutation invariance")
    for cls in [MinMaxRegressor, SumRegressor, CosineRegressor]:
        m = cls(1)
        y_pred12 = m(x12)
        y_pred21 = m(x21)
        assert torch.allclose(y_pred12, y_pred21), "Model is not permutation invariant"
        assert torch.allclose(y_pred12, y_pred21), "Model is not permutation invariant"
