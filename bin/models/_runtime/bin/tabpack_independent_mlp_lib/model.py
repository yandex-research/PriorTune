from __future__ import annotations

import sys
from typing import Any

import torch
from torch import Tensor

from bin.tabpack_lib import project as tabpack_project
from bin.tabpack_lib import vendor as tabpack_vendor


# The vendored upstream code uses absolute imports internally.
sys.modules.setdefault('project', tabpack_project)
sys.modules.setdefault('vendor', tabpack_vendor)

import project.nn  # noqa: E402
import project.tabpack  # noqa: E402


def stream_seed(seed: int, stream: int) -> int:
    """Derive a stable positive torch seed without depending on Python hashing."""
    value = (int(seed) ^ (0x9E3779B97F4A7C15 * (stream + 1))) & ((1 << 63) - 1)
    return value or 1


class IndependentDropoutPack(project.nn.ModulePack):
    """Dropout whose RNG stream is independent for every logical model.

    Stock ``DropoutPack`` draws one tensor from the process-global generator.  That
    is statistically fine, but a member then receives different masks when it is
    trained alone.  This implementation gives each member/layer pair its own stream,
    making pack size and neighboring members irrelevant.
    """

    def __init__(
        self, probabilities: list[float], seeds: list[int], stream: int
    ) -> None:
        if len(probabilities) != len(seeds) or not probabilities:
            raise ValueError('Dropout probabilities and seeds must have equal size.')
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError('Dropout probabilities must be in [0, 1].')
        super().__init__()
        self.probabilities = tuple(float(value) for value in probabilities)
        self.seeds = tuple(stream_seed(seed, 10_000 + stream) for seed in seeds)
        self._generators: dict[tuple[int, str], torch.Generator] = {}

    @property
    def pack_size(self) -> int:
        return len(self.probabilities)

    def _generator(self, member: int, device: torch.device) -> torch.Generator:
        key = (member, str(device))
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(self.seeds[member])
            self._generators[key] = generator
        return generator

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if not self.training:
            return x
        members = (
            list(range(self.pack_size))
            if pack_idx is None
            else [int(value) for value in pack_idx.detach().cpu().tolist()]
        )
        if len(members) != x.shape[0]:
            raise RuntimeError('Dropout pack index does not match its input.')
        rows = []
        for row, member in enumerate(members):
            probability = self.probabilities[member]
            if probability == 0.0:
                rows.append(x[row : row + 1])
            elif probability == 1.0:
                rows.append(torch.zeros_like(x[row : row + 1]))
            else:
                keep = (
                    torch.rand(
                        x[row].shape,
                        device=x.device,
                        generator=self._generator(member, x.device),
                    )
                    >= probability
                )
                rows.append(x[row : row + 1] * keep.unsqueeze(0) / (1.0 - probability))
        return torch.cat(rows, dim=0)

    def remove(self, remove_indices: Tensor) -> None:
        remove = set(remove_indices.detach().cpu().tolist())
        keep = [index for index in range(self.pack_size) if index not in remove]
        old_generators = self._generators
        self.probabilities = tuple(self.probabilities[index] for index in keep)
        self.seeds = tuple(self.seeds[index] for index in keep)
        self._generators = {
            (new_index, device): old_generators[(old_index, device)]
            for new_index, old_index in enumerate(keep)
            for device in {key[1] for key in old_generators if key[0] == old_index}
        }


def remove_independent_dropout_members(
    model: torch.nn.Module, remove_indices: Tensor
) -> None:
    for module in model.modules():
        if isinstance(module, IndependentDropoutPack):
            module.remove(remove_indices)


def _model_arguments(
    members: list[dict[str, Any]],
    *,
    n_num_features: int,
    cat_cardinalities: list[int],
    n_classes: int | None,
    d_block: int,
    activation: str,
    native_padded: bool = False,
    max_d_embedding: int | None = None,
    max_n_blocks: int | None = None,
) -> dict[str, Any]:
    dimensions = [
        int(member['model']['num_embeddings']['d_embedding']) for member in members
    ]
    if not native_padded and len(set(dimensions)) != 1:
        raise ValueError(
            'A pure pack must have one d_embedding. Group configs by embedding width.'
        )
    n_blocks = [int(member['model']['n_blocks']) for member in members]
    if native_padded:
        max_d_embedding = (
            max(dimensions) if max_d_embedding is None else max_d_embedding
        )
        max_n_blocks = max(n_blocks) if max_n_blocks is None else max_n_blocks
        if max_d_embedding < max(dimensions):
            raise ValueError('max_d_embedding is smaller than an active member width.')
        if max_n_blocks < max(n_blocks):
            raise ValueError('max_n_blocks is smaller than an active member depth.')
        d_embedding: int | list[int] = dimensions
    else:
        d_embedding = dimensions[0]
        max_d_embedding = None
        max_n_blocks = max(n_blocks)
    return {
        'n_num_features': n_num_features,
        'cat_cardinalities': cat_cardinalities,
        'n_classes': n_classes,
        'pack_size': len(members),
        'num_embeddings': (
            None
            if n_num_features == 0
            else {
                'type': 'CosineEmbeddingsPack',
                'd_embedding': d_embedding,
                **(
                    {}
                    if max_d_embedding is None
                    else {'max_d_embedding': int(max_d_embedding)}
                ),
                'init_scale': [
                    float(member['model']['num_embeddings']['init_scale'])
                    for member in members
                ],
            }
        ),
        'n_blocks': n_blocks,
        'max_n_blocks': int(max_n_blocks),
        'd_block': int(d_block),
        'dropout': [float(member['model']['dropout']) for member in members],
        'activation': activation,
    }


def make_independent_model_pack(
    members: list[dict[str, Any]],
    *,
    n_num_features: int,
    cat_cardinalities: list[int],
    n_classes: int | None,
    d_block: int,
    activation: str,
    device: torch.device,
    loop_linear: bool = False,
    native_padded: bool = False,
    pack_invariant_dropout: bool = True,
    max_d_embedding: int | None = None,
    max_n_blocks: int | None = None,
) -> project.tabpack.ModelPack:
    """Initialize every packed member exactly through its singleton model."""
    if not members:
        raise ValueError('At least one member is required.')
    packed_arguments = _model_arguments(
        members,
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        d_block=d_block,
        activation=activation,
        native_padded=native_padded,
        max_d_embedding=max_d_embedding,
        max_n_blocks=max_n_blocks,
    )
    packed = project.tabpack.ModelPack(**packed_arguments)
    max_d_embedding = (
        int(packed_arguments['num_embeddings']['max_d_embedding'])
        if native_padded and packed_arguments['num_embeddings'] is not None
        else None
    )
    max_n_blocks = int(packed_arguments['max_n_blocks'])
    packed_state = packed.state_dict()
    for index, member in enumerate(members):
        torch.manual_seed(int(member['seed']))
        singleton = project.tabpack.ModelPack(
            **_model_arguments(
                [member],
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                n_classes=n_classes,
                d_block=d_block,
                activation=activation,
                native_padded=native_padded,
                max_d_embedding=max_d_embedding,
                max_n_blocks=max_n_blocks,
            )
        )
        singleton_state = singleton.state_dict()
        for name, source in singleton_state.items():
            target = packed_state.get(name)
            if target is None:
                raise RuntimeError(f'Singleton state is absent from the pack: {name}')
            if (
                source.ndim == 0
                or source.shape[0] != 1
                or target.shape[0] != len(members)
            ):
                raise RuntimeError(f'Unexpected non-packed state tensor: {name}')
            if source.shape[1:] != target.shape[1:]:
                raise RuntimeError(
                    f'Active member shape differs from packed shape for {name}: '
                    f'{tuple(source.shape)} != {tuple(target.shape)}'
                )
            target[index].copy_(source[0])
    packed.load_state_dict(packed_state)

    for layer, block in enumerate(packed.backbone._iter_blocks()):
        if pack_invariant_dropout:
            member_seeds = [int(member['seed']) for member in members]
            probabilities = [float(member['model']['dropout']) for member in members]
            block.dropout = IndependentDropoutPack(probabilities, member_seeds, layer)
        block.linear._loop = bool(loop_linear)
    packed.output._loop = bool(loop_linear)
    return packed.to(device)


def make_independent_batch_generators(
    members: list[dict[str, Any]], device: torch.device
) -> list[torch.Generator]:
    return [
        torch.Generator(device=device).manual_seed(stream_seed(int(member['seed']), 1))
        for member in members
    ]


def training_batches(
    train_size: int,
    batch_size: int,
    generators: list[torch.Generator],
    device: torch.device,
) -> list[Tensor]:
    indices = torch.stack(
        [
            torch.randperm(train_size, generator=generator, device=device)
            for generator in generators
        ]
    )
    return list(indices.split(batch_size, dim=1))


def native_training_batches(
    train_size: int,
    batch_size: int,
    pack_size: int,
    generator: torch.Generator,
) -> list[Tensor]:
    """Vectorized independent permutations using the native TabPack RNG stream."""
    random_values = torch.rand(
        (pack_size, train_size),
        generator=generator,
        device=generator.device,
    )
    return list(random_values.argsort(dim=1).split(batch_size, dim=1))
