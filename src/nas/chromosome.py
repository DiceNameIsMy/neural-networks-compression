from dataclasses import dataclass, field, fields
from typing import Generic, TypeVar

import numpy as np

from src.models.compression.enums import Activation, NNParamsCompMode, QMode

BITWIDTHS_MAPPING = (1, 2, 3, 4, 5, 6, 7, 8)

ACTIVATION_MAPPING = tuple(act for act in Activation)
RESTE_O_MAPPING = (1.5,)
RESTE_THRESHOLD_MAPPING = (3.0,)
QMODE_MAPPING = (QMode.DET,)
NN_PARAMS_COMP_MODE_MAPPING = tuple(q for q in NNParamsCompMode)

LEARNING_RATES_MAPPING = (
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
)
WEIGHT_DECAY_MAPPING = (
    0.01,
    0.005,
    0.002,
    0.001,
)
DROPOUT_MAPPING = (0.0, 0.1, 0.2)


def with_options(options: tuple):
    return field(metadata={"default_options": options})


@dataclass
class Chromosome:
    pass


T = TypeVar("T", bound="Chromosome")


class ChromosomeConfig(Generic[T]):
    ChromosomeCls: type[T]

    options: dict[str, tuple]
    options_override: dict[str, tuple]

    def __init__(
        self, ChromosomeCls: type[T], override: dict[str, tuple] | None = None
    ):
        self.ChromosomeCls = ChromosomeCls

        # Setup default options
        self.options = {}
        for f in fields(ChromosomeCls):
            if "default_options" not in f.metadata:
                raise ValueError(
                    f"Field '{f.name}' in {ChromosomeCls.__name__} must have "
                    + f"'default_options' metadata entry. Metadata: {f.metadata}"
                )
            options = f.metadata["default_options"]

            assert isinstance(options, tuple)
            self.options[f.name] = options

        # Setup option overrides
        self.options_override = override or {}

        for key, _ in self.options_override.items():
            if key not in self.options:
                raise ValueError(
                    f"Override key '{key}' not found in chromosome bounds."
                )

    def decode(self, encoded: np.ndarray) -> T:
        """Decode an encoded chromosome into a Chromosome instance."""
        idx = 0
        init_dict = {}
        for f in fields(self.ChromosomeCls):
            if f.name in self.options_override:
                options = self.options_override[f.name]
            else:
                options = self.options[f.name]

            index = self._get_from_encoded(encoded, idx)
            idx += 1

            init_dict[f.name] = self._get_value_from_options(options, index)

        if idx != len(encoded):
            raise ValueError(
                f"Chromosome too long. Expected length of `{len(fields(self.ChromosomeCls))}` but got `{len(encoded)}`."
            )

        return self.ChromosomeCls(**init_dict)

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = [], []
        for f in fields(self.ChromosomeCls):
            if f.name in self.options_override:
                options = self.options_override[f.name]
            else:
                options = self.options[f.name]

            low.append(0)
            high.append(len(options) - 1)

        return np.array(low), np.array(high)

    def get_size(self) -> int:
        """Get amount of genes contained in a chromosome"""
        return len(fields(self.ChromosomeCls))

    def _get_from_encoded(self, x: np.ndarray, idx: int) -> int:
        try:
            return x[idx]
        except IndexError:
            raise ValueError(
                "Encoded chromosome is too short. "
                + f"Expected at least {len(fields(self.ChromosomeCls))} values, got {len(x)}."
            )

    def _get_value_from_options(self, options: tuple, index: int):
        try:
            return options[index]
        except IndexError as e:
            raise ValueError(
                f"Encoded gene {index} for field with options {options} is out of bounds."
            ) from e
