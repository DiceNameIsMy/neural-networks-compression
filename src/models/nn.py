from dataclasses import dataclass

from src.models.quant.enums import ActivationModule, QMode


@dataclass
class ActivationParams:
    activation: ActivationModule
    binary_qmode: QMode
    reste_o: float = 1
    reste_threshold: float = 1.5
