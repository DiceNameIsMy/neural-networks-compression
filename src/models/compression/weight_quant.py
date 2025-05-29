# Taken & adapted from: https://github.com/itayhubara/BinaryNet.pytorch

import torch

from src.constants import DEVICE
from src.models.compression.enums import QMode


class FnQuantize(torch.autograd.function.InplaceFunction):
    def forward(
        self, input: torch.Tensor, quant_mode=QMode.DET, numBits=4, inplace=False
    ):
        self.inplace = inplace
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = (2**numBits - 1) / (input.max() - input.min())

        output = output.mul(scale).clamp(-(2 ** (numBits - 1) + 1), 2 ** (numBits - 1))

        if quant_mode == QMode.DET:
            output = output.round().div(scale)
        else:
            noise = torch.rand(output.size()).to(DEVICE)
            output = output.add(-0.5).round().add(noise).div(scale)

        return output

    def backward(self, etain_graph=None, create_graph=False, inputs=None):
        # STE (Straight-Through Estimator)
        return (None, None, None)


def quantize(input, quant_mode, numBits) -> torch.Tensor:
    return FnQuantize.apply(input, quant_mode, numBits)  # type: ignore


class Quantize(torch.nn.Module):
    qmode: QMode
    num_bits: int

    def __init__(self, qmode: QMode, num_bits: int):
        super(Quantize, self).__init__()
        self.qmode = qmode
        self.num_bits = num_bits

    def forward(self, x):
        return quantize(x, self.qmode, self.num_bits)


class LinearQunatized(torch.nn.Linear):
    def __init__(self, nbits: int, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.nbits = nbits

    def forward(self, input):
        weight_b = quantize(self.weight, QMode.DET, self.nbits)
        out = torch.nn.functional.linear(input, weight_b)
        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            bias_b = quantize(self.bias, QMode.DET, self.nbits)
            out += bias_b

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"nbits={self.nbits}, "
            f"bias={self.bias is not None})"
        )
