# Taken & adapted from: https://github.com/itayhubara/BinaryNet.pytorch
import enum

import torch

from src.constants import DEVICE


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class ActivationFunc(enum.Enum):
    RELU = "relu"
    BINARIZE = "binarize"


class Binarize(torch.autograd.function.InplaceFunction):
    def forward(
        ctx, input: torch.Tensor, quant_mode=QMode.DET, allow_scale=False, inplace=False
    ):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == QMode.DET:
            output = output.div(scale).sign().mul(scale)
        else:
            noise = torch.rand(output.size()).to(DEVICE)
            output = (
                output.div(scale)
                .add_(1)
                .div_(2)
                .add_(noise.add(-0.5))
                .clamp_(0, 1)
                .round()
                .mul_(2)
                .add_(-1)
                .mul(scale)
            )
        return output

    def backward(ctx, grad_output):
        # STE
        grad_input = grad_output
        return grad_input, None, None, None


def binarize(input, quant_mode):
    return Binarize.apply(input, quant_mode)


class BinaryActivation(torch.nn.Module):
    qmode: QMode

    def __init__(self, qmode: QMode):
        super(BinaryActivation, self).__init__()
        self.qmode = qmode

    def forward(self, x):
        return binarize(x, self.qmode)


class Quantize(torch.autograd.function.InplaceFunction):
    def forward(
        ctx, input: torch.Tensor, quant_mode=QMode.DET, numBits=4, inplace=False
    ):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
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

    def backward(grad_output, etain_graph=None, create_graph=False, inputs=None):
        # STE (Straight-Through Estimator)
        return (None, None, None)


def quantize(input, quant_mode, numBits):
    return Quantize.apply(input, quant_mode, numBits)


class QuantizeLayer(torch.nn.Module):
    qmode: QMode
    num_bits: int

    def __init__(self, qmode: QMode, num_bits: int):
        super(QuantizeLayer, self).__init__()
        self.qmode = qmode
        self.num_bits = num_bits

    def forward(self, x):
        return quantize(x, self.qmode, self.num_bits)
