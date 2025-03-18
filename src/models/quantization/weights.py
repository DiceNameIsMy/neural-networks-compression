import torch

from constants import DEVICE
from models.quantization.enums import QMode


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
