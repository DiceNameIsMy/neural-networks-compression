# Taken & adapted from: https://github.com/itayhubara/BinaryNet.pytorch

import torch

from constants import DEVICE
from models.quant.enums import QMode


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
        # STE (Straight-Through Estimator)
        grad_input = grad_output
        return grad_input, None, None, None


def binarize(input, quant_mode):
    return Binarize.apply(input, quant_mode)


class Module_Binarize(torch.nn.Module):
    qmode: QMode

    def __init__(self, qmode: QMode):
        super(Module_Binarize, self).__init__()
        self.qmode = qmode

    def forward(self, x):
        return binarize(x, self.qmode)
