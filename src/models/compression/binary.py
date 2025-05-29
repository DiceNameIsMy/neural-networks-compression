# Taken & adapted from: https://github.com/itayhubara/BinaryNet.pytorch

import torch

from src.models.compression.enums import QMode


class FnBinarize(torch.autograd.function.InplaceFunction):
    def forward(
        self,
        input: torch.Tensor,
        quant_mode=QMode.DET,
        allow_scale=False,
        inplace=False,
    ):
        self.inplace = inplace
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == QMode.DET:
            output = output.div(scale).sign().mul(scale)
        else:
            noise = torch.rand(output.size()).to(output.device)
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

    def backward(self, grad_output):
        # STE (Straight-Through Estimator)
        grad_input = grad_output
        return grad_input, None, None, None


def binarize(input, quant_mode=QMode.DET):
    return FnBinarize.apply(input, quant_mode)


class Binarize(torch.nn.Module):
    """
    Module for binarizing the input.
    """

    qmode: QMode

    def __init__(self, qmode=QMode.DET):
        super(Binarize, self).__init__()
        self.qmode = qmode

    def forward(self, x):
        return binarize(x, self.qmode)

    def __repr__(self):
        return f"Module_Binarize(qmode={self.qmode})"


class LinearBinary(torch.nn.Linear):
    """
    Linear layer with binarized weights and bias (if exists).
    """

    def __init__(self, *kargs, **kwargs):
        super(LinearBinary, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        weight_b = binarize(self.weight, QMode.DET)
        out = torch.nn.functional.linear(input, weight_b)
        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )
