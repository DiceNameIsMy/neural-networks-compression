# Taken & adapted from: https://github.com/itayhubara/BinaryNet.pytorch

import torch

from src.constants import DEVICE
from src.models.compression.conv import Conv2dWrapper
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


class FnConv2DQuantized(torch.autograd.Function):
    def __init__(self):
        super().__init__()
        self.com_num = 0
        self.weight_fp32 = None

    @staticmethod
    def forward(
        ctx,
        input,
        weight: torch.Tensor,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        nbits: int = 32,
    ):
        ctx.weight_fp32 = (
            weight.data.clone().detach()
        )  # save a copy of fp32 precision weight

        weight.data[:, :, :, :] = quantize(
            weight.data.clone().detach(), QMode.DET, nbits
        )[:, :, :, :]

        ctx.save_for_backward(input, weight, bias)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
            stride,
            padding,
            dilation,
            groups,
        )
        output = torch.nn.functional.conv2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = (
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )
        grad_input = grad_weight = grad_bias = None
        grad_stride = grad_padding = grad_dilation = grad_groups = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output, stride, padding, dilation, groups
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output, stride, padding, dilation, groups
            )
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3), dtype=None, keepdim=False).squeeze(0)

        ctx.saved_tensors[1].data[:, :, :, :] = ctx.weight_fp32[
            :, :, :, :
        ]  # recover the fp32 precision weight for parameter update

        return (
            grad_input,
            grad_weight,
            grad_bias,
            grad_stride,
            grad_padding,
            grad_dilation,
            grad_groups,
            None,
        )


class Conv2DQuantized(Conv2dWrapper):
    def __init__(self, nbits: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nbits = nbits

    def forward(self, x):
        return FnConv2DQuantized.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.nbits,
        )
