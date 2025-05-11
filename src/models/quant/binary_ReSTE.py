# Taken & adapted from: https://github.com/DravenALG/ReSTE

import math

import torch
from torch.nn.functional import conv2d

from src.models.quant.conv import Conv2dWrapper


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        tmp = torch.ones_like(input).to(input.device)
        tmp[torch.abs(input) > 1] = 0
        grad_input = tmp * grad_output.clone()
        return grad_input, None, None


class Binarize_ReSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, t, o):
        ctx.save_for_backward(input, t, o)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, t, o = ctx.saved_tensors

        interval = 0.1

        tmp = torch.zeros_like(input)
        mask1 = (input <= t) & (input > interval)
        tmp[mask1] = (1 / o) * torch.pow(input[mask1], (1 - o) / o)
        mask2 = (input >= -t) & (input < -interval)
        tmp[mask2] = (1 / o) * torch.pow(-input[mask2], (1 - o) / o)
        tmp[(input <= interval) & (input >= 0)] = (
            approximate_function(interval, o) / interval
        )
        tmp[(input <= 0) & (input >= -interval)] = (
            -approximate_function(-interval, o) / interval
        )

        # calculate the final gradient
        grad_input = tmp * grad_output.clone()

        return grad_input, None, None


def approximate_function(x, o):
    if x >= 0:
        return math.pow(x, 1 / o)
    else:
        return -math.pow(-x, 1 / o)


class Module_Binarize_ReSTE(torch.nn.Module):
    def __init__(self, threshold: float = 1.5, o: float = 1):
        super().__init__()
        self.threshold = torch.tensor(threshold).float()
        self.o = torch.tensor(o).float()

    def forward(self, x: torch.Tensor):
        return Binarize_ReSTE.apply(x, self.threshold.to(x.device), self.o.to(x.device))


class Binary_ReSTE_Conv2d(Conv2dWrapper):
    def __init__(self, threshold: float = 1.5, o: float = 1, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.threshold = torch.tensor(threshold).float()
        self.o = torch.tensor(o).float()

    def forward(self, input):
        a0 = input
        w0 = self.weight

        # binarize
        bw = Binarize_ReSTE().apply(
            w0, self.threshold.to(w0.device), self.o.to(w0.device)
        )
        ba = a0

        # scaling factor
        scaler = torch.mean(torch.abs(w0), dim=(0, 1, 2, 3), keepdim=True)
        bw = bw * scaler

        # 1bit conv
        output = conv2d(
            ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return output
