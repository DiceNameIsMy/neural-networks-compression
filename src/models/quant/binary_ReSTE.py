# Taken & adapted from: https://github.com/DravenALG/ReSTE

import math

import torch


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
    def __init__(self):
        # TODO: Make t and o configurable
        super(Module_Binarize_ReSTE, self).__init__()
        self.t = torch.tensor(1.5).float()
        self.o = torch.tensor(1).float()

    def forward(self, x):
        return Binarize_ReSTE.apply(x, self.t, self.o)
