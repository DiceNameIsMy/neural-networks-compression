# Taken & adapted from: https://github.com/Thinklab-SJTU/twns

import torch

from src.models.compression.conv import Conv2dWrapper


def Alpha(tensor: torch.Tensor, delta):
    Alpha = []
    for i in range(tensor.size()[0]):
        count = 0
        abssum = 0
        absvalue = tensor[i].view(1, -1).abs()
        if isinstance(delta, int):
            truth_value = absvalue > delta
        else:
            truth_value = absvalue > delta[i]

        count = truth_value.sum()
        # print (count, truth_value.numel())
        abssum = torch.matmul(absvalue, truth_value.to(torch.float32).view(-1, 1))
        Alpha.append(abssum / count)

    alpha = torch.cat(Alpha, dim=0)
    return alpha


def Delta(tensor: torch.Tensor):
    n = tensor[0].nelement()
    if len(tensor.size()) == 4:  # convolution layer
        delta = 0.75 * torch.sum(tensor.abs(), dim=(1, 2, 3)) / n
    elif len(tensor.size()) == 2:  # fc layer
        delta = 0.75 * torch.sum(tensor.abs(), dim=(1,)) / n
    else:
        raise ValueError("Unknown tensor size")
    return delta


def binarize(tensor: torch.Tensor):
    output = torch.zeros(tensor.size(), device=tensor.device)
    delta = 0
    alpha = Alpha(tensor, delta)
    for i in range(tensor.size()[0]):
        pos_one = (tensor[i] > delta).to(torch.float32)
        neg_one = pos_one - 1
        out = torch.add(pos_one, neg_one)
        output[i] = torch.add(output[i], torch.mul(out, alpha[i]))

    return output


def ternarize(tensor: torch.Tensor):
    output = torch.zeros(tensor.size(), device=tensor.device)
    delta = Delta(tensor)
    alpha = Alpha(tensor, delta)
    for i in range(tensor.size()[0]):
        pos_one = (tensor[i] > delta[i]).to(torch.float32)
        neg_one = -1 * (tensor[i] < -delta[i]).to(torch.float32)
        out = torch.add(pos_one, neg_one)
        output[i] = torch.add(output[i], torch.mul(out, alpha[i]))
    return output


class Conv2DFunctionQUAN(torch.autograd.Function):
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
        quan_mode="TERNARY",
    ):
        ctx.weight_fp32 = (
            weight.data.clone().detach()
        )  # save a copy of fp32 precision weight
        if quan_mode == "TERNARY":
            weight.data[:, :, :, :] = ternarize(weight.data.clone().detach())[
                :, :, :, :
            ]  # do ternarization
        elif quan_mode == "BINARY":
            weight.data[:, :, :, :] = binarize(weight.data.clone().detach())[
                :, :, :, :
            ]  # do binarization
        else:
            pass

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


class Conv2dTernary(Conv2dWrapper):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return Conv2DFunctionQUAN.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            "TERNARY",
        )


class Conv2dBinary(Conv2dWrapper):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return Conv2DFunctionQUAN.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            "BINARY",
        )


class Binarize(torch.nn.Module):
    """
    Module for binarizing the input.
    Uses variable alpha (output scaling) for each channel.
    """

    def forward(self, x):
        return binarize(x)


class Ternarize(torch.nn.Module):
    """
    Module for ternarizing the input.
    Uses variable alpha (output scaling) and delta (threshold value) for each channel.
    """

    def forward(self, x):
        return ternarize(x)


class LinearTernary(torch.nn.Linear):
    """
    Linear layer with ternarized weights and bias (if exists).
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def forward(self, input):
        weight_b = ternarize(self.weight)
        out = torch.nn.functional.linear(input, weight_b)
        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
