import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable


class MyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        """
        x:       (N, C_in, H, W)
        weight:  (C_out, C_in, kH, kW)
        bias:    (C_out,) or None
        stride, padding, dilation: int
        """
        if padding:
            x = F.pad(x, [padding] * 4)

        ctx.save_for_backward(x, weight)
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.stride = stride

        N, C_in, H, W = x.shape
        C_out, C_in, kH, kW = weight.shape

        H_out = (H - kH - (kH - 1) * (dilation - 1)) // stride + 1
        W_out = (W - kW - (kW - 1) * (dilation - 1)) // stride + 1

        ctx.H_out = H_out
        ctx.W_out = W_out

        out = torch.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

        for i in range(N):
            for j in range(C_out):
                for k in range(H_out):
                    for l in range(W_out):
                        out[i, j, k, l] = (
                            weight[j]
                            * x[
                                i,
                                :,
                                k * stride : k * stride
                                + kH
                                + (kH - 1) * (dilation - 1) : dilation,
                                l * stride : l * stride
                                + kW
                                + (kW - 1) * (dilation - 1) : dilation,
                            ]
                        ).sum()

                        if bias is not None:
                            out[i, j, k, l] += bias[j]

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (N, C_out, H_out, W_out)
        returns gradients for (x, weight, bias, stride, padding, dilation)
        """
        # Backward implementation here

        H_out = ctx.H_out
        W_out = ctx.W_out
        dilation = ctx.dilation
        stride = ctx.stride

        x, weight = ctx.saved_tensors

        N, C_in, H, W = x.shape
        C_out, C_in, kH, kW = weight.shape

        x_grad = torch.zeros_like(x)
        weight_grad = torch.zeros_like(weight)
        bias_grad = torch.zeros(grad_output.size(1))

        for i in range(N):
            for j in range(C_out):
                for k in range(H_out):
                    for l in range(W_out):
                        x_grad[
                            i,
                            :,
                            k * stride : k * stride
                            + kH
                            + (kH - 1) * (dilation - 1) : dilation,
                            l * stride : l * stride
                            + kW
                            + (kW - 1) * (dilation - 1) : dilation,
                        ] += grad_output[i, j, k, l].item() * weight[j]

                        weight_grad[j] += (
                            grad_output[i, j, k, l].item()
                            * x[
                                i,
                                :,
                                k * stride : k * stride
                                + kH
                                + (kH - 1) * (dilation - 1) : dilation,
                                l * stride : l * stride
                                + kW
                                + (kW - 1) * (dilation - 1) : dilation,
                            ]
                        )

                        bias_grad[j] += grad_output[i, j, k, l]

        x_grad = x_grad[:, :, ctx.padding : -ctx.padding, ctx.padding : -ctx.padding]

        return (
            Variable(x_grad),
            Variable(weight_grad),
            Variable(bias_grad),
            None,
            None,
            None,
        )


class MyConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        return MyConv2dFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )


# Test
device = "cuda" if torch.cuda.is_available() else "cpu"
B, C, H, W = 2, 1, 23, 28
C_out = 2

x = torch.randn(B, C, H, W, device=device, requires_grad=True)

myconv = MyConv2d(C, C_out, (3, 3), stride=4, padding=2, dilation=3, bias=True).to(
    device
)
target_conv = nn.Conv2d(
    C, C_out, (3, 3), stride=4, padding=2, dilation=3, bias=True
).to(device)

with torch.no_grad():
    target_conv.weight.copy_(myconv.weight)
    if myconv.bias is not None:
        target_conv.bias.copy_(myconv.bias)

# Forward check
y_my = myconv(x)
y_target = target_conv(x)
print(y_my.shape, y_target.shape)

print("Forward check:", torch.allclose(y_my, y_target, atol=1e-6, rtol=1e-5))

# Backward check
loss_my = y_my.square().mean()
loss_ref = y_target.square().mean()
loss_my.backward()
loss_ref.backward()

print(
    "w grad close:",
    torch.allclose(myconv.weight.grad, target_conv.weight.grad, atol=1e-6, rtol=1e-5),
)
if myconv.bias is not None:
    print(
        "b grad close:",
        torch.allclose(myconv.bias.grad, target_conv.bias.grad, atol=1e-6, rtol=1e-5),
    )
