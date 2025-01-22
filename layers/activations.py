import torch
import torch.nn as nn
import numpy as np


def soft_thresholding(x, alpha, beta):
    return alpha * nn.functional.relu(x - beta) - \
           alpha * nn.functional.relu(-x -beta)


class SoftThresholding(nn.Module):
    def __init__(self, 
                 alpha: float = 0.9,
                 beta: float = 0.,
                 auto_scale: bool=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = beta
        self.alpha = alpha
        self.auto_scale = auto_scale

        if auto_scale:
            self.beta = nn.Parameter(torch.ones(1) * beta, requires_grad=True)
            self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=True)

    def forward(self, x):
        if self.auto_scale:
            self.alpha.data.clamp_(1e-5, 1)
            self.beta.data.clamp_(1e-5)

        return soft_thresholding(x, self.alpha, self.beta)


class MultiSoftThresholding(nn.Module):
    """
    Activation containing multiple trainable soft_thresholding.
    """
    def __init__(self, N=5, *args, **kwargs):
        """
        :param N: int, number of soft_thresholding
        """
        super().__init__(*args, **kwargs)

        assert N >= 2, ValueError("Number of soft_thresholding should be >=2.")

        self.N = N
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.randn(1), requires_grad=True) for _ in range(N)]
        )
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.randn(1), requires_grad=True) for _ in range(N)]
        )

        # Centering parameters - i.e., shift value
        self.cetas = nn.ParameterList(
            [nn.Parameter(torch.randn(1), requires_grad=True) for _ in range(N)]
        )

    def forward(self, x):
        self.betas[0].data.clamp_(0)
        y = soft_thresholding(x + self.cetas[0], self.alphas[0], self.betas[0])

        for i in range(1, self.N):
            self.betas[i].data.clamp_(0)
            y = y + soft_thresholding(x + self.cetas[i], self.alphas[i], self.betas[i])

        return y / sum(abs(alpha.item()) for alpha in self.alphas)


def tri_func(x, alpha, beta):
    """
    Base triangle function, with non-zero domain [x - beta, x + beta], 
    height alpha, and slope +/- alpha.

    :param a: slope of the triangle
    :param b: width of the triangle
    """

    return alpha * (
        nn.functional.relu(x - beta) - 2 * nn.functional.relu(x) + nn.functional.relu(x + beta)
    )


class MultiTri(nn.Module):
    """
    Activation containing multiple trainable tri_func.
    """
    def __init__(self, N=10, *args, **kwargs):
        """
        :param N: int, number of tri_func
        """
        super().__init__(*args, **kwargs)

        assert N >= 2, ValueError("Number of tri_func should be >=2.")

        self.N = N if N % 2 == 1 else N + 1
        self.alphas = nn.Parameter(torch.rand(N), requires_grad=True)

        # triangle width
        self.beta = 0.1

        # data scale
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

        # bias
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        self.alphas.data.clamp_(-1, 1)

        x = x * self.scale

        y = 0
        for i in range(-self.N//2 + 1, self.N//2 + 1):
            # add non-overlapping triangle functions
            y = y + tri_func(x - 2 * self.beta * i, self.alphas[i], self.beta)

        y = y / self.scale + self.bias

        return y


class ABS(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.abs(x)


class CappedLeakyyReLU(nn.Module):
    def __init__(self,
                 alpha: float = 0.1,
                 beta: float = 5,
                 auto_scale: bool=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.auto_scale = auto_scale

        if self.auto_scale:
            self.beta = nn.Parameter(torch.ones(1) * beta, requires_grad=True)
            self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=True)

    def forward(self, x):
        if self.auto_scale:
            self.alpha.data.clamp_(0, 1)
            self.beta.data.clamp_(0)

        y = torch.maximum(self.alpha * x, x)
        z = torch.where(y < self.beta, y, (y - self.beta) * self.alpha + self.beta)

        return z


class CappedSymmetricLeakyReLU(nn.Module):
    def __init__(self,
                 alpha: float = 0.1,
                 beta: float = 5,
                 auto_scale: bool=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = beta
        self.alpha = alpha
        self.auto_scale = auto_scale

        if self.auto_scale:
            self.beta = nn.Parameter(torch.ones(1) * self.beta, requires_grad=True)
            self.alpha = nn.Parameter(torch.ones(1) * self.alpha, requires_grad=True)

    def forward(self, x):
        if self.auto_scale:
            self.alpha.data.clamp_(0, 1)
            self.beta.data.clamp_(0)

        y = torch.maximum(self.alpha * (x + self.beta), x + self.beta)
        z = torch.where(y < self.beta * 2, y, (y - self.beta * 2) * self.alpha + self.beta * 2) - self.beta

        return z


class MultiCappedSymmetricLeakyReLU(nn.Module):
    """
    Activation functions containing multiple trainable CappedSymmetricLeakyReLU.
    Tries to recreate a PieceWiseLinear activation function, which is Symmetric w.r.t. origin.
    """
    def __init__(self, N, *args, **kwargs):
        """
        :param N: int, number of CappedSymmetricLeakyReLU
        """
        super().__init__(*args, **kwargs)

        assert N >= 2, ValueError("Number of CappedSymmetricLeakyReLUs should be >=2.")

        self.N = N
        self.fs = nn.ModuleList([
            CappedSymmetricLeakyReLU(
                alpha = torch.randn(1),
                beta = torch.randn(1) * 3, 
                auto_scale=True
            ) for i in range(self.N)
        ])

    def forward(self, x):
        y = self.fs[0](x)
        for i in range(1, self.N):
            y = y + self.fs[i](x)

        return y / self.N
    

class SignSqrt(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x - torch.sign(x) * (torch.sqrt(1 + x**2) - 1)
    

class ModifiedTanh(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.a = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x):
        self.a.data.clamp_(0.1, 1)

        return (nn.functional.tanh(x) + self.a * x) / (1 + self.a)
    

class ModifiedSigmoid(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.a = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x):
        self.a.data.clamp_(0.1, 1)

        return (x + self.a * torch.sin(1 / self.a * x)) / 2


class InverseCLR(nn.Module):
    def __init__(self, 
                 beta: float = 1,
                 auto_scale: bool=False,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.beta = beta
        self.scale = 1
        self.auto_scale = auto_scale

        if self.auto_scale:
            self.beta = nn.Parameter(torch.ones(1) * self.beta, requires_grad=True)
            self.scale = nn.Parameter(torch.ones(1) * self.scale, requires_grad=True)

    def forward(self, x):
        if self.auto_scale:
            self.beta.data.clamp_(0)
            self.scale.data.clamp_(0.01)

        x = self.scale * x

        z = torch.where(torch.abs(x) < self.beta, x, -x + 2 * self.beta * torch.sign(x))

        return z / self.scale