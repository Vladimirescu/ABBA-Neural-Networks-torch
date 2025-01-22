import torch
import torch.nn as nn


class MyBatchNorm1D(nn.Module):
    """
    Custom BN for 1D: uses running mean/variance in both training and testing ->
    these running statistics are used to account for the final Lipschitz of the network,
    therefore even the training has to be performed using the running instead of the current batch statistics
    """
    def __init__(self, num_features, 
                 momentum=0.01, eps=1e-5, gamma_min=1e-9, gamma_max=0.1, affine=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.affine = affine

        self.gamma = nn.Parameter(torch.ones(1, num_features), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, num_features), requires_grad=self.affine)
        
        # Running mean and variance
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.ones(1, num_features))
        
    def forward(self, x):
        if self.training:            
            with torch.no_grad():
                mean = torch.mean(x, dim=0, keepdim=True)
                var = torch.var(x, dim=0, correction=0, keepdim=True)

                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        else:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # Constrain gamma
        if self.affine:
            self.gamma.data.clamp_(self.gamma_min, self.gamma_max)

        return self.gamma * x + self.beta


class MyBatchNorm2D(nn.Module):
    """
    Custom BN for 2D: same as MyBatchNorm1D for 2D case
    """
    def __init__(self, num_features, 
                 momentum=0.001, eps=1e-5, affine=True, lip_max_bn=3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.lip_max_bn = lip_max_bn
        
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1), requires_grad=self.affine)

        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
                var = torch.var(x, dim=(0, 2, 3), correction=0, keepdim=True)
                
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        else:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # Constrain gamma
        if self.affine:
            self.gamma.data.clamp_(self.running_mean, self.lip_max_bn * self.running_mean)
        
        return self.gamma * x + self.beta


class ScaledAvgPool2D(nn.Module):
    """
    2D Pooling Layer with Lip const = 1.
    """
    def __init__(self, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(self.kernel_size)

    def forward(self, x):

        return self.pool(x) * self.kernel_size


class L2NormPool2D(nn.Module):
    def __init__(self, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(self.kernel_size)

    def forward(self, x):

        return torch.sqrt(self.pool(x**2))