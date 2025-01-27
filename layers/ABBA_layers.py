from torch import nn
import torch
import einops


class DoubleEntry(nn.Module):
    def __init__(self,  dense=True, *args, **kwargs):
        """
        :param dense: bool, whether the following layer is a dense or conv
        """
        super().__init__(*args, **kwargs)
        self.dense = dense

    def forward(self, x):
        if self.dense:
            return torch.cat((x, -x), dim=-1)
        else:
            return torch.cat((x, -x), dim=1)


class OutputHalve(nn.Module):
    def __init__(self, dense=True, *args, **kwargs):
        """
        :param dense: bool, whether the previous layer is a dense or conv
        """
        super().__init__(*args, **kwargs)
        self.dense = dense

    def forward(self, x):
        if self.dense:
            dim = -1
        else:
            dim = 1
        x1, x2 = torch.chunk(x, 2, dim=dim)

        assert x1.shape[dim] == x2.shape[dim], \
            ValueError(f"Output halves shape mismatch: {x1.shape} vs {x2.shape}")

        return (x1 - x2) / 2.0


class ABBA_Dense_Layer(nn.Module):
    def __init__(self,
                 in_units: int,
                 units: int,
                 bias: bool = True,
                 custom_init: str = None,
                 simplified: bool = False,
                 *args, **kwargs):
        """
        ABBA Dense layer, accepting as input tensors of shape (B, 2 * in_units)
        and returning outputs of shape (B, 2 * units). Both A & B matrices have shape (in_units, units).
        
        :param in_units: size of input vector
        :param units: size of output vectors, i.e. the final output is 2 * units
        :param bias: if True, apply bias
        :param custom_init: string, defines what custom initialization to use for A/B
        :param simplified: bool, whether to use a simplified approach for computing the output
        """
        super().__init__(*args, **kwargs)

        self.in_units = in_units
        self.units = units
        self.bias = bias
        self.simplified = simplified

        self.A = nn.Parameter(torch.Tensor(in_units, units), requires_grad=True)
        self.B = nn.Parameter(torch.Tensor(in_units, units), requires_grad=True)

        if custom_init:
            raise ValueError(f"Custom initialization {custom_init} not yet supported.")
        else:
            xu = torch.empty(in_units, units)
            torch.nn.init.orthogonal_(xu, gain=1)  # works best

            xu_plus = (torch.abs(xu) + xu) / 2.0
            xu_minus = xu_plus - xu
            with torch.no_grad():
                self.A.copy_(xu_plus)
                self.B.copy_(xu_minus)

        if self.bias:
            if self.simplified:
                self.b = nn.Parameter(torch.Tensor(units), requires_grad=True)
            else:
                self.b = nn.Parameter(torch.Tensor(units * 2), requires_grad=True)
            self.b.data.fill_(0.0)

    def forward(self, x):

        if self.simplified:
            assert x.shape[-1] == self.in_units, \
                ValueError(f"Input size of {x.shape[-1]} does not correspond to expected size of {self.in_units}")
            
            out = torch.matmul(x, self.A - self.B)
        else:
            assert x.shape[-1] == 2 * self.in_units, \
                ValueError(f"Input size of {x.shape[-1]} does not correspond to expected size of 2 x {self.in_units}")

            ab = torch.matmul(x[..., :self.in_units], self.A) + torch.matmul(x[..., self.in_units:], self.B)
            ba = torch.matmul(x[..., :self.in_units], self.B) + torch.matmul(x[..., self.in_units:], self.A)

            out = torch.cat((ab, ba), dim=-1)

        if self.bias:
            return out + self.b
        else:
            return out


class ABBA_Conv_Layer(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 k: int, 
                 s: int = 1,
                 p: str='same', 
                 bias: bool=True, 
                 custom_init: str=None, 
                 simplified: bool=False,
                 *args, **kwargs):
        """
        ABBA Conv layer, accepting as input tensors of shape (B, 2 * in_ch, *, *)
        and returning outputs of shape (B, 2 * out_ch, *, *). 
        Both A & B matrices have shape (out_ch, in_ch, k, k).

        :param in_ch: int, input channels to A/B -> input has 2 * in_ch input channels
        :param out_ch: int, output channels/#filters for A/B -> output has 2 * out_ch output channels
        :param k: int, kernel size
        :param s: int, stride for A/B
        :param bias: bool, whether this layer applies some bias operator
        :param simplified: bool, whether to use a simplified approach for computing the output
        """
        super().__init__(*args, **kwargs)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.s = s
        self.p = p
        self.bias = bias
        self.simplified = simplified

        self.A = nn.Parameter(torch.Tensor(self.out_ch, self.in_ch, self.k, self.k), requires_grad=True)
        self.B = nn.Parameter(torch.Tensor(self.out_ch, self.in_ch, self.k, self.k), requires_grad=True)

        if custom_init:
            raise ValueError(f"Custom initialization {custom_init} not yet supported.")
        else:
            # higher gain helps in deeper networks
            xu = torch.empty(self.out_ch, self.in_ch, self.k, self.k)
            torch.nn.init.orthogonal_(xu, gain=1)
            ###            

            xu_plus = (torch.abs(xu) + xu) / 2.0
            xu_minus = xu_plus - xu
            with torch.no_grad():
                self.A.copy_(xu_plus)
                self.B.copy_(xu_minus)

        if self.bias:
            if self.simplified:
                self.b = nn.Parameter(torch.Tensor(1, self.out_ch, 1, 1), requires_grad=True)
            else:
                self.b = nn.Parameter(torch.Tensor(1, self.out_ch * 2, 1, 1), requires_grad=True)
            self.b.data.fill_(0.0)

    def forward(self, x):
        """
        :param x: input of shape (batch, self.in_ch, height, width)
        :return: output of shape (batch, self.out_ch, height', width')
        """

        if self.simplified:
            assert x.shape[1] == self.in_ch, \
                ValueError(f"Input channels of {x.shape[-1]} does not correspond to expected channels {self.in_ch}")
        
            out = torch.nn.functional.conv2d(x, (self.A - self.B), stride=self.s, padding=self.p)
        else:
            assert x.shape[1] == 2 * self.in_ch, \
                ValueError(f"Input channels of {x.shape[-1]} does not correspond to expected channels 2 x {self.in_ch}")

            # ab = torch.nn.functional.conv2d(x[:, :self.in_ch], self.A, stride=self.s, padding=self.p) + \
            #     torch.nn.functional.conv2d(x[:, self.in_ch:], self.B, stride=self.s, padding=self.p)

            # ba = torch.nn.functional.conv2d(x[:, :self.in_ch], self.B, stride=self.s, padding=self.p) + \
            #     torch.nn.functional.conv2d(x[:, self.in_ch:], self.A, stride=self.s, padding=self.p)

            x = nn.functional.pad(x, (self.p, self.p, self.p, self.p), mode="circular")

            ab = torch.nn.functional.conv2d(x[:, :self.in_ch], self.A, stride=self.s) + \
                torch.nn.functional.conv2d(x[:, self.in_ch:], self.B, stride=self.s)

            ba = torch.nn.functional.conv2d(x[:, :self.in_ch], self.B, stride=self.s) + \
                torch.nn.functional.conv2d(x[:, self.in_ch:], self.A, stride=self.s)

            out = torch.cat((ab, ba), dim=1)

        if self.bias:
            return out + self.b
        else:
            return out

