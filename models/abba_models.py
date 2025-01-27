import torch
from torch import nn
import einops
from einops.layers.torch import Rearrange, Reduce
from typing import Union, List, Tuple
import numpy as np
from deel import torchlip

from layers import *


def double_dense_init(layer: nn.Linear):
    out_units, in_units = layer.weight.shape

    w_init = torch.cat((torch.eye(in_units), -torch.eye(in_units)), dim=0)

    try:
        layer.bias.data.fill_(0)
        layer.weight.data.copy_(w_init)
    except:
        """layer doesn't have out_units = 2 * in_units"""
        pass


def halve_dense_init(layer: nn.Linear):
    out_units, in_units = layer.weight.shape

    w_init = torch.cat((torch.eye(out_units), -torch.eye(out_units)), dim=1)

    try:
        layer.bias.data.fill_(0)
        layer.weight.data.copy_(w_init)
    except:
        """layer doesn't have out_units = in_units // 2"""
        pass


class FullyDenseABBA(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 depths: list,
                 drop: float = 0.0,
                 use_bias: bool = True,
                 custom_init: str = None,
                 learn_double: bool = False,
                 learn_halve: bool = False,
                 first_projection: bool = True,
                 simplified: bool=False,
                 add_bn: bool=False,
                 add_last_dense: bool=True,
                 *args, **kwargs):
        """
        :param in_size:
        :param out_size:
        :param depths:
        :param drop:
        :param custom_init:
        :param learn_double_halve: bool, whether the first/last operators are learned or fixed, as in theory
        :param first_projection: bool, whether to first use a Linear projection on the input, with units = depth[0],
        if len(depths) > 0, or units = out_size else.
        :param simplified: bool, whether to use the simplified approach for computing the activations
        :param add_bn: bool, whether to add MyBatchNorm1D layers after each ABBA Dense or not (only for simplified=True)
        """

        super().__init__(*args, **kwargs)
        self.in_size = in_size
        self.out_size = out_size
        self.depths = depths
        self.drop = drop
        self.use_bias = use_bias
        self.custom_init = custom_init
        self.learn_double = learn_double
        self.learn_halve = learn_halve
        self.first_projection = first_projection
        self.simplified = simplified
        self.add_bn = add_bn
        self.add_last_dense = add_last_dense

        self.standard_layers = []
        "Define input & output operators"
        if self.first_projection:
            proj_size = out_size if len(depths) == 0 else depths[0]
            fp = nn.Linear(self.in_size, proj_size, bias=False)
            nn.init.orthogonal_(fp.weight)

            self.in_size = proj_size // 2

            self.standard_layers.append(fp)
        else:
            fp = nn.Identity()

        if self.first_projection or self.simplified:
            input_double = nn.Identity()
        else:
            if self.learn_double:
                input_double = nn.Linear(self.in_size, 2 * self.in_size)
                double_dense_init(input_double)
                
                self.standard_layers.extend([input_double])
            else:
                input_double = DoubleEntry(dense=True)

        if self.add_last_dense:
            abba_out_size = self.out_size if self.simplified else 2 * self.out_size
        else:
            abba_out_size = self.depths[-1] if self.simplified else 2 * self.depths[-1]

        if self.learn_halve:
            output_halve = nn.Linear(abba_out_size, self.out_size, bias=False)

            halve_dense_init(output_halve)
            # nn.init.xavier_uniform_(output_halve.weight)

            output_halve.name = "last_dense"
            self.standard_layers.extend([output_halve])
        else:
            if self.simplified:
                output_halve = nn.Identity()
            else:
                output_halve = OutputHalve(dense=True)

        self.input = nn.Sequential(fp, input_double)
        self.output = nn.Sequential(output_halve)
        self.abba_layers = []

        print(self.in_size)
        """Construct ABBA network"""
        if len(depths) == 0:
            abba_layer = ABBA_Dense_Layer(self.in_size, 
                                          self.out_size, 
                                          self.use_bias, 
                                          self.custom_init,
                                          self.simplified)
            self.abba_model = nn.Sequential(abba_layer)
            self.abba_layers.append(abba_layer)
        else:
            prev_d = self.in_size
            modules = []
            for d in self.depths:

                abba_layer = ABBA_Dense_Layer(prev_d, 
                                              d, 
                                              self.use_bias, 
                                              self.custom_init,
                                              simplified=self.simplified)
                
                if self.add_bn:
                    modules = modules + [
                        abba_layer,
                        MyBatchNorm1D(d),
                        CappedSymmetricLeakyReLU(),
                        nn.Dropout(p=self.drop)
                    ]
                else:
                    modules = modules + [
                        abba_layer,
                        CappedSymmetricLeakyReLU(auto_scale=True, beta=5),
                        # CappedLeakyyReLU(auto_scale=True, beta=5),
                        nn.Dropout(p=self.drop)
                    ]

                self.abba_layers.append(abba_layer)
                prev_d = d

            if self.add_last_dense:
                last_abba = ABBA_Dense_Layer(prev_d, 
                                            self.out_size, 
                                            self.use_bias, 
                                            self.custom_init,
                                            self.simplified)
                modules.append(last_abba)

                self.abba_layers.append(last_abba)

            self.abba_model = nn.Sequential(*modules)

        """Separate ABBA matrices from other params"""
        abba_matrices = []
        others = list(self.input.parameters()) + list(self.output.parameters())
        for m in self.abba_model:
            if isinstance(m, ABBA_Dense_Layer):
                abba_matrices.append(m.A)
                abba_matrices.append(m.B)
                if hasattr(m, "b"):
                    others.append(m.b)
            else:
                others = others + list(m.parameters())

        abba_matrices = nn.ParameterList(abba_matrices)
        others = nn.ParameterList(others)

        self.params = nn.ModuleDict({
            "others": others,
            "abba": abba_matrices
        })

    def forward(self, x):
        return self.output(self.abba_model(self.input(x)))


def double_conv_init(layer: nn.Conv2d):
    out_units, in_units, k1, k2 = layer.weight.shape

    w_init = torch.zeros_like(layer.weight.data)
    w_init[:out_units // 2, :, k1 // 2, k2 // 2] = 1
    w_init[out_units // 2:, :, k1 // 2, k2 // 2] = -1

    layer.bias.data.fill_(0)
    layer.weight.data.copy_(w_init)


class FullyConvABBA(nn.Module):
    def __init__(self,
                 in_ch: int,
                 inner_ch: List[int],
                 inner_k: Union[int, List[int]],
                 inner_s: Union[int, List[int]],
                 inner_p: Union[int, List[int], List[str]],
                 agg_op: List[Union[nn.Identity, nn.MaxPool2d, nn.AvgPool2d]],
                 drop: float = 0,
                 use_bias: bool = True,
                 learn_double: bool = False,
                 learn_halve: bool = False,
                 first_projection: bool = True,
                 out_ch: int = None,
                 add_last: bool = False,
                 simplified: bool = False,
                 add_bn: bool = False,
                 *args, **kwargs
                 ):
        """
        :param inner_ch: int or list, inner channel dimension(s)
        :param inner_k: int or list, inner kernel size(s)
        :param inner_s: int or List, inner stride(s)
        :param inner_p: int or List, inner padding(s)
        :param agg_op: sList of aggregation/pooling modules after each conv
        :param add_last: bool, whether to include an additional final layer with a
        predefined number of channels -- assumes out_ch is not None 
        :param learn_double_halve: bool, whether the input/output operators are set to learnable,
        as two convolutional layers
        :param simplified: bool, whether to use the simplified approach for computing the activations
        :param add_bn: bool, whether to add MyBatchNorm2D layers after each ABBA Conv or not (only for simplified=True)
        """
        super().__init__(*args, **kwargs)

        self.in_ch = in_ch
        self.inner_ch = inner_ch
        self.inner_k = inner_k
        self.inner_s = inner_s
        self.inner_p = inner_p
        self.agg_op = agg_op
        self.drop = drop
        self.use_bias = use_bias
        self.learn_double = learn_double
        self.learn_halve = learn_halve
        self.first_projection = first_projection
        self.add_last = add_last
        self.out_ch = out_ch
        self.simplified = simplified
        self.add_bn = add_bn

        if isinstance(self.inner_ch, int):
            self.inner_ch = [inner_ch]
        if isinstance(self.inner_k, int):
            self.inner_k = [self.inner_k] * len(self.inner_ch)
        if isinstance(self.inner_s, int):
            self.inner_s = [self.inner_s] * len(self.inner_ch)
        if isinstance(self.inner_p, int):
            self.inner_p = [self.inner_p] * len(self.inner_ch)

        assert len(self.inner_ch) <= len(self.inner_k), ValueError("Not enough values for k.")
        assert len(self.inner_ch) <= len(self.inner_s), ValueError("Not enough values for s.")
        assert len(self.inner_ch) <= len(self.inner_p), ValueError("Not enough values for p.")
        assert len(self.inner_ch) <= len(self.agg_op), ValueError("Not enough aggregators.")

        self.standard_layers = []
        "Define input & output operators"
        if self.first_projection and len(self.inner_ch) > 0:
            proj_ch = self.inner_ch[0]
            fp = nn.Conv2d(self.in_ch, proj_ch, (1, 1), (1, 1), padding=0, bias=False)

            if self.simplified:
                self.in_ch = proj_ch
            else:
                self.in_ch = proj_ch // 2   

            self.standard_layers.append(fp)
        else:
            fp = nn.Identity()

        if self.add_last:
            assert self.out_ch is not None
        else:
            self.out_ch = self.inner_ch[-1] if len(self.inner_ch) >= 1 else self.in_ch

        if self.first_projection or self.simplified or len(self.inner_ch) == 0:
            input_double = nn.Identity()
        else:
            if self.learn_double:
                input_double = nn.Conv2d(self.in_ch, 2 * self.in_ch, (1, 1), (1, 1), 0)
                double_conv_init(input_double)

                self.standard_layers.append(input_double)
            else:
                input_double = DoubleEntry(dense=False)

        if self.learn_halve and len(self.inner_ch) > 0:
            if self.simplified:
                output_halve = nn.Conv2d(self.out_ch, self.out_ch, (1, 1), (1, 1), 0, bias=False)
            else:
                output_halve = nn.Conv2d(2 * self.out_ch, self.out_ch, (1, 1), (1, 1), 0, bias=False)

            nn.init.orthogonal_(output_halve.weight)

            self.standard_layers.append(output_halve)
        else:
            if self.simplified or len(self.inner_ch) == 0:
                output_halve = nn.Identity()
            else:
                output_halve = OutputHalve(dense=False)

        self.input = nn.Sequential(fp, input_double)
        self.output = nn.Sequential(output_halve)
        self.abba_layers = []
        modules = []

        prev_in_ch = self.in_ch
        for i in range(len(self.inner_ch)):
            abba_conv_layer = ABBA_Conv_Layer(prev_in_ch,
                                                self.inner_ch[i],
                                                self.inner_k[i],
                                                self.inner_s[i],
                                                self.inner_p[i],
                                                self.use_bias,
                                                simplified=self.simplified)

            self.abba_layers.append(abba_conv_layer)

            if self.add_bn:
                modules = modules + [
                    abba_conv_layer,
                    MyBatchNorm2D(self.inner_ch[i]), 
                    CappedSymmetricLeakyReLU(),
                    nn.Dropout(p=self.drop),
                    self.agg_op[i]
                ]
            else:
                modules = modules + [
                    abba_conv_layer,
                    CappedSymmetricLeakyReLU(auto_scale=True, beta=5),
                    nn.Dropout(p=self.drop),
                    self.agg_op[i],
                ]

            if isinstance(self.agg_op[i], torchlip.InvertibleDownSampling):
                prev_in_ch = self.inner_ch[i] * self.agg_op[i].kernel_size[0] * self.agg_op[i].kernel_size[1]
            else:
                prev_in_ch = self.inner_ch[i]

        if self.add_last:
            last_abba = ABBA_Conv_Layer(prev_in_ch, self.out_ch, 1, 1, p=0, bias=False)
            self.abba_layers.append(last_abba)
            modules.append(last_abba)

        self.abba_model = nn.Sequential(*modules)

        """Separate ABBA matrices from other params"""
        abba_matrices = []
        others = list(self.input.parameters()) + list(self.output.parameters())
        for m in self.abba_model:
            if isinstance(m, ABBA_Conv_Layer):
                abba_matrices.append(m.A)
                abba_matrices.append(m.B)
                if hasattr(m, "b"):
                    others.append(m.b)
            else:
                others = others + list(m.parameters())

        abba_matrices = nn.ParameterList(abba_matrices)
        others = nn.ParameterList(others)

        self.params = nn.ModuleDict({
            "others": others,
            "abba": abba_matrices
        })

    def forward(self, x):
        inpt = self.input(x)
        abba = self.abba_model(inpt)
        outpt = self.output(abba)
        # return self.output(self.abba_model(self.input(x)))
        return outpt


class ConvDenseABBA(nn.Module):
    def __init__(self,
                 agg_op: List[Union[nn.Identity, nn.MaxPool2d, nn.AvgPool2d]],
                 in_ch: int,
                 n_classes: int,
                 inner_ch: List[int],
                 inner_k: Union[int, List[int]],
                 inner_s: Union[int, List[int]],
                 inner_p: Union[int, List[int], List[str]],
                 depths_dense: list,
                 hw: Tuple[int, int] = (None, None),
                 drop_conv: float = 0.0,
                 use_bias_conv: bool = True,
                 learn_halve_conv: bool = False,
                 learn_double_conv: bool = False,
                 first_projection: bool = False,
                 drop_dense: float = 0.0,
                 use_bias_dense: bool = True,
                 learn_halve_dense: bool = False,
                 learn_double_dense: bool = False,
                 global_pool: bool = True,
                 simplified: bool = False,
                 add_bn_dense: bool = False,
                 add_bn_conv: bool = False,
                 add_last_conv: bool = False,
                 add_last_dense: bool = True,
                 *args, **kwargs):
        """
        :param simplified: whether to use the simplified approach for computing the activations
        :param add_bn_dense/conv: bools, whether to add MyBatchNorm1D/2D layers after each ABBA Dense/Conv layer
        :param add_last_conv: bool, whether another 1x1 ABBA Conv Layer is added after the usual sequence
        """
        super().__init__(*args, **kwargs)

        self.in_ch = in_ch
        self.h, self.w = hw

        if not global_pool and (self.h is None or self.w is None):
            raise ValueError("Need input spatial dimension to be able to apply Flatten.")

        self.conv_net = FullyConvABBA(
            in_ch=in_ch, out_ch=inner_ch[-1] if len(inner_ch) >= 1 else None, inner_ch=inner_ch, inner_k=inner_k, inner_s=inner_s,
            inner_p=inner_p, agg_op=agg_op, drop=drop_conv, use_bias=use_bias_conv,
            learn_halve=learn_halve_conv, learn_double=learn_double_conv, first_projection=first_projection,
            simplified=simplified, add_bn=add_bn_conv, add_last=add_last_conv
        )
        abba = nn.ParameterList().extend(self.conv_net.params["abba"])
        others = nn.ParameterList().extend(self.conv_net.params["others"])

        if global_pool:
            self.global_pooling = Reduce("b c h w -> b c 1 1", "max")
            self.middle_dim = inner_ch[-1]
        else:
            """Define some flatten"""
            self.global_pooling = Rearrange("b c h w -> b (c h w)")
            self.middle_dim = self.compute_out_conv_dim()

        self.dense_net = FullyDenseABBA(
            in_size=self.middle_dim, out_size=n_classes, depths=depths_dense, drop=drop_dense,
            use_bias=use_bias_dense, learn_halve=learn_halve_dense, learn_double=learn_double_dense,
            first_projection=False, simplified=simplified, add_bn=add_bn_dense, add_last_dense=add_last_dense
        )

        abba = abba.extend(self.dense_net.params["abba"])
        others = others.extend(self.dense_net.params["others"])

        self.params = nn.ModuleDict({
            "abba": abba,
            "others": others
        })

    def forward(self, x):
        conv = self.conv_net(x)
        flat = self.global_pooling(conv).squeeze()
        out = self.dense_net(flat)

        return out
    
    def compute_out_conv_dim(self):

        x = torch.randn(1, self.in_ch, self.h, self.w)
        out = self.conv_net(x)

        return np.prod(out.shape[1:])


def get_intermediate_outputs(abba_model, dataloader, device="cuda"):
    """
    Returns a dictionary containing averaged activations for each ABBA layer.

    :param abba_model: an instance of ConvDenseABBA
    :param dataloaer: some data to compute activations for
    """

    features = {}

    def get_features(name):
        def hook(model, input, output):
            if name in features.keys():
                features[name] += output.detach().mean(dim=0).flatten()
            else:
                features[name] = output.detach().mean(dim=0).flatten()

        return hook
    
    for i, layer in enumerate(abba_model.conv_net.abba_layers):
        layer.register_forward_hook(get_features(f"{layer.__class__.__name__}_{i}"))
        print(f"{layer.__class__.__name__}_{i}", torch.sum(layer.A.data * layer.B.data))
    for i, layer in enumerate(abba_model.dense_net.abba_layers):
        layer.register_forward_hook(get_features(f"{layer.__class__.__name__}_{i}"))
        print(f"{layer.__class__.__name__}_{i}", torch.sum(layer.A.data * layer.B.data))

    abba_model.conv_net.output.register_forward_hook(get_features("Conv-out"))
    abba_model.dense_net.output.register_forward_hook(get_features("Dense-out"))
    abba_model.dense_net.input.register_forward_hook(get_features("Dense-in"))

    for batch in dataloader:
        x, y = batch
        out = abba_model(x.to(device))

    for k in features.keys():
        features[k] /= len(dataloader)

    return features