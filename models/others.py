from torch import nn
import einops
from einops.layers.torch import Rearrange, Reduce
from typing import Union, List, Tuple

from layers.activations import *


class ConvDenseStandard(nn.Module):
    def __init__(self,
                 in_ch: int,
                 n_classes: int,
                 inner_ch: List[int],
                 inner_k: Union[int, List[int]],
                 inner_s: Union[int, List[int]],
                 inner_p: Union[int, List[int], List[str]],
                 agg_op: List[Union[nn.Identity, nn.MaxPool2d, nn.AvgPool2d]],
                 depths_dense: list,
                 drop_conv: float = 0.0,
                 use_bias_conv: bool = True,
                 drop_dense: float = 0.0,
                 use_bias_dense: bool = True,
                 hw: Tuple[int, int] = (None, None),
                 global_pool: bool=False,
                 *args, **kwargs
                 ):

        super().__init__(*args, **kwargs)

        self.in_ch = in_ch
        self.h, self.w = hw
        self.conv = []
        self.dense = []

        if isinstance(inner_k, int):
            inner_k = [inner_k] * len(inner_ch)
        if isinstance(inner_s, int):
            inner_s = [inner_s] * len(inner_ch)
        if isinstance(inner_p, int):
            inner_p = [inner_p] * len(inner_ch)

        prev_ch = in_ch
        for i in range(len(inner_ch)):
            self.conv = self.conv + [
                nn.Conv2d(prev_ch, inner_ch[i], inner_k[i], inner_s[i], inner_p[i], bias=use_bias_conv),
                CappedSymmetricLeakyReLU(),
                nn.Dropout(p=drop_conv),
                agg_op[i]
            ]

            prev_ch = inner_ch[i]

        self.conv = nn.Sequential(*self.conv)

        if global_pool:
            self.global_pooling = Reduce("b c h w -> b c", "max")
            prev_ch = inner_ch[-1]
        else:
            """Define some flatten"""
            self.global_pooling = Rearrange("b c h w -> b (c h w)")
            prev_ch = self.compute_out_conv_dim()

        prev_d = prev_ch
        for i in range(len(depths_dense)):
            self.dense = self.dense + [
                nn.Linear(prev_d, depths_dense[i], bias=use_bias_dense),
                CappedSymmetricLeakyReLU(auto_scale=True),
                nn.Dropout(p=drop_dense)
            ]
            prev_d = depths_dense[i]

        self.dense.append(
            nn.Linear(prev_d, n_classes, bias=False)
        )

        self.network = nn.Sequential(
            *self.conv, self.global_pooling, *self.dense
        )

    def forward(self, x):
        return self.network(x)

    def compute_out_conv_dim(self):

        x = torch.randn(1, self.in_ch, self.h, self.w)
        out = self.conv(x)

        return np.prod(out.shape[1:])


class ConvDenseDeel(nn.Module):
    def __init__(self,
                 in_ch: int,
                 n_classes: int,
                 inner_ch: List[int],
                 inner_k: Union[int, List[int]],
                 inner_s: Union[int, List[int]],
                 inner_p: Union[int, List[int], List[str]],
                 agg_op: List[Union[nn.Identity, nn.MaxPool2d, nn.AvgPool2d]],
                 depths_dense: list,
                 drop_conv: float = 0.0,
                 use_bias_conv: bool = True,
                 drop_dense: float = 0.0,
                 use_bias_dense: bool = True,
                 lip: float = 1,
                 hw: Tuple[int, int] = (None, None),
                 global_pool: bool=False,
                 *args, **kwargs
                 ):

        super().__init__(*args, **kwargs)

        self.in_ch = in_ch
        self.h, self.w = hw
        self.conv = []
        self.dense = []

        if isinstance(inner_k, int):
            inner_k = [inner_k] * len(inner_ch)
        if isinstance(inner_s, int):
            inner_s = [inner_s] * len(inner_ch)
        if isinstance(inner_p, int):
            inner_p = [inner_p] * len(inner_ch)

        if len(inner_ch) > 0:
            prev_ch = in_ch
            for i in range(len(inner_ch)):
                self.conv = self.conv + [
                    torchlip.SpectralConv2d(prev_ch, inner_ch[i], inner_k[i], inner_s[i], inner_p[i], 
                                            bias=use_bias_conv),
                    CappedSymmetricLeakyReLU(auto_scale=True),

                    nn.Dropout(p=drop_conv),
                    agg_op[i]
                ]

                prev_ch = inner_ch[i]
        else:
            self.conv = [nn.Identity()]

        self.conv_part = nn.Sequential(*self.conv)

        if global_pool:
            self.global_pooling = Reduce("b c h w -> b c 1 1", "max")
            prev_ch = inner_ch[-1]
        else:
            """Define some flatten"""
            self.global_pooling = Rearrange("b c h w -> b (c h w)")
            prev_ch = self.compute_out_conv_dim()

        prev_d = prev_ch
        for i in range(len(depths_dense)):
            self.dense = self.dense + [
                torchlip.SpectralLinear(prev_d, depths_dense[i], bias=use_bias_dense),
                CappedSymmetricLeakyReLU(auto_scale=True),
                # torchlip.GroupSort2(),
                # nn.ReLU(),
                nn.Dropout(p=drop_dense)
            ]
            prev_d = depths_dense[i]

        self.dense.append(
            torchlip.SpectralLinear(prev_d, n_classes, bias=False)
        )

        self.network = torchlip.Sequential(
            *self.conv, self.global_pooling, *self.dense,
            k_coef_lip=lip
        )

    def forward(self, x):
        return self.network(x)
    

    def compute_out_conv_dim(self):
        x = torch.randn(1, self.in_ch, self.h, self.w)
        out = self.conv_part(x)

        return np.prod(out.shape[1:])


def get_intermediate_outputs_deel(deel_model, dataloader):
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

    for i, layer in enumerate(deel_model.network):
        if isinstance(layer, torchlip.LipschitzModule):
            layer.register_forward_hook(get_features(f"{layer.__class__.__name__}_{i}"))

    for batch in dataloader:
        x, y = batch
        out = deel_model(x.to("cuda"))

    for k in features.keys():
        features[k] /= len(dataloader)

    return features