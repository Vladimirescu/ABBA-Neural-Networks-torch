from torch import nn
import torch
import einops
import numpy as np
from scipy.signal import convolve2d

from layers import ABBA_Dense_Layer, ABBA_Conv_Layer, MyBatchNorm2D, MyBatchNorm1D
from models import ConvDenseABBA, FullyConvABBA, FullyDenseABBA


def construct_abba_dense(x: ABBA_Dense_Layer):
    """
    :param x: ABBA_Dense_Layer
    :return: full matrix [A B;B A]
    """

    in_u, out_u = x.in_units, x.units
    A, B = x.A.data, x.B.data

    W = torch.empty(in_u * 2, out_u * 2)

    W[:in_u, :out_u], W[:in_u, out_u:] = A, B
    W[in_u:, :out_u], W[in_u:, out_u:] = B, A

    return W


def deconstruct_abba_dense(W_ABBA):
    """
    :param W_ABBA: tensor matrix of ABBA type
    :returns: A, B
    """

    assert len(W_ABBA.shape) == 2

    r, c = W_ABBA.shape

    assert r % 2 == 0 and c % 2 == 0

    return W_ABBA[:r//2, :c//2], W_ABBA[:r//2, c//2:]


def construt_abba_conv(x: ABBA_Conv_Layer, simplified=False):
    """
    :param x: ABBA_Conv_Layer
    :return: full, equivalent convolution kernel
    """

    with torch.no_grad():
        in_ch, out_ch, k = x.in_ch, x.out_ch, x.k
        A, B = x.A.data, x.B.data

        if simplified:
            W = A - B
        else:
            W = torch.empty(2 * out_ch, 2 * in_ch, k, k)

            W[:out_ch, :in_ch], W[:out_ch, in_ch:] = A, B
            W[out_ch:, :in_ch], W[out_ch:, in_ch:] = B, A

    return W


def freq_lip(w, nf=8, stride=1):
    """
    :param w: ABBA MIMO Conv 2D filter bank, of shape (ch_out, ch_in, k, k)
    :param nf:
    :param stride:
    :return: Lip const computed using the freqency method
    """

    W = torch.fft.fft2(w, dim=(-2, -1), s=(nf, nf))
    if stride == 1:
        return torch.linalg.matrix_norm(W, dim=(0, 1), ord=2).max()
    else:
        raise NotImplementedError("Frequency method not yet implemented fro stride > 1.")


def spatial_lip(w, stride=1):
    """
    :param w: ABBA MIMO Conv 2D filter bank
    :param stride:
    :return: Lip const computed using the spatial method
    """

    if stride == 1:
        W = torch.sum(w, dim=(-2, -1))
        W_ = torch.matmul(W, W.transpose(0, 1))
        return torch.sqrt(torch.linalg.matrix_norm(W_, ord=2))
    else:
        # raise NotImplementedError("Frequency method not yet implemented fro stride > 1.")
        W_ = torch.zeros((w.shape[0], w.shape[0])).to(w.device)
        for j1 in range(stride):
            for j2 in range(stride):
                W_ += torch.matmul(
                    w[:, :, j1::stride, j2::stride].sum(dim=(-2, -1)), 
                    w[:, :, j1::stride, j2::stride].sum(dim=(-2, -1)).transpose(0, 1)
                )

        return torch.sqrt(torch.linalg.matrix_norm(W_, ord=2))


def compute_lip_conv(x, freq=True, simplified=False):
    """
    :param x: an ABBA_Conv_Layer or nn.Conv2d layer
    :param freq: bool, wehether to use the Frequency, or Spatial method for computing
    :return: it's Lipschitz constant
    """

    if isinstance(x, ABBA_Conv_Layer):
        W = construt_abba_conv(x)
        if freq:
            if simplified:
                return freq_lip(x.A.data - x.B.data)
            else:
                return freq_lip(W)
        else:
            # raise NotImplementedError("Spatial method not yet implemented")
            return spatial_lip(W)
    else:
        return freq_lip(x.weight)


def convMIMO(h1, h2):
    """
    :params h1, h2: two Conv2d filter banks of shapes (out, in, k, k) (different)
    """
    n_out1, n_in1, k1, k1 = h1.shape
    n_out2, n_in2, k2, k2 = h2.shape

    assert n_in1 == n_out2

    y = torch.zeros(n_out1, n_in2, k1 + k2 - 1, k1 + k2 - 1)
    for i in range(n_out1):
        for j in range(n_in2):
            for k in range(n_in1):
                c = convolve2d(h1[i, k], h2[k, j])
                y[i, j] += c

    return y


def compute_global_conv(abba_conv_layers, simplified=False):
    if len(abba_conv_layers) == 1:
        return construt_abba_conv(abba_conv_layers[0], simplified=simplified)
    else:
        H = construt_abba_conv(abba_conv_layers[0], simplified=simplified)
        SS = abba_conv_layers[0].s
        for i in range(1, len(abba_conv_layers)):
            out_ch, in_ch, k = abba_conv_layers[i].out_ch, abba_conv_layers[i].in_ch, abba_conv_layers[i].k
            if simplified:
                hs = torch.zeros(out_ch, in_ch, SS * (k - 1) + 1, SS * (k - 1) + 1)
            else:
                hs = torch.zeros(2 * out_ch, 2 * in_ch, SS * (k - 1) + 1, SS * (k - 1) + 1)
            hs[..., 0::SS, 0::SS] = construt_abba_conv(abba_conv_layers[i], simplified=simplified)
            H = convMIMO(hs, H)

            SS *= abba_conv_layers[0].s

        return H


def get_lip_abba_dense_seq(abba_seq, simplified=False):
    """
    :param abba_seq: list of ABBA_Dense_Layer objects
    :param simplified: bool, this will return an approx Lip if the simplified ABBA architecture was used
    :return: list of per-layer lipschitz constants, their product,
     and, if applicable, global Lipschitz constant
    """

    assert all(isinstance(x, ABBA_Dense_Layer) for x in abba_seq), ValueError("Not all ABBA.")

    per_layer_lips = torch.ones(len(abba_seq))
    inner_prod = None

    if len(abba_seq) > 1:
        for i, x in enumerate(abba_seq):
            w = x.A + x.B if not simplified else x.A - x.B
            if not simplified:
                assert w.min() >= 0
            per_layer_lips[i] = torch.linalg.matrix_norm(w, ord=2)

            inner_prod = construct_abba_dense(x) if inner_prod is None else \
                torch.matmul(inner_prod, construct_abba_dense(x))

        if not simplified:
            s_global = torch.linalg.matrix_norm(inner_prod, ord=2)
        else:
            A_global, B_global = deconstruct_abba_dense(inner_prod)
            s_global = torch.linalg.matrix_norm(A_global - B_global, ord=2)

        return per_layer_lips, torch.prod(per_layer_lips), s_global
    else:
        w = abba_seq[0].A + abba_seq[0].B if not simplified else abba_seq[0].A - abba_seq[0].B
        if not simplified:
            assert w.min() >= 0
        per_layer_lips[0] = torch.linalg.matrix_norm(w, ord=2)

        return per_layer_lips, torch.prod(per_layer_lips), torch.prod(per_layer_lips)


def get_lip_standard_dense(dense_seq):
    """
    :param dense_seq: list of Linear objects
    :return: list of per-layer lipschitz constants
    """

    assert isinstance(dense_seq, list), ValueError("dense_seq should be list.")
    assert all(isinstance(x, nn.Linear) for x in dense_seq), ValueError("Not all nn.Linear.")

    per_layer_lips = torch.ones(len(dense_seq))
    for i, x in enumerate(dense_seq):
        per_layer_lips[i] = torch.linalg.svdvals(x.weight.data)[0]

    return per_layer_lips


def get_lip_abba_conv_seq(abba_seq, freq=True, simplified=False, compute_global=False):
    """
    :param abba_seq: list of ABBA_Conv_Layers objects, in the order given by the network
    :param freq: bool, whether the Lip for ABBA is computed by freq method, or spatial one
    :param compute_global: bool, whether the global bound is computed for ABBA conv
    :return: list of per-layer Lipschitz constants, their product,
    and, if applicable, global Lipschitz constant
    """

    assert all(isinstance(x, ABBA_Conv_Layer) for x in abba_seq), ValueError("Not all ABBA.")

    per_layer_lips = torch.ones(len(abba_seq))
    inner_mimo = None

    if len(abba_seq) >= 1:
        for i, x in enumerate(abba_seq):
            per_layer_lips[i] = compute_lip_conv(x, freq=freq, simplified=simplified)

        if compute_global:
            conv_global = compute_global_conv(abba_seq, simplified=simplified)
            if simplified:
                lip_global = freq_lip(conv_global)
            else:
                lip_global = spatial_lip(conv_global)
            return per_layer_lips, torch.prod(per_layer_lips), lip_global
        else:
            return per_layer_lips, torch.prod(per_layer_lips)
    else:
        if compute_global:
            return [], 1, 1
        else:
            return [], 1


def get_lip_standard_conv(conv_seq):
    """
    """
    assert isinstance(conv_seq, list), ValueError("dense_seq should be list.")
    assert all(isinstance(x, nn.Conv2d) for x in conv_seq), ValueError("Not all nn.Conv.")

    per_layer_lips = torch.ones(len(conv_seq))
    for i, x in enumerate(conv_seq):
        per_layer_lips[i] = compute_lip_conv(x)

    return per_layer_lips


def get_bn_factors(bn_layers, return_variances=False):
    
    if len(bn_layers) > 0:
        lip_factor_bn = 1.
        variances = []

        for layer in bn_layers:
            lip_factor_bn = lip_factor_bn * \
                torch.max(layer.gamma.squeeze() / (layer.running_var.squeeze() + 1e-3))
            variances.append(layer.running_var.squeeze())

        if return_variances:
            return lip_factor_bn.item(), variances
        else:
            return lip_factor_bn.item()
    else:
        if return_variances:
            return 1, []
        else:
            return 1


def get_lips(model, batch_norm_layers, simplified=False, return_lips=False, compute_global_conv=False):
    """
    Return all possible Lips for model.
    """
    standard_conv_layers = []
    standard_dense_layers = []
    abba_conv_layers = []
    abba_dense_layers = []
    if isinstance(model.module, ConvDenseABBA):
        standard_conv_layers = model.conv_net.standard_layers
        abba_conv_layers = model.conv_net.abba_layers
        
        standard_dense_layers = model.dense_net.standard_layers
        abba_dense_layers = model.dense_net.abba_layers
    elif isinstance(model.module, FullyConvABBA):
        standard_conv_layers = model.standard_layers
        abba_conv_layers = model.abba_layers
    elif isinstance(model.module, FullyDenseABBA):
        standard_dense_layers = model.standard_layers
        abba_dense_layers = model.abba_layers
    else:
        raise NotImplementedError(f"Unknown model type {type(model)}")

    with torch.no_grad():
        standard_conv_lips = get_lip_standard_conv(standard_conv_layers)
        if compute_global_conv:
            abba_conv_lips, abba_conv_prod, abba_conv_global = get_lip_abba_conv_seq(abba_conv_layers, simplified=simplified, compute_global=True)
        else:
            abba_conv_lips, abba_conv_prod = get_lip_abba_conv_seq(abba_conv_layers, simplified=simplified)
            abba_conv_global = None

        standard_dense_lips = get_lip_standard_dense(standard_dense_layers)
        abba_dense_lips, abba_dense_prod, abba_dense_global = get_lip_abba_dense_seq(abba_dense_layers,
                                                                                    simplified=simplified)
        bn_lip = get_bn_factors(batch_norm_layers)

        total_prod_no_bn = abba_dense_global * torch.prod(torch.Tensor(abba_conv_lips)) * torch.prod(standard_dense_lips) * torch.prod(standard_conv_lips)
        total_prod = total_prod_no_bn * bn_lip

        print("===============================")
        print(f"Simplified mode = {simplified}")

        print("Standard Dense Lips: ", standard_dense_lips)
        print("ABBA Dense Lips: ", abba_dense_lips)
        print(f"ABBA Dense Prod / Global: {abba_dense_prod} / {abba_dense_global}")
        print("Standard Conv Lips: ", standard_conv_lips)
        print("ABBA Conv Lips: ", abba_conv_lips)
        print(f"ABBA Conv Prod / Global: {abba_conv_prod} / {abba_conv_global}")
        print("BN Lip: ", bn_lip)
        print("Total prod: ", total_prod)
        print("Total prod without BN: ", total_prod_no_bn)
        print("===============================")

        if return_lips:
            return standard_dense_lips, abba_dense_lips, abba_dense_prod, abba_dense_global, \
                    standard_conv_lips, abba_conv_lips, abba_conv_global, bn_lip, total_prod, total_prod_no_bn


def get_lips_standard(model):

    dense_norms = []
    conv_norms = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            dense_norms.append(
                torch.linalg.matrix_norm(layer.weight.data.cpu(), ord=2).cpu()
                )
        if isinstance(layer, nn.Conv2d):
            conv_norms.append(
                freq_lip(layer.weight.data.cpu())
                )

    print("Denses: ", dense_norms, " Prod: ", np.prod(dense_norms))
    print("Convs: ", conv_norms, " Prod: ", np.prod(conv_norms))
    print("Total: ", np.prod(dense_norms) * np.prod(conv_norms))


def get_lips_standard_as_abba(conv_layers, dense_layers, simplified=False):
    """
    Creates ABBA layers starting from standard ones, and measures the proposed bounds.
    """

    abba_conv_layers = []
    abba_dense_layers = []

    for i, layer in enumerate(conv_layers):
        if isinstance(layer, nn.Conv2d):
            abba_conv_layers.append(
                ABBA_Conv_Layer(layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.stride[0], layer.padding)
            )
            abba_conv_layers[-1].A.data.copy_(
                (torch.abs(layer.weight.data) + layer.weight.data) / 2.0
            )
            abba_conv_layers[-1].B.data.copy_(
                (torch.abs(layer.weight.data) - layer.weight.data) / 2.0
            )

    for i, layer in enumerate(dense_layers):
        if isinstance(layer, nn.Linear):
            abba_dense_layers.append(
                ABBA_Dense_Layer(layer.in_features, layer.out_features)
            )
            abba_dense_layers[-1].A.data.copy_(
                ((torch.abs(layer.weight.data) + layer.weight.data) / 2.0).transpose(0, 1)
            )
            abba_dense_layers[-1].B.data.copy_(
                ((torch.abs(layer.weight.data) - layer.weight.data) / 2.0).transpose(0, 1)
            )

    abba_conv_lips, abba_conv_prod, abba_conv_global = get_lip_abba_conv_seq(abba_conv_layers, simplified=simplified, compute_global=True)
    abba_dense_lips, abba_dense_prod, abba_dense_global = get_lip_abba_dense_seq(abba_dense_layers, simplified=simplified)

    print("ABBA Dense Lips: ", abba_dense_lips)
    print(f"ABBA Dense Prod / Global: {abba_dense_prod} / {abba_dense_global}")
    print("ABBA Conv Lips: ", abba_conv_lips)
    print(f"ABBA Conv Prod / Global: {abba_conv_prod} / {abba_conv_global}")
    print(f"ABBA Global (separable): ", abba_conv_prod * abba_dense_prod)
    print(f"ABBA Global (2-separable): ", abba_conv_global * abba_dense_global)