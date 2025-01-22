import torch
from torch import nn
from typing import List
from tqdm import tqdm
import einops

from layers import ABBA_Dense_Layer, ABBA_Conv_Layer
from utils.conv_utils import spatial_lip, freq_lip


class ABBAConvConstraint(nn.Module):
    """
    Module for constraining an ABBA Conv Layer -- defines all necessary variables ONCE,
    and reuses the previous values to reduce the constraint time.
    """

    def __init__(self, abba_conv_layer: ABBA_Conv_Layer, device,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = abba_conv_layer
        self.device = device

        if self.layer.s > 1:
            raise NotImplementedError("Constraint not yet implemented for stride > 1.")

        self.W_ABBA = torch.empty(self.layer.out_ch * 2, self.layer.in_ch * 2, self.layer.k, self.layer.k).to(device)

        # Define variables used during constraint
        self.X = torch.zeros_like(self.W_ABBA).to(device)
        self.proj_resul = torch.empty_like(self.W_ABBA).to(device)
        self.r = min(self.W_ABBA.shape[0], self.W_ABBA.shape[1])
        self.Su = torch.zeros(self.W_ABBA.shape[0], self.W_ABBA.shape[1])
        self.U = torch.zeros(self.W_ABBA.shape[0], self.r).to(device)
        self.S = torch.zeros(self.r).to(device)
        self.V = torch.zeros(self.r, self.W_ABBA.shape[1]).to(device)
        self.current_lip = None

        self.init_iter_variables()
        self.construt_abba_conv()

    def init_iter_variables(self):
        self.Z = torch.zeros_like(self.W_ABBA).to(self.device)
        self.Y_old = torch.zeros_like(self.W_ABBA).to(self.device)
        self.Y = torch.zeros_like(self.W_ABBA).to(self.device)
        self.Yt = torch.zeros_like(self.W_ABBA).to(self.device)

    def projLipSpatial(self, lip, gamma):
        self.Su = torch.sum(self.Y / gamma, dim=(-1, -2))
        self.U, self.S, self.V = torch.linalg.svd(self.Su, full_matrices=False)
        self.S = torch.clamp(self.S, 0, lip)
        self.proj_resul = self.Y / gamma + (self.U @ torch.diag(self.S) @ self.V - self.Su).\
                          unsqueeze(-1).unsqueeze(-1) / self.Y.shape[-2] / self.Y.shape[-1]

    def projDFBconv(self, lip, n_it, gamma=0.1, alpha=2.1, cnst=1e-2):
        for i in range(1, n_it + 1):
            self.X = torch.clamp(self.W_ABBA - self.Z, 0)

            if abs(spatial_lip(self.X) - lip) <= cnst:
                break

            self.Yt = self.Z + gamma * self.X
            self.Y_old = self.Y
            self.projLipSpatial(lip, gamma)
            self.Y = self.Yt - gamma * self.proj_resul
            self.Z = self.Y + i / (i + 1 + alpha) * (self.Y_old - self.Y)

        # self.X = torch.clamp(self.W_ABBA - self.Z, 0)

    def construt_abba_conv(self):
        """
        :param x: ABBA_Conv_Layer
        :return: full, equivalent convolution kernel
        """
        A, B = self.layer.A.data, self.layer.B.data

        assert A.min() >= 0 and B.min() >= 0

        self.W_ABBA[:self.layer.out_ch, :self.layer.in_ch] = A
        self.W_ABBA[:self.layer.out_ch, self.layer.in_ch:] = B
        self.W_ABBA[self.layer.out_ch:, :self.layer.in_ch] = B
        self.W_ABBA[self.layer.out_ch:, self.layer.in_ch:] = A

        self.current_lip = spatial_lip(self.W_ABBA, stride=self.layer.s)

    def run(self, lip=1, n_it=10):
        assert n_it >= 0
        if self.current_lip >= 1e-1 + lip:
            self.init_iter_variables()
            self.projDFBconv(lip, n_it)
            # result is in self.X
            with torch.no_grad():
                # X now is the full filter bank of combined A/B filters
                self.layer.A.copy_(self.X[:self.layer.out_ch, :self.layer.in_ch])
                self.layer.B.copy_(self.X[:self.layer.out_ch, self.layer.in_ch:])


def constrain_abba_conv(abba_conv_constraints, lips, n_it=100):
    """
    Constrains a series of ABBAConvConstraint, individually, with the imposed Lipschitz constants,
    using the Spatial DFB algorithm. The constraint objects are already linked to their corresponding layer.
    :param abba_conv_constraints: list of ABBAConvConstraint objects
    :param lips: list of floats, same length as abba_layers; Lipschitz constants to constrain ABBA_Conv_Layers to
    :return: nothing, this functions changes the values of each object from abba_layers, passed as a reference
    """
    for i, constraint in enumerate(abba_conv_constraints):
        constraint.run(lips[i], n_it)


class ABBADenseConstraint(nn.Module):
    """
    Module for constraining a sequence of consecutive ABBA Dense layers, or a single ABBA Dense layer.
    """
    def __init__(self, abba_dense_layers: List[ABBA_Dense_Layer], device, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense_layers = abba_dense_layers
        self.device = device

        self.W_ABBAs = [
            torch.empty(x.in_units * 2, x.units * 2).to(device) for x in self.dense_layers
        ]
        self.W_ABBAs_ = [
            torch.empty(x.in_units * 2, x.units * 2).to(device) for x in self.dense_layers
        ]

        # These are shared amongst all layers -- can help for faster convergence
        self.Y = torch.zeros(self.W_ABBAs[0].shape[0], self.W_ABBAs[-1].shape[1]).to(device)
        self.Yt = torch.zeros(self.W_ABBAs[0].shape[0], self.W_ABBAs[-1].shape[1]).to(device)
        self.T = torch.zeros(self.W_ABBAs[0].shape[0], self.W_ABBAs[-1].shape[1]).to(device)

        r = min(self.W_ABBAs[0].shape[0], self.W_ABBAs[-1].shape[1])
        self.U = torch.zeros(self.W_ABBAs[0].shape[0], r).to(device)
        self.S = torch.zeros(r).to(device)
        self.V = torch.zeros(r, self.W_ABBAs[-1].shape[-1]).to(device)

        # not to be confused with A and B from an ABBA layer!!!
        # these A_B points to tensors representing previous and subsequent composition of operators
        # A -- (i+1)->m layers; B -- 1->i-1 layers; i -- layer index
        self.B_A = []
        for i, x in enumerate(self.W_ABBAs):
            if i == 0:
                B = torch.eye(x.shape[0]).to(device)
                A = torch.empty(x.shape[1], self.W_ABBAs[-1].shape[1]).to(device)
            elif i == len(self.W_ABBAs) - 1:
                B = torch.empty(self.W_ABBAs[0].shape[0], x.shape[0]).to(device)
                A = torch.eye(x.shape[1]).to(device)
            else:
                B = torch.empty(self.W_ABBAs[0].shape[0], x.shape[0]).to(device)
                A = torch.empty(x.shape[1], self.W_ABBAs[-1].shape[1]).to(device)
            self.B_A.append([B, A])

        self.current_lips = torch.empty(len(self.dense_layers)).to(device)
        self.current_global = None

        self.construct_abba_denses()
        self.construct_before_after_operators(0)

    def iterative_dfb(self, i, lip, n_it, cnst=1e-2):
        """
        Performs iterative DFB algorithm to constrain an ABBA Dense layer, given it's index
        from the full sequence.
        :param i: Index of the layer to be constrained
        :param lip: Imposed norm
        :param n_it:
        :param cnst:
        """
        gamma = 2.0 / (
            (torch.linalg.matrix_norm(self.B_A[i][0], ord=2) * torch.linalg.matrix_norm(self.B_A[i][1], ord=2))
            + 1e-7
        )**2
        for _ in range(n_it):
            self.W_ABBAs_[i] = self.W_ABBAs[i] - self.multiply_matrices([self.B_A[i][0].transpose(0, 1),
                                                                         self.Y,
                                                                         self.B_A[i][1].transpose(0, 1)])
            self.W_ABBAs_[i] = torch.clamp(self.W_ABBAs_[i], 1e-9)
            self.T = self.multiply_matrices([self.B_A[i][0], self.W_ABBAs_[i], self.B_A[i][1]])

            self.current_global = torch.linalg.matrix_norm(self.T, ord=2)
            if torch.abs(self.current_global - lip) < cnst:
                break

            self.Yt = self.Y + gamma * self.T
            self.U, self.S, self.V = torch.linalg.svd(self.Yt / gamma, full_matrices=False)
            self.S = torch.clamp(self.S, 0, lip)
            self.Y = self.Yt - gamma * self.multiply_matrices([self.U, torch.diag(self.S), self.V])

    def iterate_layers(self, lip, n_it, cnst=1e-2):
        if len(self.dense_layers) == 1:
            self.current_global = self.current_lips[0]
            if abs(self.current_global - lip) < cnst or self.current_global < lip:
                return
            else:
                # Just do SVD clipping for the only layer
                self.U, self.S, self.V = torch.linalg.svd(self.W_ABBAs[0], full_matrices=False)
                self.S = torch.clamp(self.S, 0, lip)
                self.W_ABBAs_[0] = self.multiply_matrices([self.U, torch.diag(self.S), self.V])

        elif len(self.dense_layers) > 1:
            """Constrain layers with the highest Lip first"""
            lip_descending_idx = torch.argsort(self.current_lips, descending=True)
            # lip_descending_idx = list(range(len(self.current_lips)))
            for i in lip_descending_idx:
                self.construct_before_after_operators(i)
                if abs(self.current_global - lip) < cnst or self.current_global < lip:
                    return
                else:
                    self.iterative_dfb(i, lip, n_it, cnst)
                    self.W_ABBAs[i] = self.W_ABBAs_[i]

    def multiply_matrices(self, m_list):
        result = m_list[0]
        for i in range(1, len(m_list)):
            result = result @ m_list[i]
        return result

    def construct_abba_denses(self):
        for i, x in enumerate(self.dense_layers):
            A, B = x.A.data, x.B.data

            assert A.min() >= 0 and B.min() >= 0

            in_u, out_u = x.in_units, x.units

            self.W_ABBAs[i][:in_u, :out_u] = A
            self.W_ABBAs[i][:in_u, out_u:] = B
            self.W_ABBAs[i][in_u:, :out_u] = B
            self.W_ABBAs[i][in_u:, out_u:] = A

            self.W_ABBAs_[i][:in_u, :out_u] = A
            self.W_ABBAs_[i][:in_u, out_u:] = B
            self.W_ABBAs_[i][in_u:, :out_u] = B
            self.W_ABBAs_[i][in_u:, out_u:] = A

            self.current_lips[i] = torch.linalg.matrix_norm(A + B, ord=2)

    def construct_before_after_operators(self, i):
        with torch.no_grad():
            if len(self.dense_layers) == 1:
                self.current_global = self.current_lips[0]
                
            elif len(self.dense_layers) > 1:
                if i == 0:
                    self.B_A[i][1] = self.multiply_matrices(self.W_ABBAs[1:])
                    self.T = self.multiply_matrices([self.W_ABBAs[0], self.B_A[i][1]])
                elif i == len(self.W_ABBAs) - 1:
                    self.B_A[i][0] = self.multiply_matrices(self.W_ABBAs[:-1])
                    self.T = self.multiply_matrices([self.B_A[i][0], self.W_ABBAs[-1]])
                else:
                    self.B_A[i][0] = self.multiply_matrices(self.W_ABBAs[:i])
                    self.B_A[i][1] = self.multiply_matrices(self.W_ABBAs[(i+1):])
                    self.T = self.multiply_matrices([self.B_A[i][0], self.W_ABBAs[i], self.B_A[i][1]])
                self.current_global = torch.linalg.matrix_norm(self.T, ord=2)

    def run(self, lip=1, n_it=10):
        assert n_it >= 1
        self.iterate_layers(lip, n_it)
        with torch.no_grad():
            for i, x in enumerate(self.dense_layers):
                x.A.copy_(self.W_ABBAs_[i][:x.in_units, :x.units])
                x.B.copy_(self.W_ABBAs_[i][:x.in_units, x.units:])


class StandardDenseConstraint(nn.Module):
    """
    Module for constraining separately standard Linear layers.
    """
    def __init__(self, dense_layers: List[nn.Linear], device, typ="divide", method="exact", *args, **kwargs):
        """
        Standard module for constraining a sequence of nn.Linear layers to desired Lip.

        :param typ: string, how to constrain:
            "divide" - divides each operator to it's norm and multiplies by a factor 
            "clip" - performs SVD and clips the S matrix, then projects it back 
        :param method: string, how to estimate L_2:
            "exact" - use an already implemented functionality for exact calculation
            "power" - use power iteration to estimate the dominant singular vector, and thus the L_2
                    - only applicable to typ="divide"
        """
        super().__init__(*args, **kwargs)

        self.dense_layers = dense_layers
        self.device = device
        self.typ = typ
        self.method = method

        assert self.typ in ["divide", "clip"], ValueError(f"Unknown value for typ: {self.typ}.")
        assert self.method in ["exact", "power"], ValueError(f"Unknown value for method: {self.method}")

        self.Ws_ = [torch.empty(x.out_features, x.in_features).to(device) for x in self.dense_layers]

        if self.method == "power":
            ### Construct variables to be updated as dominant singular vectors
            ### Next update will start from the previous estimate
            self.v = [(torch.ones(x.in_features) / x.in_features ** 0.5).to(device) for x in self.dense_layers]

        ### Define space SVD operator for each layer
        r = [min(self.Ws_[i].shape[0], self.Ws_[i].shape[1]) for i in range(len(self.Ws_))]
        self.U = [torch.zeros(self.Ws_[i].shape[0], r[i]).to(device) for i in range(len(self.Ws_))]
        self.S = [torch.zeros(r[i]).to(device) for i in range(len(self.Ws_))]
        self.V = [torch.zeros(r[i], self.Ws_[i].shape[1]).to(device) for i in range(len(self.Ws_))]

        # Updated only for non-exact power method
        self.lips = torch.ones(len(self.dense_layers)).to(device)

    def multiply_matrices(self, m_list):
        result = m_list[0]
        for i in range(1, len(m_list)):
            result = result @ m_list[i]
        return result

    def iterate_layers(self, lips):
        if self.typ == "clip":
            for i in range(len(self.dense_layers)):           
                self.U[i], self.S[i], self.V[i] = torch.linalg.svd(self.dense_layers[i].weight.data, 
                                                            full_matrices=False)
                # self.S[i] = torch.clamp(self.S[i], 0, lips[i])
                self.S[i] = torch.clamp(self.S[i], lips[i], lips[i]) # make all singular values same
                self.Ws_[i] = self.multiply_matrices([self.U[i], torch.diag(self.S[i]), self.V[i]])

                self.lips[i] = lips[i]
        elif self.typ == "divide":
            if self.method == "exact":
                for i in range(len(self.dense_layers)):
                    self.Ws_[i] = lips[i] * self.dense_layers[i].weight.data / torch.linalg.norm(self.dense_layers[i].weight.data, ord=2)
            
                    self.lips[i] = lips[i]
            elif self.method == "power":
                for i in range(len(self.dense_layers)):
                    self.lips[i] = power_iteration(self.dense_layers[i].weight.data, self.v[i])
                    self.Ws_[i] = lips[i] * self.dense_layers[i].weight.data / self.lips[i]

                    self.lips[i] = lips[i]
    
    def compute_lips(self):
        for i in range(len(self.dense_layers)):  
            self.lips[i] = torch.linalg.matrix_norm(self.dense_layers[i].weight.data, ord=2)

    def run(self, lips=1.0):
        if isinstance(lips, (int, float)):
            lips = [lips] * len(self.dense_layers)

        self.iterate_layers(lips)

        with torch.no_grad():
            for i, x in enumerate(self.dense_layers):
                x.weight.copy_(self.Ws_[i])


class StandardConvConstraint(nn.Module):
    """
    Module for constraining separately standard Conv2d layers.
    Uses the frequency constraint for arbitrary-signed operators.

    In torch, the Conv2d weight operator is of shape (ch_out, ch_in, k, k).
    """
    def __init__(self, conv_layers: List[nn.Conv2d], device, nf=8, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.conv_layers = conv_layers
        self.device = device
        self.nf = nf

        self.Ws_ = [
            torch.empty(x.out_channels, x.in_channels, *x.kernel_size).to(device) 
            for x in self.conv_layers
        ]

        self.W = [
            torch.empty(x.out_channels, x.in_channels, nf, nf).to(device) 
            for x in self.conv_layers        
        ]

        # Updated only for non-exact power method
        self.lips = torch.ones(len(self.conv_layers)).to(device)

    def iterate_layers(self, lips):
        for i in range(len(self.conv_layers)):
            lip_i = freq_lip_standard(self.W[i], self.conv_layers[i].weight.data)
            self.Ws_[i] = lips[i] * self.conv_layers[i].weight.data / lip_i
            self.lips[i] = lips[i]

    def compute_lips(self):
        for i in range(len(self.conv_layers)):
            self.lips[i] = freq_lip_standard(self.W[i], self.conv_layers[i].weight.data)

    def run(self, lips=1.0):
        if isinstance(lips, (int, float)):
            lips = [lips] * len(self.conv_layers)
        
        self.iterate_layers(lips)

        with torch.no_grad():
            for i, x in enumerate(self.conv_layers):
                x.weight.copy_(self.Ws_[i])


def power_iteration(W, v, n_iters=5):
    """
    Returns the approximate $L_2$ norm of matrix W, resulted from power iteration.

    :param W: torch.Tensor, weight matrix for which L_2 is estimated - (n_out, n_in)
    :param v: torch.Tensor, variable iteratively updated to converge to the dominant right singular vector - (n_in)
    """

    for _ in range(n_iters):
        v = torch.mv(W.t(), torch.mv(W, v))
    # don't need to normalize at each iter - only if we want v to estimate the 
    # dominant right singular vector of W
    v = v / torch.norm(v)
    
    return torch.norm(torch.mv(W, v))


def freq_lip_standard(W, w, stride=1):
    """
    Returns the $L_2$ norm of an arbitrary-signed Conv2d layer.

    :param W: placeholder for storing the 2D FFT - (ch_out, ch_in, nf, nf), where nf=number of FFT components
    :param w: torch.Tensor containing the Conv2d weights - (ch_out, ch_in, k, k)
    :param nf: int, number of FFT components to compute 
    """

    W = torch.fft.fft2(w, dim=(-2, -1), s=(W.shape[-2], W.shape[-1]))
    if stride == 1:
        return torch.linalg.matrix_norm(W, dim=(0, 1), ord=2).max()
    else:
        raise NotImplementedError("Frequency method not yet implemented fro stride > 1.")