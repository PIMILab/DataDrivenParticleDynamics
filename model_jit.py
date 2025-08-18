"""model_jit.py"""

import torch
from torch import Tensor
import torch.nn as nn
from scatter import scatter_add, scatter_mean
from torch_geometric.utils import softmax
import torch.nn.functional as F
from typing import Optional, List, Dict
import torch.func as func


# Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self, layer_vec):
        super().__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k+1])
            self.layers.append(layer)
            if k != len(layer_vec) - 2: self.layers.append(nn.SiLU())

    def forward(self, x: List[Tensor]):
        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]
        for layer in self.layers:
            x = layer(x)
        return x


# Equation of State Monotonic + Convex MLP
class MonotonicMLP(nn.Module):
    def __init__(self, layer_vec, t = [1, -1]):
        super().__init__()

        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k+1])
            self.layers.append(layer)

        self.register_buffer('t', torch.tensor(t))

    def forward(self, x: List[Tensor]):

        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]

        for i, layer in enumerate(self.layers):

            # Monotonicity: Modify weight matrix according to t
            if i == 0:
                weight = torch.abs(layer.weight) * self.t
            else:
                # Rest of weights should be positive
                weight = torch.abs(layer.weight)

            # Convexity: Convex activation function
            if i < len(self.layers) - 1: 
                x = F.softplus(F.linear(x, weight, layer.bias))
            else:
                # Last layer linear
                x = F.linear(x, weight, layer.bias)
        return x 


# Kernel Network
class KernelNet(nn.Module):
    def __init__(self, dim_hidden, n_hidden, h):
        super().__init__()
        
        self.MLP = MLP([1] + n_hidden * [dim_hidden] + [1])
        self.h = h


    def forward(self, r_ij):

        r = dot(r_ij, r_ij)**0.5
        S = r / self.h
        W = torch.exp(self.MLP([S])) * (1 - S**2)

        return W


def dot(a, b):
    '''
    Inner product of vectors a and b
    '''
    # out = (a * b).sum(-1, keepdim=True)
    out = torch.einsum('...i,...i->...', a, b)[..., None]
    return out


# Entropy model
class CG_model_S_jit(torch.nn.Module):
    def __init__(self, args, dims):
        super().__init__()

        self.D = dims        
        self.h = args.h
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.dt = args.dt
        self.boxsize = args.boxsize

        # Entropy teacher
        self.teacher = MLP([1 + self.D] + self.n_hidden * [self.dim_hidden] + [1])

    def forward(self, inputs: Dict[str, Tensor]):

        edge_index = inputs['edge_index']
        r_ij = inputs['r_ij']
        v = inputs['v']
        v_ij = v[edge_index[0]] - v[edge_index[1]]
        N = v.size(0)

        return self.model_S(r_ij, v_ij, edge_index, N)


    # Entropy encoder
    def model_S(self, r_ij, v_ij, edge_index, N:int):

        i = edge_index[0]
        j = edge_index[1]

        r_ = dot(r_ij, r_ij)**0.5

        S = scatter_mean(self.teacher([r_/self.h, v_ij]), i, dim=0, dim_size=N) + \
            scatter_mean(self.teacher([r_/self.h, -v_ij]), j, dim=0, dim_size=N)

        return S


# Model Kernel
class CG_model_W_jit(torch.nn.Module):
    def __init__(self, args, dims):
        super().__init__()

        self.D = dims        
        self.h = args.h
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.dt = args.dt
        self.boxsize = args.boxsize

        # Density
        self.model_W = KernelNet(self.dim_hidden, self.n_hidden, self.h)


    def forward(self, inputs: Dict[str, Tensor]):
        edge_index = inputs['edge_index']
        r_ij = inputs['r_ij']
        N = inputs['N']
        N = N[0]
        edges = r_ij.size(0)

        i = edge_index[0,:]
        j = edge_index[1,:]

        r_all = torch.cat([r_ij, -r_ij, torch.zeros((1, 3), device=r_ij.device)], dim=0)
        W_all = self.model_W(r_all)
        W_ij = W_all[0:edges]       # [N, 1]
        W_ji = W_all[edges:2*edges]  #optimize this later because W_ij = W_ji
        W_ii = W_all[-1]

        d = scatter_add(W_ij, i, dim=0, dim_size=N) + scatter_add(W_ji, j, dim=0, dim_size=N) + W_ii
        return d



# Machine Learning of Smoothed Dissipative Dynamics
class CG_model_jit(torch.nn.Module):
    def __init__(self, args, dims):
        super().__init__()

        self.D = dims        
        self.h = args.h
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.dt = args.dt
        self.boxsize = args.boxsize

        # Density
        self.model_W = KernelNet(self.dim_hidden, self.n_hidden, self.h)

        # Internal Energy
        self.model_U_vol = MonotonicMLP([2] + self.n_hidden * [self.dim_hidden] + [1])
        
        # Stochastic Coefficients
        self.model_A = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])
        self.model_B = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])
        self.model_C = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])

        # Extra constants
        self.log_k_B = nn.Parameter(torch.log(torch.tensor(args.k_B)))
        self.log_m = nn.Parameter(torch.log(torch.tensor(args.m)))


    def forward(self, inputs: Dict[str, Tensor]):

        v = inputs['v']
        edge_index = inputs['edge_index']
        r_ij = inputs['r_ij']
        S = inputs['S']
        d = inputs['d']
        dW = inputs['dW']
        dV_ij = inputs['dV']


        N = v.size(0)
        edges = edge_index.size(1)

        i = edge_index[0,:]
        j = edge_index[1,:]
        k_B = torch.exp(self.log_k_B)
        m = torch.exp(self.log_m)

        v_ij = v[i] - v[j]
        EPS = 1e-3
        
        # Auxiliary Variables
        r_ = dot(r_ij, r_ij)**0.5
        e_ij = r_ij / (r_ + 1e-8)

        # Density: Kernel Approximation    
        r_perturb = torch.cat([r_ + EPS, r_ - EPS], dim=0) 

        W_ij_perturb = self.model_W(r_perturb)
        W_plus = W_ij_perturb[0:edges]       
        W_minus = W_ij_perturb[edges:2*edges]    

        dW_dr = (W_plus - W_minus) / (2 * EPS)  
        grad_W_ij = dW_dr * e_ij
        grad_W_ji = -grad_W_ij


        # Entropy estimation
        V = 1/d

        EPS = 1e-2
        S_perturb = torch.cat([S,S+EPS,S,S-EPS], dim=0) 
        V_perturb = torch.cat([V,V,V+EPS,V], dim=0) 
        U_cat = self.model_U_vol([S_perturb, V_perturb])
        U = U_cat[0:N]
        U_Splus = U_cat[N:2*N]
        U_Vplus = U_cat[2*N:3*N]
        U_Sminus = U_cat[3*N:4*N]

        T = (U_Splus - U) / EPS
        P = -(U_Vplus - U) / EPS
        C = T * EPS**2 / (U_Splus - 2*U + U_Sminus)

        gradP_d_term1 = P[j] / d[j]**2 * grad_W_ji - P[i] / d[i]**2 * grad_W_ij

        T_i = T[i]
        T_j = T[j]
        EPS = 1e-3
        T_cat = torch.cat([T_i, T_j, T_i + EPS, T_j + EPS], dim=0)
        r_cat = (r_/self.h).repeat(4, 1)

        A_cat = self.model_A([r_cat, T_cat])
        B_cat = self.model_B([r_cat, T_cat])
        C_cat = self.model_C([r_cat, T_cat])
        
        A_i       = A_cat[0:edges]
        A_j       = A_cat[edges:2*edges]
        A_i_eps   = A_cat[2*edges:3*edges]
        A_j_eps   = A_cat[3*edges:4*edges]

        B_i       = B_cat[0:edges]
        B_j       = B_cat[edges:2*edges]
        B_i_eps   = B_cat[2*edges:3*edges]
        B_j_eps   = B_cat[3*edges:4*edges]

        C_i       = C_cat[0:edges]
        C_j       = C_cat[edges:2*edges]
        C_i_eps   = C_cat[2*edges:3*edges]
        C_j_eps   = C_cat[3*edges:4*edges]

        A_ij = A_i * A_j
        B_ij = B_i * B_j
        C_ij = C_i * C_j
        gradA_T_i = 2. * A_ij * (A_i_eps * A_j - A_ij) / EPS
        gradB_T_i = 2. * B_ij * (B_i_eps * B_j - B_ij) / EPS
        gradC_T_i = 2. * C_ij * (C_i_eps * C_j - C_ij) / EPS

        gradA_T_j = 2. * A_ij * (A_i * A_j_eps - A_ij) / EPS
        gradB_T_j = 2. * B_ij * (B_i * B_j_eps - B_ij) / EPS
        gradC_T_j = 2. * C_ij * (C_i * C_j_eps - C_ij) / EPS

        aux = A_ij**2/2  * v_ij + (A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D) * dot(e_ij, v_ij) * e_ij
        MgradS_v_term1 = -1/2 * (1./T_i + 1./T_j) * aux

        # Divergence terms
        divM_v_term1 = 1/2 * (1/C[i]/T_i + 1/C[j]/T_j) * aux 
        divM_v_term2 = -1/2 * (gradA_T_i/2 * v_ij + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * dot(e_ij, v_ij) * e_ij) / C[i] \
                       -1/2 * (gradA_T_j/2 * v_ij + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * dot(e_ij, v_ij) * e_ij) / C[j]

        aux = (A_ij**2/2 * dot(v_ij, v_ij) + (A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D) * dot(e_ij, v_ij)**2) / 4
        MgradS_S_term1 = (1/T[i] + 1/T[j]) * aux 
        MgradS_S_term2 = (1/T[i] - 1/T[j]) * C_ij**2
        
        divM_S_term1 = -(2/C[i]/T_i + 1/C[j]/T_j) * aux
        divM_S_term2 = (gradA_T_i/2 * dot(v_ij, v_ij) + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * dot(e_ij, v_ij)**2) / C[i] /4 \
                     + (gradA_T_j/2 * dot(v_ij, v_ij) + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * dot(e_ij, v_ij)**2) / C[j] /4              
        divM_S_term4 = -(2/C[i]/T_i - 1/C[j]/T_j) * C_ij**2
        divM_S_term5 = gradC_T_i / C[i] - gradC_T_j / C[j]   
        divM_S_term6 = -((self.D + 1) * A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D)

        # Stochastic Dynamics
        dW_ij_bar, tr_dW_ij = self.get_Wiener(dW, torch.tensor(edges))
        dv_tilde_term1 = torch.einsum('...ij,...j->...i', A_ij[...,None] * dW_ij_bar + B_ij[...,None] * tr_dW_ij / self.D, e_ij)

        dS_tilde_term1 = -1/2 * dot(dv_tilde_term1, v_ij)
        dS_tilde_term2 = C_ij * dV_ij

        # Scatter add
        cat_idx = torch.cat([i, j], dim=0)

        dvdt_i = gradP_d_term1 + MgradS_v_term1 + k_B * (divM_v_term1 + divM_v_term2)
        dvdt_j = - gradP_d_term1 - MgradS_v_term1 - k_B * (divM_v_term1 + divM_v_term2)
        dvdt_vals = torch.cat([dvdt_i, dvdt_j], dim=0)
        dvdt = 1/m * scatter_add(dvdt_vals, cat_idx, dim=0, dim_size=N)

        dSdt_i = MgradS_S_term1 + MgradS_S_term2 + k_B * (divM_S_term1 + divM_S_term2 + divM_S_term4 + divM_S_term5 + divM_S_term6 / m)
        dSdt_j = MgradS_S_term1 - MgradS_S_term2 + k_B * (divM_S_term1 + divM_S_term2 - divM_S_term4 - divM_S_term5 + divM_S_term6 / m)
        dSdt_vals = torch.cat([dSdt_i, dSdt_j], dim=0)
        dSdt = 1/T * scatter_add(dSdt_vals, cat_idx, dim=0, dim_size=N)
        
        dv_tilde_i = dv_tilde_term1
        dv_tilde_j = - dv_tilde_term1
        dv_tilde_vals = torch.cat([dv_tilde_i, dv_tilde_j], dim=0)
        dv_tilde = (2 * k_B)**0.5 / m * scatter_add(dv_tilde_vals, cat_idx, dim=0, dim_size=N)  # dW_ij = dW_ji
        dS_tilde_i = dS_tilde_term1 + dS_tilde_term2
        dS_tilde_j = dS_tilde_term1 - dS_tilde_term2
        dS_tilde_vals = torch.cat([dS_tilde_i, dS_tilde_j], dim=0)
        dS_tilde = (2 * k_B)**0.5 / T * scatter_add(dS_tilde_vals, cat_idx, dim=0, dim_size=N) # dV_ij = -dV_ji

        E = (U + 1/2 * m * dot(v,v))

        return {'dvdt': dvdt + dv_tilde / self.dt**0.5, 'dSdt': dSdt + dS_tilde / self.dt**0.5, 'E': E}

    
    def get_Wiener(self, dW_ij, N):

        device = dW_ij.device

        I = torch.eye(self.D, device=device).repeat(N, 1, 1)
        tr_dW_ij = I * torch.einsum('...ii', dW_ij)[...,None, None]
        dW_ij_bar = 1/2 * (dW_ij + dW_ij.transpose(1,2)) - tr_dW_ij / self.D
    
        return dW_ij_bar, tr_dW_ij




if __name__ == '__main__':
    pass
