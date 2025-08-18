"""model.py"""

import torch
import torch.nn as nn
from scatter import scatter_add, scatter_mean
from torch_geometric.utils import softmax
import torch.nn.functional as F
from utils import dot, outer, grad, eye_like


# Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self, layer_vec, layer_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k+1])
            self.layers.append(layer)

            if k != len(layer_vec) - 2: 
                self.layers.append(nn.SiLU())
                if layer_norm:
                    self.layers.append(nn.LayerNorm(layer_vec[k+1]))

    def forward(self, *x):
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

    def forward(self, *x):
        
        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]

        for i, layer in enumerate(self.layers):

            # Monotonicity: Modify weight matrix according to t
            if i == 0:
                t_mask = self.t.view(1, -1)
                weight = (
                    torch.abs(layer.weight) * t_mask +  # Monotonic part
                    layer.weight * (t_mask == 0)        # Free part
                )
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
class KernelMLP(nn.Module):
    def __init__(self, dim_hidden, n_hidden, h):
        super().__init__()
        
        self.MLP = MLP([1] + n_hidden * [dim_hidden] + [1])
        self.h = h

    def forward(self, r_ij):

        r = dot(r_ij, r_ij)**0.5
        S = r / self.h
        W = torch.exp(self.MLP(S)) * (1 - S**2)
        
        return W
    

# Strain Network
class StrainMLP(nn.Module):
    def __init__(self, dim_hidden, n_hidden, D, h):
        super().__init__()
        self.D = D
        self.h = h
        self.MLP = MLP([2*self.D] + n_hidden * [dim_hidden] + [self.D*self.D])

    def forward(self, *x):

        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]
        x = x/self.h

        l_flat = self.MLP(x)
        l = l_flat.view(-1, self.D, self.D)

        # Symmetrize
        epsilon = 1/2 * (l + l.transpose(1,2))
        
        return epsilon
        

# Machine Learning of Smoothed Dissipative Dynamics
class CG_model(torch.nn.Module):
    def __init__(self, args, dims):
        super().__init__()

        self.D = dims        
        self.h = args.h
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.dt = args.dt
        self.boxsize = args.boxsize

        # Density
        self.model_W = KernelMLP(self.dim_hidden, self.n_hidden, self.h)

        # Strain
        if not self.boxsize: self.model_epsilon = StrainMLP(self.dim_hidden, self.n_hidden, self.D, self.h)

        # Entropy teacher
        self.teacher = MLP([1 + self.D] + self.n_hidden * [self.dim_hidden] + [1])

        # Internal Energy  
        self.model_U_vol = MonotonicMLP([2] + self.n_hidden * [self.dim_hidden] + [1], t = [1, -1])
        if not self.boxsize: self.model_U_dev = MonotonicMLP([2] + self.n_hidden * [self.dim_hidden] + [1], t = [1, 0])
        
        # Stochastic Coefficients
        self.model_A = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])
        self.model_B = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])
        self.model_C = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])

        # Extra constants
        self.log_k_B = nn.Parameter(torch.log(torch.tensor(args.k_B)))
        self.log_m = nn.Parameter(torch.log(torch.tensor(args.m)))

    def forward(self, x, edge_index, r0, dataset, train = False):

        x.requires_grad = True
        N = x.size(0)

        # State Variables: x = (r, v)
        r = x[:,:self.D]
        v = x[:,self.D:2*self.D]

        r_ij, v_ij = self.get_distances(r, v, edge_index, dataset)

        i, j = edge_index 
        k_B = torch.exp(self.log_k_B)
        m = torch.exp(self.log_m)
        
        # Density: Kernel Approximation
        W_ij = self.model_W(r_ij)
        W_ji = self.model_W(-r_ij)

        r_ii = torch.zeros(N, 1, device=x.device)
        d = scatter_add(W_ij, i, dim=0, dim_size=N) + scatter_add(W_ji, j, dim=0, dim_size=N) + self.model_W(r_ii)

        # Auxiliary Variables
        r_ = dot(r_ij, r_ij)**0.5
        e_ij = r_ij / (r_ + 1e-8)

        # Entropy estimation
        S = self.model_S(x, edge_index, dataset) if train else x[:,[-1]]
        V = 1/d

        # Shear stresses
        if not self.boxsize:
            # Shear stress
            r0_ij = r0[i] - r0[j]
            u_ij = r_ij - r0_ij
            epsilon_ij = self.model_epsilon(u_ij, r0_ij) - self.model_epsilon(torch.zeros_like(u_ij), r0_ij)
            epsilon_ji = self.model_epsilon(-u_ij, -r0_ij) - self.model_epsilon(torch.zeros_like(-u_ij), -r0_ij)
            epsilon = scatter_add(epsilon_ij, i, dim=0, dim_size=N) + scatter_add(epsilon_ji, j, dim=0, dim_size=N)

            # Traceless
            trace = torch.einsum("bii->b", epsilon) / self.D  # Compute trace
            trace_matrix = trace.view(-1, 1, 1) * torch.eye(self.D, device=x.device)
            epsilon_dev = epsilon - trace_matrix

            # Invariants (for 2D)
            # I1 = torch.einsum("...ii", epsilon)[...,None]
            # I2 = torch.det(epsilon_dev)[...,None]
            I2 = torch.einsum("bij,bij->b", epsilon_dev, epsilon_dev)[..., None]

            # Internal energy
            U_dev = self.model_U_dev(S, I2)

        # Equation of State
        U_vol = self.model_U_vol(S, V)
        U = U_dev + U_vol if not self.boxsize else U_vol
        T = grad(U, S)      # Temperature
        P = -grad(U, V)     # Pressure
        C = T/grad(T, S)    # Specific heat capacity at constant volume

        # Conservative Dynamics
        # gradP_d = -grad(U_vol, r)
        # div_sigma = -grad(U_dev, r)
        F_cons = -grad(U, r)

        # Dissipative dynamics (induced by dx_tilde)
        T_i = T[i]
        T_j = T[j]

        A_ij = self.model_A(r_/self.h, T_i) * self.model_A(r_/self.h, T_j)
        B_ij = self.model_B(r_/self.h, T_i) * self.model_B(r_/self.h, T_j)
        C_ij = self.model_C(r_/self.h, T_i) * self.model_C(r_/self.h, T_j)

        aux = A_ij**2/2 * v_ij + (A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D) * dot(e_ij, v_ij) * e_ij
        term = (1/T_i + 1/T_j) * aux
        MgradS_v = -1/2 * (scatter_add(term, i, dim=0, dim_size=N) - scatter_add(term, j, dim=0, dim_size=N))

        gradA_T_i, gradB_T_i, gradC_T_i = grad(A_ij**2, T_i), grad(B_ij**2, T_i), grad(C_ij**2, T_i)
        gradA_T_j, gradB_T_j, gradC_T_j = grad(A_ij**2, T_j), grad(B_ij**2, T_j), grad(C_ij**2, T_j)

        term = -(1/C[i]/T_i + 1/C[j]/T_j) * aux
        term1 = scatter_add(term, i, dim=0, dim_size=N) - scatter_add(term, j, dim=0, dim_size=N)
        term2 = scatter_add((gradA_T_i/2 * v_ij + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * dot(e_ij, v_ij) * e_ij) / C[i], i, dim=0, dim_size=N) \
            - scatter_add((gradA_T_j/2 * v_ij + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * dot(e_ij, v_ij) * e_ij) / C[j], j, dim=0, dim_size=N)
        term3 = scatter_add((gradA_T_j/2 * v_ij + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * dot(e_ij, v_ij) * e_ij) / C[j], i, dim=0, dim_size=N) \
            - scatter_add((gradA_T_i/2 * v_ij + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * dot(e_ij, v_ij) * e_ij) / C[i], j, dim=0, dim_size=N)
        divM_v = -1/2 * (term1 + term2 + term3)

        aux = (A_ij**2/2 * dot(v_ij, v_ij) + (A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D) * dot(e_ij, v_ij)**2) / 4
        MgradS_S = (scatter_add((1/T[i] + 1/T[j]) * aux + (1/T[i] - 1/T[j]) * C_ij**2, i, dim=0, dim_size=N) \
                  + scatter_add((1/T[j] + 1/T[i]) * aux + (1/T[j] - 1/T[i]) * C_ij**2, j, dim=0, dim_size=N))
        
        term1 = scatter_add(-(2/C[i]/T_i + 1/C[j]/T_j) * aux, i, dim=0, dim_size=N) \
            + scatter_add(-(2/C[j]/T_j + 1/C[i]/T_i) * aux, j, dim=0, dim_size=N)
        term2 = scatter_add((gradA_T_i/2 * dot(v_ij, v_ij) + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * dot(e_ij, v_ij)**2) / C[i] /4, i, dim=0, dim_size=N) \
            + scatter_add((gradA_T_j/2 * dot(v_ij, v_ij) + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * dot(e_ij, v_ij)**2) / C[j] /4, j, dim=0, dim_size=N)
        term3 = scatter_add((gradA_T_j/2 * dot(v_ij, v_ij) + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * dot(e_ij, v_ij)**2) / C[j] /4, i, dim=0, dim_size=N) \
            + scatter_add((gradA_T_i/2 * dot(v_ij, v_ij) + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * dot(e_ij, v_ij)**2) / C[i] /4, j, dim=0, dim_size=N)
        term4 = scatter_add(-(2/C[i]/T_i - 1/C[j]/T_j) * C_ij**2, i, dim=0, dim_size=N) \
            + scatter_add(-(2/C[j]/T_j - 1/C[i]/T_i) * C_ij**2, j, dim=0, dim_size=N)
        term5 = scatter_add(gradC_T_i / C[i] - gradC_T_j / C[j], i, dim=0, dim_size=N) \
            + scatter_add(gradC_T_j / C[j] - gradC_T_i / C[i], j, dim=0, dim_size=N)
        term = -((self.D + 1) * A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D)
        term6 = scatter_add(term, i, dim=0, dim_size=N) + scatter_add(term, j, dim=0, dim_size=N)

        divM_S = term1 + term2 + term3 + term4 + term5 + term6 / m

        # Dynamic Equations   
        drdt = v
        dvdt = (F_cons + MgradS_v + k_B * divM_v) / m
        dSdt = (MgradS_S + k_B * divM_S) / T
        
        if train:
            # Fluctuation-Dissipation Theorem: x_tilde * x_tilde.T = 2 * k_B * M * dt  
            # dvdv_dt
            term = A_ij[...,None]**2/2 * eye_like(e_ij) + (A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D)[...,None] * outer(e_ij, e_ij)
            dvdv_dt = 1/m**2 * (scatter_add(term, i, dim=0, dim_size=N) \
                              + scatter_add(term, j, dim=0, dim_size=N))  

            # dvdS_dt
            term = A_ij**2/2 * v_ij + (A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D) * dot(e_ij, v_ij) * e_ij
            dvdS_dt = -1/2/m/T * (scatter_add(term, i, dim=0, dim_size=N) - scatter_add(term, j, dim=0, dim_size=N))

            # dSdv_dt
            dSdv_dt = dvdS_dt

            # dSdS_dt
            term = A_ij**2/2 * dot(v_ij, v_ij) + (A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D) * dot(e_ij, v_ij)**2
            dSdS_dt = 1/T**2 * (scatter_add(term/4 + C_ij**2, i, dim=0, dim_size=N) \
                              + scatter_add(term/4 + C_ij**2, j, dim=0, dim_size=N))

            # Assembly of marginal M's
            cov = torch.zeros([N, 2*self.D+1, 2*self.D+1], device=x.device)
            cov[:,self.D:2*self.D,self.D:2*self.D] = dvdv_dt
            cov[:,self.D:2*self.D,-1] = dvdS_dt
            cov[:,-1,self.D:2*self.D] = dSdv_dt
            cov[:,2*self.D,[2*self.D]] = dSdS_dt

            return (drdt, dvdt, dSdt), 2 * k_B * self.dt * cov
        
        else:
            # Stochastic Dynamics
            dW_ij_bar, tr_dW_ij, dV_ij = self.get_Wiener(edge_index)

            term = torch.einsum('...ij,...j->...i', A_ij[...,None] * dW_ij_bar + B_ij[...,None] * tr_dW_ij / self.D, e_ij)
            dv_tilde = (2 * k_B)**0.5 / m * (scatter_add(term, i, dim=0, dim_size=N) - scatter_add(term, j, dim=0, dim_size=N)) # dW_ij = dW_ji

            term = -1/2 * dot(term, v_ij)
            term1 = scatter_add(term, i, dim=0, dim_size=N) + scatter_add(term, j, dim=0, dim_size=N)
            term = C_ij * dV_ij
            term2 = scatter_add(term, i, dim=0, dim_size=N) - scatter_add(term, j, dim=0, dim_size=N) # dV_ij = -dV_ji  
            dS_tilde = (2 * k_B)**0.5 / T * (term1 + term2)

            # Conservation laws
            p_sum = m * v.sum(0)
            E_sum = (U + 1/2 * m * dot(v, v)).sum()
            S_sum = S.sum()

            return (drdt, dvdt, dSdt), (dv_tilde, dS_tilde), {'p_sum': p_sum, 'E_sum': E_sum, 'S_sum': S_sum}


    # Entropy encoder
    def model_S(self, x, edge_index, dataset):

            i, j = edge_index
            N = x.size(0)

            r = x[:,:self.D]
            v = x[:,self.D:2*self.D]

            r_ij, v_ij = self.get_distances(r, v, edge_index, dataset)
            r_ = dot(r_ij, r_ij)**0.5

            S = scatter_mean(self.teacher(r_/self.h, v_ij), i, dim=0, dim_size=N) + \
                scatter_mean(self.teacher(r_/self.h, -v_ij), j, dim=0, dim_size=N)

            return S
    

    # Compute relative positions and velocities
    def get_distances(self, r, v, edge_index, dataset):

        # Edges
        i, j = edge_index

        r_ij = r[i] - r[j]
        v_ij = v[i] - v[j]

        # Periodic Boundary Conditions
        if self.boxsize:
            if dataset.name == 'shear_flow':
                shear_rate = dataset.data['shear_rate']
                # Minimum image convention
                r_ij[:,0] -= torch.round(r_ij[:,2]/self.boxsize) * shear_rate * self.boxsize * self.dt
                r_ij[:,0] -= torch.round(r_ij[:,0]/self.boxsize) * self.boxsize
                r_ij[:,1] -= torch.round(r_ij[:,1]/self.boxsize) * self.boxsize
                r_ij[:,2] -= torch.round(r_ij[:,2]/self.boxsize) * self.boxsize
            else:
                # Minimum image convention 
                r_ij -= self.boxsize * torch.round(r_ij / self.boxsize) # PBCs

        return r_ij, v_ij   


    # Noise Gaussian increments
    def get_Wiener(self, edge_index):

        device = edge_index.device
        N = edge_index.size(-1)

        # Matrix of independent Wiener processes
        dW_ij = torch.randn([N, self.D, self.D], device=device)
        # Trace
        I = torch.eye(self.D, device=device).repeat(N, 1, 1)
        tr_dW_ij = I * torch.einsum('...ii', dW_ij)[...,None, None]
        # Traceless symmetric part
        dW_ij_bar = 1/2 * (dW_ij + dW_ij.transpose(1,2)) - tr_dW_ij / self.D
        # Independent term
        dV_ij = torch.randn([N, 1], device=device)
    
        return dW_ij_bar, tr_dW_ij, dV_ij



if __name__ == '__main__':
    pass
