"""solver.py"""

import os
import numpy as np

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from scipy.spatial import KDTree
from tqdm import tqdm
import json

from model import CG_model
from model_jit import CG_model_jit
from model_jit import CG_model_S_jit
from model_jit import CG_model_W_jit
from dataset import load_dataset
from utils import save_log, plot_loss, plot_sim, plot_metrics, dot, gif_sim

import MDAnalysis as mda
from MDAnalysis.analysis.msd import EinsteinMSD
from MDAnalysis.analysis.rdf import InterRDF


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Study Case
        self.device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

        # Dataset Parameters
        self.train_set, self.val_set = load_dataset(args.dset_train, args.N_train)
        self.dims = self.train_set.dims
        self.dt = args.dt
        self.h = args.h
        self.boxsize = args.boxsize

        # Training Parameters
        self.max_epoch = args.max_epoch
        self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=args.shuffle)
        self.val_loader = DataLoader(self.val_set, batch_size=args.batch_size, shuffle=args.shuffle)

        # Load/Save options
        self.output_dir = os.path.join('outputs', args.dset_train)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Net Parameters
        self.model = CG_model(args, self.dims).to(self.device).float()
        if (args.train == False):
            # Load pretrained net
            load_dir = os.path.join('data', args.dset_train, 'params.pt')
            checkpoint = torch.load(load_dir, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint)

        params1 = [p for name, p in self.model.named_parameters() if name not in ['log_k_B', 'log_m']]
        self.optim1 = optim.Adam(params1, lr=args.lr1)
        params2 = [self.model.log_k_B, self.model.log_m]
        self.optim2 = optim.Adam(params2, lr=args.lr2)

        self.scheduler1 = optim.lr_scheduler.MultiStepLR(self.optim1, milestones=args.miles, gamma=args.gamma)
        self.scheduler2 = optim.lr_scheduler.MultiStepLR(self.optim2, milestones=args.miles, gamma=args.gamma)


    def train_model(self):
        epoch = 0
        train_log = {'epoch':[], 'loss_mu':[], 'loss_var':[], 'loss_mse':[]}
        val_log = {'epoch':[], 'loss_mu':[], 'loss_var':[], 'loss_mse':[]}

        print("\n[Training Started]\n")
        
        # Main training loop
        while (epoch < self.max_epoch):
            print('[Epoch: {}]'.format(epoch+1))

            # Train set loop
            loss_mu_sum = 0
            loss_var_sum = 0
            loss_mse_sum = 0
            for snaps in self.train_loader:
                snaps = snaps.to(self.device)

                # Get data
                x, x1, r0 = snaps.x, snaps.y, getattr(snaps, 'r0', None)
                edge_index = snaps.edge_index
                edge_index1 = snaps.edge_index1
                mask = ~snaps.mask['exterior']

                # Net forward pass
                dxdt_net, cov = self.model(x, edge_index, r0, self.train_set, train = True)
                S = self.model.model_S(x, edge_index, self.train_set)
                S1 = self.model.model_S(x1, edge_index1, self.train_set)
                x = torch.cat([x[:,:-1], S], dim=-1)
                x1 = torch.cat([x1[:,:-1], S1], dim=-1)

                # Loss: Negative Log-Likelihood
                loss_mu, loss_var, loss_mse = self.compute_loss(x, x1, dxdt_net, cov, mask)
                loss = loss_mu + loss_var

                # Backpropagation
                self.optim1.zero_grad()
                self.optim2.zero_grad()
                loss.backward()
                self.optim1.step()
                self.optim2.step()

                # Save losses
                loss_mu_sum += loss_mu.item()
                loss_var_sum += loss_var.item()
                loss_mse_sum += loss_mse.item()

            # Learning rate scheduler
            self.scheduler1.step()
            self.scheduler2.step()

            # Train log 
            loss_mu_train = loss_mu_sum / len(self.train_loader)
            loss_var_train = loss_var_sum / len(self.train_loader)
            loss_mse_train = loss_mse_sum / len(self.train_loader)
            train_log['epoch'].append(epoch+1)
            train_log['loss_mu'].append(loss_mu_train)
            train_log['loss_var'].append(loss_var_train)
            train_log['loss_mse'].append(loss_mse_train)

            # Validation set loop
            loss_mu_sum = 0
            loss_var_sum = 0
            loss_mse_sum = 0
            for snaps in self.val_loader:
                snaps = snaps.to(self.device)

                # Get data
                x, x1, r0 = snaps.x, snaps.y, getattr(snaps, 'r0', None)
                edge_index = snaps.edge_index
                edge_index1 = snaps.edge_index1
                mask = ~snaps.mask['exterior']

                # Net forward pass
                dxdt_net, cov = self.model(x, edge_index, r0, self.val_set, train = True)
                S = self.model.model_S(x, edge_index, self.val_set)
                S1 = self.model.model_S(x1, edge_index1, self.val_set)
                x = torch.cat([x[:,:-1], S], dim=-1)
                x1 = torch.cat([x1[:,:-1], S1], dim=-1)

                # Loss: Negative Log-Likelihood
                loss_mu, loss_var, loss_mse = self.compute_loss(x, x1, dxdt_net, cov, mask)

                # Save losses
                loss_mu_sum += loss_mu.item()
                loss_var_sum += loss_var.item()
                loss_mse_sum += loss_mse.item()

            # Validation log
            loss_mu_val = loss_mu_sum / len(self.val_loader)
            loss_var_val = loss_var_sum / len(self.val_loader)
            loss_mse_val= loss_mse_sum / len(self.val_loader)
            val_log['epoch'].append(epoch+1)
            val_log['loss_mu'].append(loss_mu_val)
            val_log['loss_var'].append(loss_var_val)
            val_log['loss_mse'].append(loss_mse_val)

            # Print Loss
            print('Mean Loss: {:1.2e} (Train) / {:1.2e} (Val)'.format(loss_mu_train, loss_mu_val))
            print('Var Loss: {:1.2e} (Train) / {:1.2e} (Val)'.format(loss_var_train, loss_var_val))
            print('MSE Loss: {:1.2e} (Train) / {:1.2e} (Val)'.format(loss_mse_train, loss_mse_val))
            print('k_B = {:1.2e}, m = {:1.2e}'.format(torch.exp(self.model.log_k_B).item(), torch.exp(self.model.log_m).item()))

            epoch += 1

        print("[Training Finished]\n")

        # Save net parameters
        save_dir = os.path.join(self.output_dir, 'params.pt')
        torch.save(self.model.state_dict(), save_dir)
        save_dir = os.path.join(self.output_dir, 'args.json')
        with open(save_dir, 'w') as f:
            json.dump(vars(self.args), f, indent=4)

        # Plot and save losses
        plot_loss(train_log, val_log, self.output_dir)
        save_log(train_log, self.output_dir, 'train')
        save_log(val_log, self.output_dir, 'val')   



    def test_model(self, dset):

        print('[Test Set \'' + dset + '\' Evaluation]')

        # Load/Save options
        output_dir = os.path.join(self.output_dir, dset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        test_set = load_dataset(dset)
        results_gt = test_set.data

        # Integrate simulation
        results_net = self.integrate_sim(test_set, T=len(test_set))
        
        # Plot results
        plot_sim(results_net, results_gt, output_dir)
        plot_metrics(results_net, results_gt, output_dir, self.boxsize)
        gif_sim(results_net, results_gt, self.boxsize, output_dir)

        # Compute MSE
        if self.boxsize: 
            RRMSE_VACF = torch.mean((results_net['VACF'] - results_gt['VACF'])**2).item()**0.5 / torch.mean(results_gt['VACF']**2).item()**0.5
            RRMSE_RDF = torch.mean((results_net['RDF'][0] - results_gt['RDF'][0])**2).item()**0.5 / torch.mean(results_gt['RDF'][0]**2).item()**0.5
            RRMSE_MSD = torch.mean((results_net['MSD'] - results_gt['MSD'])**2).item()**0.5 / torch.mean(results_gt['MSD']**2).item()**0.5
            print('RRMSE_VACF: {:1.2e}, RRMSE_RDF: {:1.2e}, RRMSE_MSD: {:1.2e}'.format(RRMSE_VACF, RRMSE_RDF, RRMSE_MSD))
        else:
            RRMSE_R = torch.mean((results_net['x'][...,:self.dims] - results_gt['x'][...,:self.dims])**2).item()**0.5 / torch.mean(results_gt['x'][...,:self.dims]**2).item()**0.5
            RRMSE_RDF = torch.mean((results_net['RDF'][0] - results_gt['RDF'][0])**2).item()**0.5 / torch.mean(results_gt['RDF'][0]**2).item()**0.5

            print('RRMSE_R: {:1.2e}, RRMSE_RDF: {:1.2e}'.format(RRMSE_R, RRMSE_RDF))       

        print('[Test Set \'' + dset + '\' Finished]\n')


    # Integrate a single simulation
    def integrate_sim(self, dataset, T):

        N_nodes = dataset[0].x.size(0)
        dset = dataset.name
        T_extrap = 25*self.args.N_train if self.boxsize else T

        # Preallocation
        x_net = torch.zeros([T_extrap + 1, N_nodes, 2*self.dims + 1])
        p_sum = torch.zeros([T_extrap, self.dims])
        E_sum = torch.zeros(T_extrap)
        S_sum = torch.zeros(T_extrap)

        # Initial conditions
        snap = dataset[0].to(self.device)
        x0 = snap.x.clone()
        r0 = getattr(snap, 'r0', None)
        r0 = r0.clone() if r0 is not None else None
        x0[:,[-1]] = self.model.model_S(x0, snap.edge_index, dataset).detach()
        x_net[0] = x0

        # Rollout loop
        x = x0.to(self.device)  
        for t in range(T_extrap):
            edge_index = self.get_edges(x, dset)

            # Net forward pass + Integration
            dxdt_net, dx_tilde_net, aux_vars = self.model(x, edge_index, r0, dataset)
            x1_net = self.integrate_step(x, dxdt_net, dx_tilde_net, dataset)

            if not self.boxsize:
                # Prescribe BCs
                snap = dataset[t].to(self.device)
                x1_net[snap.mask['exterior'],:self.dims*2] = snap.y[snap.mask['exterior'],:self.dims*2] 

            # Save results
            x_net[t+1] = x1_net.detach()
            p_sum[t] = aux_vars['p_sum'].detach()
            E_sum[t] = aux_vars['E_sum'].detach()
            S_sum[t] = aux_vars['S_sum'].detach()

            # Update
            x = x1_net.detach() 

        # Compute Metrics
        VACF_net, RDF_net, r_RDF_net, MSD_net = self.compute_metrics(x_net[:,snap.mask['interior'].cpu()], T)

        # Save results
        results = {'x': x_net.cpu(),
                   'p_sum': p_sum.cpu(), 'E_sum': E_sum.cpu(), 'S_sum': S_sum.cpu(),
                   'VACF': VACF_net, 'RDF': (RDF_net, r_RDF_net), 'MSD': MSD_net}

        return results


    # Compute edges from positions
    def get_edges(self, x, dset):
        r = x[:,:self.dims].cpu().detach()

        # Get directed unique edges
        if self.boxsize:
            if dset == 'shear_flow':
                tree = KDTree((r + self.boxsize/2) % self.boxsize, boxsize=self.boxsize)
            else:
                tree = KDTree((r % self.boxsize) % self.boxsize, boxsize=self.boxsize)
        else:
            tree = KDTree(r)

        pairs = tree.query_pairs(self.args.h - 1e-5, output_type='ndarray')
        edge_index = torch.tensor(pairs).T.to(self.device)

        return edge_index 
    

    # Integrates a single forward step
    def integrate_step(self, x, dxdt, dx_tilde, dataset):

        # Unpack variables
        r = x[:,:self.dims]
        v = x[:,self.dims:2*self.dims]
        S = x[:,2*self.dims:]

        _, dvdt, dSdt = dxdt
        dv_tilde, dS_tilde = dx_tilde

        # Boundary Conditions: Forcing term
        if dataset.name == 'taylor_green':
            f0, a, b, c = dataset.data['forcing_params']
            # Forcing
            dvdt[:,0] += f0 * torch.cos(a*r[:,0]) * torch.sin(b*r[:,1]) * torch.sin(c*r[:,2])
            dvdt[:,1] += f0 * torch.sin(a*r[:,0]) * torch.cos(b*r[:,1]) * torch.sin(c*r[:,2])
            dvdt[:,2] += f0 * torch.sin(a*r[:,0]) * torch.sin(b*r[:,1]) * torch.cos(c*r[:,2])

        # Integration: Leapfrog Verlet
        v1 = v + dvdt * self.dt + dv_tilde * self.dt**0.5
        r1 = r + v1 * self.dt
        S1 = S + dSdt * self.dt + dS_tilde * self.dt**0.5

        x1 = torch.cat((r1, v1, S1), dim=-1)

        # Boundary Conditions: Forcing term
        if dataset.name == 'shear_flow':
            shear_rate = dataset.data['shear_rate']
            x1[:,0] -= torch.round(x1[:,2]/self.boxsize) * shear_rate * self.boxsize * self.dt
            x1[:,0] -= torch.round(x1[:,0]/self.boxsize) * self.boxsize
            x1[:,3] -= torch.round(x1[:,2]/self.boxsize) * shear_rate * self.boxsize
            x1[:,1] -= torch.round(x1[:,1]/self.boxsize) * self.boxsize
            x1[:,2] -= torch.round(x1[:,2]/self.boxsize) * self.boxsize  

        return x1
    

    # Compute NLL loss
    def compute_loss(self, x, x1, dxdt_net, cov, mask):

        # Stochastic variables: Velocity and Entropy 
        x = x[mask, self.dims:]
        x1 = x1[mask, self.dims:]
        cov = cov[mask, self.dims:, self.dims:]  # Covariance matrix for the stochastic variables
        dxdt_net = torch.cat(dxdt_net, dim=-1)[mask, self.dims:]

        # Mean: Euler-Maruyama
        mu = x + dxdt_net * self.dt

        # Negative Log-Likelihood
        loss_mu = dot(x1 - mu, torch.einsum('...ij,...j->...i', torch.linalg.inv(cov), x1 - mu))
        loss_var = torch.logdet(cov)
        loss_mse = dot(x1 - mu, x1 - mu)

        return loss_mu.mean(), loss_var.mean(), loss_mse.mean()


    # Compute metrics
    def compute_metrics(self, x, T):

        # Extract positions and velocities
        R = x[-T-1:,:,:self.dims].numpy()
        V = x[-T-1:,:,self.dims:2*self.dims].numpy()

        N_snapshots, N_particles, D = R.shape

        # Create Universe for MDAnalysis
        universe = mda.Universe.empty(N_particles, trajectory=True, velocities=True)
        universe.load_new(R, order='fac', velocities=V, dimensions=[self.boxsize, self.boxsize, self.boxsize, 90, 90, 90])

        if self.boxsize:

            # Radial distribution function (RDF)
            rdf = InterRDF(universe.atoms, universe.atoms, nbins=75, range=(self.boxsize/100, self.boxsize/2))
            rdf.run(start=0, stop=N_snapshots, step=N_snapshots//100)
            RDF = torch.tensor(rdf.results.rdf)
            r_RDF = torch.tensor(rdf.results.bins)


            # Mean squared displacement (MSD)
            msd = EinsteinMSD(universe, select='all', msd_type="xyz", fft=True)
            msd.run()
            MSD = torch.tensor(msd.results.timeseries)

            # Velocity autocorrelation function (VACF)
            vacf = np.zeros(N_snapshots)
            for i in range(N_particles):
                # Sum of correlations over dimensions
                for d in range(D):
                    vacf_d = np.correlate(V[:,i,d], V[:,i,d], mode='full')
                    vacf += vacf_d[vacf_d.size // 2:]

            VACF = torch.tensor(vacf) / N_particles / N_snapshots

        else:
            # Radial distribution function (RDF)
            r_max = 3*self.h 
            dr = r_max/100
            volume = 1.0
            bin_edges = np.arange(0, r_max + dr, dr)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            rdf_hist = np.zeros_like(bin_centers)

            for snap in range(N_snapshots):
                positions = R[snap]

                # Compute pairwise distances
                for i in range(N_particles):
                    deltas = positions - positions[i]
                    distances = np.linalg.norm(deltas, axis=1)

                    # Remove self-distance
                    distances = distances[distances > 1e-8]

                    # Histogram the distances
                    hist, _ = np.histogram(distances, bins=bin_edges)
                    rdf_hist += hist

            # Normalize RDF
            density = N_particles / volume
            shell_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
            norm = density * shell_areas * N_particles * N_snapshots

            RDF = torch.tensor(rdf_hist / norm)
            r_RDF = torch.tensor(bin_centers)
            VACF, MSD = None, None

        return VACF, RDF, r_RDF, MSD


    # Export JIT model
    def export_jit(self):

        # Load parameters to JIT class
        load_path = os.path.join('data', self.args.dset_train, 'params.pt')
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model_S_jit = CG_model_S_jit(self.args, self.dims).to(self.device).float()
        self.model_S_jit.load_state_dict({k: v for k, v in checkpoint.items() if k.startswith('teacher.')})

        self.model_W_jit = CG_model_W_jit(self.args, self.dims).to(self.device).float()
        self.model_W_jit.load_state_dict({k: v for k, v in checkpoint.items() if k.startswith('model_W.')})

        self.model_jit = CG_model_jit(self.args, self.dims).to(self.device).float()
        self.model_jit.load_state_dict({k: v for k, v in checkpoint.items() if not k.startswith('teacher.')})

        # Check I/O
        inputs = {'r_ij': torch.rand((1000,3), device=self.device),
                  'v': torch.rand((100,3), device=self.device),
                  'edge_index': torch.randint(0,100,(2,1000), device=self.device)}
        _ = self.model_S_jit(inputs)

        inputs = {'r_ij': torch.rand((1000,3), device=self.device),
                  'edge_index': torch.randint(0,100,(2,1000), device=self.device),
                  'N': 100 * torch.ones((1), device=self.device, dtype=torch.int64)}
        _ = self.model_W_jit(inputs)

        inputs = {'v': torch.rand((1000,3), device=self.device), 
                  'S': torch.rand((1000,1), device=self.device), 
                  'edge_index': torch.randint(0,3,(2,10000), device=self.device),
                  'r_ij': torch.rand((10000,3), device=self.device),
                  'd': torch.rand((1000,1), device=self.device),
                  'dW': torch.rand((10000,3,3), device=self.device),
                  'dV': torch.rand((10000,1), device=self.device)}
        _ = self.model_jit(inputs)
        #from model_jit import CG_model_jit
        #self.model_jit = CG_model_jit(self.args, self.dims).to(self.device).float()
        #self.model_jit.load_state_dict({k: v for k, v in checkpoint.items() if not k.startswith('teacher.')})

        # Save JIT network
        save_dir = os.path.join(self.output_dir, 'params_S_jit.pt')
        torch.jit.script(self.model_S_jit).save(save_dir)
        save_dir = os.path.join(self.output_dir, 'params_W_jit.pt')
        torch.jit.script(self.model_W_jit).save(save_dir)
        save_dir = os.path.join(self.output_dir, 'params_jit.pt')
        torch.jit.script(self.model_jit).save(save_dir)



if __name__ == '__main__':
    pass
