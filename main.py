"""main.py"""

import argparse

import numpy as np
import torch

from solver import Solver
from utils import str2bool, none_or_float


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    solver = Solver(args)
    # Train
    if args.train: 
        solver.train_model()
        # Save JIT
        solver.export_jit()
    # Test
    for dset in args.dset_test:
        solver.test_model(dset)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Learning of Smoothed Dissipative Dynamics')

    # Study Case
    parser.add_argument('--train', default=False, type=str2bool, help='train or test')
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')

    # Dataset Parameters
    parser.add_argument('--dset_train', default='taylor_green', type=str, help='training dataset directory')
    parser.add_argument('--dset_test', default=['self_diffusion', 'shear_flow', 'taylor_green'], nargs='+', help='test dataset directory')
    parser.add_argument('--dt', default=1.0, type=float, help='time step')
    parser.add_argument('--h', default=0.2, type=float, help='cutoff radius')
    parser.add_argument('--boxsize', default=1.0, type=none_or_float, help='length of periodic box or None')

    # Net Parameters
    parser.add_argument('--n_hidden', default=2, type=int, help='number of hidden layers per MLP')
    parser.add_argument('--dim_hidden', default=50, type=int, help='dimension of hidden units')
    parser.add_argument('--m', default=1.0, type=float, help='mass per fluid particle initial value')
    parser.add_argument('--k_B', default=1.0, type=float, help='Boltzmann constant initial value')

    # Training Parameters
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--lr1', default=1e-3, type=float, help='learning rate networks')
    parser.add_argument('--lr2', default=1e-3, type=float, help='learning rate parameters')
    parser.add_argument('--batch_size', default=5, type=int, help='training batch size')
    parser.add_argument('--shuffle', default=True, type=str2bool, help='shuffle train snapshots')
    parser.add_argument('--max_epoch', default=3000, type=int, help='maximum training iterations')
    parser.add_argument('--miles', default=[1000, 2000], nargs='+', type=int, help='learning rate scheduler milestones')
    parser.add_argument('--gamma', default=1e-1, type=float, help='learning rate milestone decay')
    parser.add_argument('--N_train', default=300, type=int, help='number of training snapshots')

    args = parser.parse_args()

    main(args)
