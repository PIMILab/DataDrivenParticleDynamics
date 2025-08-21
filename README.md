
<div align="center">  
  
# Data-driven particle dynamics: Structure-preserving coarse-graining for non-equilibrium systems

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2508.12569)

</div>

## Abstract

Multiscale systems are ubiquitous in science and technology, but are notoriously challenging to simulate as short spatiotemporal scales must be appropriately linked to emergent bulk physics. When expensive high-dimensional dynamical systems are coarse-grained into low-dimensional models, the entropic loss of information leads to emergent physics which are dissipative, history-dependent, and stochastic. To machine learn coarse-grained dynamics from time-series observations of particle trajectories, we propose a framework using the metriplectic bracket formalism that preserves these properties by construction; most notably, the framework guarantees discrete notions of the first and second laws of thermodynamics, conservation of momentum, and a discrete fluctuation-dissipation balance crucial for capturing non-equilibrium statistics. We introduce the mathematical framework abstractly before specializing to a particle discretization. As labels are generally unavailable for entropic state variables, we introduce a novel self-supervised learning strategy to identify emergent structural variables. We validate the method on benchmark systems and demonstrate its utility on two challenging examples: (1) coarse-graining star polymers at challenging levels of coarse-graining while preserving non-equilibrium statistics, and (2) learning models from high-speed video of colloidal suspensions that capture coupling between local rearrangement events and emergent stochastic dynamics. We provide open-source implementations in both PyTorch and LAMMPS, enabling large-scale inference and extensibility to diverse particle-based systems.

For more information, please refer to the following:

- Hernandez, Quercus and Win, Max and O'Connor, Thomas C. and Arratia, Paulo E. and Trask, Nathaniel. "[Data-driven particle dynamics: Structure-preserving coarse-graining for emergent behavior in non-equilibrium systems](https://arxiv.org/abs/2508.12569)." Arxiv, under review (2025).

<div align="center">
<img src="/data/teaser.png" width="500">
</div>

## Setting it up

First, clone the project.

```bash
# clone project
git clone https://github.com/PIMILab/DataDrivenParticleDynamics.git
cd DataDrivenParticleDynamics
```

Then, install the needed dependencies. The code is implemented in [Pytorch](https://pytorch.org). _Note that this has been tested using Python 3.11_.

```bash
# install dependencies
pip install numpy scipy matplotlib pytorch torch-geometric tidynamics MDAnalysis
 ```

To download the datasets, you can use the following Googe Drive [Link](https://drive.google.com/uc?export=download&id=1M67-Ty3M9vKNAKsCK-clFXUGPoFspI6-).

## How to run the code  

### Test pretrained nets

The results of the paper (Ideal gas, Star Polymer 11, Star Polymer 51, Viscoelastic and Needle) can be reproduced with the following scripts, found in the `executables/` folder.

```bash
bash executables/run_ideal_gas_test.sh
bash executables/run_star_polymer_11_test.sh
bash executables/run_star_polymer_51_test.sh
bash executables/run_viscoelastic_test.sh
bash executables/run_needle_test.sh
```

The `data/` folder includes the database and the pretrained parameters of the networks. The resulting time evolution, correlation statistics of the state variables and GIF of the system is plotted and saved in a the `outputs/` folder.

### Train a custom net

You can run your own experiments for the toy sdpd dataset by setting custom parameters manually. The trained parameters and output plots are saved in the `outputs/` folder.

```bash
e.g.
python main.py --dset_train 'self_diffusion' --train True --lr 1e-3 ...
```

General Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--train`                 | Train mode                                        | `True`, `False`                                       |
| `--gpu`                   | Enable GPU acceleration                           | `True`, `False`                                       |

Dataset Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--dset_train`            | Training dataset                                  | `self_diffusion`, `shear_flow`, `taylor_green`, `star_polymer`, `viscoelastic`, `needle` |
| `--dset_test`             | Test dataset                                      | `self_diffusion`, `shear_flow`, `taylor_green`, `star_polymer`, `viscoelastic`, `needle` |
| `--dt`                    | Time step                                         | Default: `1.0`                                       |
| `--h`                     | Cutoff radius                                     | Default: `0.2`                                       |
| `--boxsize`               | Box size of PBCs                                  | Default: `1.0`                                       |

Network Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--n_hidden`              | Number of MLP hidden layers                       | Default: `2`                                          |
| `--dim_hidden`            | Dimension of hidden layers                        | Default: `50`                                         |
| `--m`                     | Mass initial value                                | Default: `1.0`                                        |
| `--k_B`                   | Boltzmann constant initial value                  | Default: `1.0`                                       | 

Training Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--lr1`                   | Learning rate networks                            | Default: `1e-2`                                       |
| `--lr2`                   | Learning rate parameters                          | Default: `1e-2`                                       |
| `--batch_size`            | Training batch size                               | Default: `50`                                         |
| `--shuffle`               | Shuffle train snapshots                           | `True`, `False`                                       |
| `--max_epoch`             | Maximum number of training epochs                 | Default: `3000`                                       |
| `--miles`                 | Learning rate scheduler milestones                | Default: `1000 2000`                                  |
| `--gamma`                 | Learning rate scheduler decay                     | Default: `1e-1`                                       |
| `--N_train`               | Number of training snapshots                      | Default: `300`                                        |

## LAMMPS Interface
To use the trained Pytorch model for SDPD simulations in LAMMPS program please refer to its own [Github repository](https://github.com/PIMILab/DataDrivenParticleDynamicsForLAMMPS).

## Citation
If you found this code useful please cite our work as:

```
@article{hernandez2025data,
  title={Data-driven particle dynamics: Structure-preserving coarse-graining for emergent behavior in non-equilibrium systems},
  author={Hernandez, Quercus and Win, Max and O'Connor, Thomas C and Arratia, Paulo E and Trask, Nathaniel},
  journal={arXiv preprint arXiv:2508.12569},
  year={2025}
}
```
