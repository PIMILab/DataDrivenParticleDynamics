#!/bin/bash
python main.py --train False --dset_train 'taylor_green' --dset_test 'self_diffusion' 'shear_flow' 'taylor_green' \
    --dt 0.0005 --h 0.2 --boxsize 1.0 --N_train 300
