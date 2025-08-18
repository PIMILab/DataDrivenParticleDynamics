#!/bin/bash
python main.py --train True --dset_train 'taylor_green' --dset_test 'self_diffusion' 'shear_flow' 'taylor_green' \
    --dt 0.0005 --h 0.2 --boxsize 1.0 \
    --miles 1000 2000 --max_epoch 3000 --lr1 1e-3 --lr2 1e-2 --N_train 300
