#!/bin/bash
python main.py --train True --dset_train 'viscoelastic' --dset_test 'viscoelastic' \
    --dt 0.01 --h 0.15 --boxsize None \
    --dim_hidden 100 --miles 9999 --max_epoch 2000 --lr1 1e-3 --lr2 1e-2 --N_train 800
