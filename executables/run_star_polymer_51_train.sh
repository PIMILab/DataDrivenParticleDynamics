#!/bin/bash
python main.py --train True --dset_train 'star_polymer_51' --dset_test 'star_polymer_51' \
    --dt 0.04 --h 7.61 --boxsize 63.41 \
    --dim_hidden 50 --miles 2500 --max_epoch 5000 --lr1 1e-3 --lr2 1e-2 --N_train 400
