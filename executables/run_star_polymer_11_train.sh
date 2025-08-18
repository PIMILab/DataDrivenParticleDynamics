#!/bin/bash
python main.py --train True --dset_train 'star_polymer_11' --dset_test 'star_polymer_11' \
    --dt 0.025 --h 4.75 --boxsize 30.18 \
    --dim_hidden 100 --miles 2500 --max_epoch 5000 --lr1 1e-3 --lr2 1e-2 --N_train 400
