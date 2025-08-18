#!/bin/bash
python main.py --train False --dset_train 'viscoelastic' --dset_test 'viscoelastic' \
    --dt 0.01 --h 0.15 --boxsize None --dim_hidden 100 --N_train 800