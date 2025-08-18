#!/bin/bash
python main.py --train False --dset_train 'needle' --dset_test 'needle' \
    --dt 0.0005 --h 0.11 --boxsize None --dim_hidden 100 --N_train 2000