#!/bin/bash

python run_spawn.py -m ll8b -o /data/alsordon/test_rft_big_ncp_ema_10/ -a rft -s 42 -k 5 --onlbsz 64 -b sglc --epc 30 --offepc 2 --offbsz 32 --lr 1e-5 -P 2 --kl_ctl 0. --task s_ncp
python run_spawn.py -m ll8b -o /data/alsordon/test_rloo_big_ncp_ema_10/ -a rloo -s 42 -k 5 --onlbsz 64 -b sglc --epc 10 --offepc 2 --offbsz 32 --lr 5e-6 -P 2 --kl_ctl 0. --task s_ncp
