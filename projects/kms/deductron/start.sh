#!/bin/bash

python run_spawn.py -m ll8b -o /data/alsordon/test_ql_grpo_sae/ -a grpo -s 42 -k 5 --dataset quality --onlbsz 64 -b sglc --epc 30 --offepc 2 --offbsz 32 --lr 3e-6 -P 2 --kl_ctl 1e-4 --task s_ae
