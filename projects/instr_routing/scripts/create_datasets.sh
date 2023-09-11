#!/bin/bash

set -e
 
mkdir -p data/
cd data

### MMLU ###
mkdir -p mmlu
cd mmlu

wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
rm data.tar
export MMLU_DATA_DIR=$PWD/data

cd ../..

### Natural Instructions ###
git clone https://github.com/allenai/natural-instructions.git sni
export NI_DATA_DIR=$PWD/sni
rm -rf /tmp/ni
# run test instance reordering
cd $NI_DATA_DIR
python src/reorder_instances_for_testing.py
python eval/leaderboard/create_reference_file.py
