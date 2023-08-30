#!/bin/bash

set -e
 
mkdir -p data/
cd data

### Natural Instructions ###
git clone https://github.com/allenai/natural-instructions.git sni
export NI_DATA_DIR=$PWD/sni
rm -rf /tmp/ni

### MMLU ###
mkdir -p mmlu
cd mmlu

wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
rm data.tar
export MMLU_DATA_DIR=$PWD/data

cd ../..