#!/bin/bash

set -e

### Natural Instructions ###

git clone https://github.com/allenai/natural-instructions.git /tmp/ni

for num_examples in 16 32 64 100 1000; do
for seed in 13 42 58; do
    echo "Working on seed: $seed"
    python misc/create_ni_dataset.py /tmp/ni --output_path ./dataloader/ni_data/processed/ --seed ${seed} --num_examples ${num_examples}
done
done

rm -rf /tmp/ni

### CrossFit ###

git clone https://github.com/INK-USC/CrossFit.git /tmp/xfit

cd /tmp/xfit/tasks/
python _build_gym.py --build --n_proc 20
cd ..
mkdir -p ../../dataloader/xfit_data/processed
mv data ../../dataloader/xfit_data/processed
cd ../..

rm -rf /tmp/xfit

## T0 ##

python misc/create_t0_dataset.py --output_path ./dataloader/t0_data/processed/
