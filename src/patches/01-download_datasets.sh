#!/usr/bin/bash

# Question Generation dataset
mkdir -p data/question_generation/
wget -N -q --show-progress \
    https://raw.githubusercontent.com/GXimingLu/a_star_neurologic/main/dataset/question_generation/constraints.jsonl \
    -O data/question_generation/raw_constraints.jsonl

# END2END dataset
mkdir -p data/e2enlg{,_filter}/
wget -N -q --show-progress \
    https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip \
    -O data/e2enlg/e2e-dataset.zip

# unzip and cleanup
unzip -u -d data/e2enlg/ data/e2enlg/e2e-dataset.zip
mv data/e2enlg/e2e-dataset/* data/e2enlg/
mv data/e2enlg/testset{_w_refs,}.csv
rm -r data/e2enlg/{README.md,e2e-dataset/}