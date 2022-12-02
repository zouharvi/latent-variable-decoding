#!/usr/bin/bash

# Question Generation dataset
mkdir -p data/question_generation/
wget -N -q --show-progress \
    https://raw.githubusercontent.com/GXimingLu/a_star_neurologic/main/dataset/question_generation/constraints.jsonl \
    -O data/question_generation/raw_constraints.jsonl

# END2END dataset
mkdir -p data/end_to_end/
wget -N -q --show-progress \
    https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip \
    -O data/end_to_end/e2e-dataset.zip

# unzip and cleanup
unzip -u -d data/end_to_end/ data/end_to_end/e2e-dataset.zip
mv data/end_to_end/e2e-dataset/* data/end_to_end/
rm -r data/end_to_end/{README.md,e2e-dataset/}