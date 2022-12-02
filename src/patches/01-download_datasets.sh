#!/usr/bin/bash

mkdir -p data/question_generation/

wget -N \
    https://raw.githubusercontent.com/GXimingLu/a_star_neurologic/main/dataset/question_generation/constraints.jsonl \
    -O data/question_generation/raw_constraints.jsonl

# TODO: E2E by Ondra Dušek