#!/usr/bin/bash

mkdir -p computed

wget \
    https://raw.githubusercontent.com/GXimingLu/a_star_neurologic/main/dataset/question_generation/constraints.jsonl \
    -O computed/qg_constraints.jsonl

