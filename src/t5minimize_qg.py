#!/usr/bin/env python3

import os

# make sure we have the data
os.system("bash ./src/patches/01-download_datasets.sh")

from qg_find_questions import qg_find_questions
from qg_to_masked import qg_to_masked
from qg_verify_tensorizer import qg_verify

qg_find_questions(
    input_constraints="data/question_generation/raw_constraints.jsonl",
    output="data/question_generation/raw_merged.jsonl",
)
qg_to_masked(
    input="data/question_generation/raw_merged.jsonl",
    output="data/question_generation/test.flant5_aug.jsonl"
)

qg_verify("data/question_generation/test.flant5_aug.jsonl")