#!/usr/bin/env python3

import os

os.system("bash ./src/patches/01-download_datasets.sh")
os.system("bash ./src/patches/02-generate_qg.sh")

from qg_verify_tensorizer import verify

verify("data/question_generation/test.flant5_aug.jsonl")