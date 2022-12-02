#!/usr/bin/env python3

import os

# make sure we download the data
os.system("bash ./src/patches/01-download_datasets.sh")


from e2e_to_masked import e2e_to_masked

for split in {"dev", "train", "test"}:
    e2e_to_masked(
        input=f"data/e2enlg/{split}set.csv",
        output=f"data/e2enlg/{split}.flant5_aug.jsonl"
    )

from e2e_verify_tensorizer import e2e_verify
e2e_verify("data/e2enlg/dev.flant5_aug.jsonl")