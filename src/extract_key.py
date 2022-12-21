#!/usr/bin/env python3

from e2e_tensorizer import E2ETensorizer, E2EDataProcessor
import argparse
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # args.add_argument("-i", "--input", default="data/e2enlg_filter/test.flant5_aug.jsonl")
    # args.add_argument("-o", "--output", default="data/e2enlg_filter/test.flant5_aug.target_sentence_raw.txt")
    args.add_argument("-i", "--input", default="data/question_generation/test.flant5_aug.jsonl")
    args.add_argument("-o", "--output", default="data/question_generation/test.flant5_aug.target_sentence_raw.txt")
    args.add_argument("-k", "--key", default="target_sentence_raw")
    args = args.parse_args()

    data = [json.loads(x) for x in open(args.input, "r").readlines()]
    data = [x[args.key] for x in data]
    with open(args.output, "w") as f:
        f.write("\n".join(data))