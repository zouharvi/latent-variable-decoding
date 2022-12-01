#!/usr/bin/env python3

from qg_tensorizer import QGTensorizer, QTDataProcessor
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/qg_masked.flant5_aug.jsonl")
args = args.parse_args()

print("Verifying QGTensorizer")
with open(args.input, "r") as f:
    data = [json.loads(x) for x in f]

tensorizer = QGTensorizer()
output = tensorizer.tensorize_example(data[0])
print(output)

print("Verifying QTDataProcessor")
data = QTDataProcessor()
output = data.get_tensor_examples()[0]
print(output)