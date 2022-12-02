#!/usr/bin/env python3

from qg_tensorizer import QGTensorizer, QTDataProcessor
import argparse
import json


def verify(input):
    print("Verifying QGTensorizer")
    with open(input, "r") as f:
        data = [json.loads(x) for x in f]

    tensorizer = QGTensorizer()
    output = tensorizer.tensorize_example(data[0])
    print(output)

    print("Verifying QTDataProcessor")
    data = QTDataProcessor()
    output = data.get_tensor_examples()[0]
    print(output)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="data/question_generation/test.flant5_aug.jsonl")
    args = args.parse_args()
    verify(args.input)