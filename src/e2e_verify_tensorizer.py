#!/usr/bin/env python3

from e2e_tensorizer import E2ETensorizer, E2EDataProcessor
import argparse
import json

def e2e_verify(input):
    print("Verifying E2ETensorizer")
    with open(input, "r") as f:
        data = [json.loads(x) for x in f]

    tensorizer = E2ETensorizer()
    output = tensorizer.tensorize_example(data[0])
    print(output)

    print("Verifying E2EDataProcessor")
    data = E2EDataProcessor()
    output = data.get_tensor_examples()[0]
    print(output)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="data/e2enlg/test.flant5_aug.jsonl")
    args = args.parse_args()
    e2e_verify(args.input)