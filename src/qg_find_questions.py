#!/usr/bin/env python3

import argparse
import datasets
import tqdm
import json
from utils import BOS_CONSTRAINTS

args = argparse.ArgumentParser()
args.add_argument(
    "-ic", "--input-constraints",
    default="computed/qg_constraints.jsonl"
)
args.add_argument(
    "-it", "--input-targets",
    default="computed/qg_targets.jsonl"
)
args.add_argument("-o", "--output", default="computed/qg_merged.jsonl")
args = args.parse_args()

data_squad = datasets.load_dataset("squad_v2")
data_squad = [
    (x["question"].lower().strip(), x["question"], x["context"])
    for x in tqdm.tqdm(data_squad["validation"],)
]

data_out = []

with open(args.input_constraints, "r") as f:
    data_constraints = [json.loads(x) for x in f]

for line_i, line_constrs in enumerate(tqdm.tqdm(data_constraints)):
    # take all contraints but lowercase them
    constrs = [
        [x.lower() for x in constr]
        for constr in line_constrs if len(constr) == 1
    ]
    constrs_large = [
        [x.lower() for x in constr]
        for constr in line_constrs if len(constr) != 1
    ][0] + BOS_CONSTRAINTS
    constrs.append(constrs_large)

    matching = [
        (question, context) for question_lower, question, context in data_squad
        if all(
            any(x in question_lower for x in constr)
            for constr in constrs
        )
    ]

    # assert that there is always one matching
    assert len(matching) >= 1

    data_out.append({
        "doc_id": line_i,
        "target_questions": [x[0] for x in matching],
        "contexts": [x[1] for x in matching],
        "constraints": line_constrs,
    })


with open(args.output, "w") as f:
    f.write("\n".join([json.dumps(x) for x in data_out]))
