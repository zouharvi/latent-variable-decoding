#!/usr/bin/env python3

import argparse
import tqdm
import json
from utils import BOS_CONSTRAINTS, BOS_TOKEN, get_flant5_tokenizer

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/qg_merged.jsonl")
args.add_argument("-o", "--output", default="computed/qg_masked.flant5_aug.jsonl")
args = args.parse_args()

data_out = []

with open(args.input, "r") as f:
    data = [json.loads(x) for x in f]

tokenizer = get_flant5_tokenizer()

BOS_CONSTRAINTS = [
    tokenizer.tokenize(BOS_TOKEN + " " + x.capitalize()) for x in BOS_CONSTRAINTS
]


def index_subsequence(hay, needle):
    # inefficient way of finding a subsequence
    # turn this into Aho-Corasick using regexes
    for i in range(len(hay) - len(needle)):
        subhay = hay[i:i + len(needle)]
        if all(x == y for x, y in zip(subhay, needle)):
            return i, i + len(needle)
    return None


def compute_mask(sent_tok, constraints):
    term_mask = [-1] * len(sent_tok)

    for clause_i, clause in enumerate(constraints):
        # check if anything from the clause is
        for literal in clause:
            indicies = index_subsequence(sent_tok, literal)
            if indicies is None:
                continue

            for i in range(indicies[0], indicies[1]):
                term_mask[i] = clause_i

    return term_mask


for line in tqdm.tqdm(data):
    clause_constrs_single = [x[0] for x in line["constraints"] if len(x) == 1]
    clause_constrs = line["constraints"]
    keywords_tok = [
        [tokenizer.tokenize(x) for x in clause]
        for clause in clause_constrs
    ]

    target_sentence = BOS_TOKEN + " " + line["target_questions"][0]
    target_sentence_tok = tokenizer.tokenize(target_sentence)

    constraints = [BOS_CONSTRAINTS] + keywords_tok

    input_sentence_tok = tokenizer.tokenize(
        BOS_TOKEN + " Question about " + " ".join(clause_constrs_single) + ":"
    )

    target_term_mask = compute_mask(target_sentence_tok, constraints)
    input_term_mask = compute_mask(input_sentence_tok, constraints)

    line_out = {
        "doc_id": line["doc_id"],
        "input_sentence": input_sentence_tok,
        "target_sentence": target_sentence_tok,
        "input_term_mask": input_term_mask,
        "target_term_mask": target_term_mask,
        "target_constraints": constraints,
    }
    data_out.append(line_out)

with open(args.output, "w") as f:
    f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in data_out]))
