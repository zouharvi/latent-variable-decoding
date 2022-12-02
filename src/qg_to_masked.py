import tqdm
import json
from utils import get_flant5_tokenizer, compute_mask

def qg_to_masked(input, output):
    from utils import BOS_CONSTRAINTS, BOS_TOKEN
    data_out = []

    with open(input, "r") as f:
        data = [json.loads(x) for x in f]

    tokenizer = get_flant5_tokenizer()

    BOS_CONSTRAINTS = [
        tokenizer.tokenize(BOS_TOKEN + " " + x.capitalize()) for x in BOS_CONSTRAINTS
    ]


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

    with open(output, "w") as f:
        f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in data_out]))
