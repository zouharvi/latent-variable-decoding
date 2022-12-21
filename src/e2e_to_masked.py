import csv
import json
import tqdm
import re
from utils import get_flant5_tokenizer, compute_mask

RE_VERB_SUBJ = re.compile(r"(.+)\[(.+)\]")
RE_SPLIT = re.compile(r"([a-z]+).*")


def get_verb_subj(mr):
    match = RE_VERB_SUBJ.match(mr)
    assert len(match.groups()) == 2
    return match.group(1), match.group(2)


def e2e_to_masked(input, output, filter_unsatisfied_constraints=False):
    split = RE_SPLIT.match(input.split("/")[-1]).group(1).removesuffix("set")
    print("Processing", split)

    with open(input, "r") as f:
        data = list(csv.DictReader(f))

    tokenizer = get_flant5_tokenizer()

    data_out = []
    for line_i, line in enumerate(tqdm.tqdm(data)):
        line = get_constraints(line, filter=filter_unsatisfied_constraints)

        input_sentence = f"Generate a sentence with the following topics [ {line['topic']} ]:"
        target_sentence_tok = tokenizer.tokenize(
            "<BOS> " + line["target_sentence"])
        input_sentence_tok = tokenizer.tokenize("<BOS> " + input_sentence)
        # none of these constraints have to be explicitly at the start of the sentence
        constraints_tok = [
            [tokenizer.tokenize(x) for x in constr] for constr in line["constraints"]]

        target_term_mask = compute_mask(target_sentence_tok, constraints_tok)
        input_term_mask = compute_mask(input_sentence_tok, constraints_tok)

        line_out = {
            "doc_id": f"{split}_{line_i}",
            "input_sentence": input_sentence_tok,
            "input_sentence_raw": input_sentence,
            "target_sentence": target_sentence_tok,
            "target_sentence_raw": line["target_sentence"],
            "input_term_mask": input_term_mask,
            "target_term_mask": target_term_mask,
            "target_constraints": constraints_tok,
        }
        data_out.append(line_out)

    with open(output, "w") as f:
        f.write("\n".join([json.dumps(x, ensure_ascii=False)
                for x in data_out]))


def get_constraints(line, filter):
    mrs = [get_verb_subj(mr) for mr in line["mr"].split(", ")]
    constrs = []

    topic_list = []

    for verb, subj in mrs:
        if verb in {"name", "eatType"}:
            topic_list.append(subj)
            # the name has to be there
            constrs.append([subj])
        elif verb in {"priceRange"}:
            topic_list.append(subj)
            if subj == "moderate":
                constrs.append([subj, "mid", "medium"])
            elif subj == "high":
                constrs.append([subj, "pricey"])
            else:
                constrs.append([subj])
        elif verb in {"area", "near"}:
            topic_list.append(subj)
            constrs.append([subj])
        elif verb in {"food"}:
            topic_list.append(subj)
            if subj == "Fast food":
                constrs.append([f"fast food"])
            else:
                constrs.append([subj])
        elif verb in {"customer rating"}:
            topic_list.append(subj)
            constrs.append([subj])
            if "1 out of" in subj:
                constrs[-1] += ["low", "one star"]
            elif "3 out of" in subj:
                constrs[-1] += ["three star"]
            elif "5 out of" in subj:
                constrs[-1] += ["high", "five star"]
        elif verb in {"familyFriendly"}:
            if subj == "no":
                pass
            else:
                topic_list.append(subj)
                original = [
                    "child friendly", "kids friendly", "kid friendly",
                    "family friendly", "children friendly"
                ]
                hyphenated = [x.replace(" ", "-") for x in original]
                constrs.append(original + hyphenated)
        else:
            print("Skipping", verb)
            continue

    assert len(topic_list) == len(constrs)

    if filter:
        tmp = [
            (c, t) for c, t in zip(constrs, topic_list)
            # only satisfied
            if any([x.lower() in line["ref"].lower() for x in c])
        ]
        constrs = [c for c, t in tmp]
        topic_list = [t for c, t in tmp]

    return {"target_sentence": line["ref"], "constraints": constrs, "topic": " ".join(topic_list), "mrs_parsed": mrs}
