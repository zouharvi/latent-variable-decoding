BOS_TOKEN = "<BOS>"

def get_flant5_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokenizer.add_tokens(BOS_TOKEN)
    return tokenizer

BOS_CONSTRAINTS = [
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how"
]

def tokenizer_none_wrap(tokenizer, sequence):
    sequence = tokenizer.tokenize(sequence)
    sequence = [x if x != "▁None" else None for x in sequence]
    return sequence


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
