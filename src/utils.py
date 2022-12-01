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

