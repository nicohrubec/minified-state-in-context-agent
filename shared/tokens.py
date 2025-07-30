import tiktoken


def count_tokens(string):
    encoder = tiktoken.get_encoding("cl100k_base")  # gpt-4 tokenizer
    tokens = encoder.encode(string)
    return len(tokens)
