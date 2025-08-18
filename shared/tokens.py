import tiktoken


def count_tokens(string):
    encoder = tiktoken.get_encoding("cl100k_base")  # gpt-4 tokenizer
    tokens = encoder.encode(string)
    return len(tokens)


def count_tokens_list(source_list):
    num_tokens = 0

    for source in source_list:
        num_tokens += count_tokens(source)

    return num_tokens
