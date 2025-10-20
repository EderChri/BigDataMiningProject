from typing import List


def split_preprocessed_tokens(text: str) -> List[str]:
    """
    Split a preprocessed message (as produced by the DataLoader's preprocessing)
    into tokens. Assumes the input is already lowercased, lemmatized, and cleaned,
    with tokens separated by whitespace.
    """
    if not text:
        return []
    return [t for t in text.split() if t]
