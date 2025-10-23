from typing import List


def split_preprocessed_tokens(text: str) -> List[str]:
    """
    Split preprocessed text into tokens.
    Assumes text is already preprocessed and space-separated.
    """
    return [token.strip() for token in text.split() if token.strip()]
