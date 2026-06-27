import re

class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]) -> None:
        """Store token-to-id and id-to-token vocabulary mappings."""
        self.str_to_int = vocab
        self.int_to_str: dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Split text into tokens and convert each token to its integer id."""
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back to text and remove extra spaces before punctuation."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        """Store token-to-id and id-to-token vocabulary mappings."""
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        """Split text into tokens and convert each token to its integer id."""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """Convert token ids back to text and remove extra spaces before punctuation."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text