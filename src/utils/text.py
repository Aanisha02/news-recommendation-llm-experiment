from typing import List, Union
from transformers import AutoTokenizer
import torch


def create_transform_fn_from_pretrained_tokenizer(
    pretrained_name: str,
    max_length: int,
):
    """
    Create a function that converts a string or list of strings into
    a torch.LongTensor of input_ids with shape [batch_size, max_length].
    """
    # This is a real Hugging Face tokenizer object, not a string
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def transform(texts: Union[str, List[str]]) -> torch.LongTensor:
        # Allow both a single string and a list of strings
        if isinstance(texts, str):
            texts = [texts]

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        # encoded["input_ids"]: [batch_size, max_length]
        return encoded["input_ids"]

    return transform
