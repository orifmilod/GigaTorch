from gigatorch import Tensor
from gigatorch.weight_init import WightInitializer
import numpy as np


class Embedding:
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_size, self.embed_size = vocab_size, embed_size
        # What should be fan_in fan_out here?
        self.weight = WightInitializer.xavier_normal(1, 2, vocab_size, embed_size)

    def __call__(self, idx: Tensor) -> Tensor:
        return (self.vocab_counter == idx.unsqueeze(2)).expand(*idx.shape, self.vocab_size) @ self.weight


@staticmethod
def prepare_data(raw_text, context_size=2):
    raw_text = raw_text.split()
    vocab = set(raw_text)

    word_to_index = {word: ix for ix, word in enumerate(vocab)}
    index_to_word = {ix: word for ix, word in enumerate(vocab)}

    data = []
    for i in range(context_size, len(raw_text) - context_size):
        target = raw_text[i]
        context = raw_text[i - context_size : i + context_size + 1]
        data.append((target, context))

    return data, word_to_index, index_to_word


@staticmethod
def make_context_vector(context, word_to_index):
    indexes = [word_to_index[w] for w in context]
    return Tensor(indexes, dtype=np.long)
