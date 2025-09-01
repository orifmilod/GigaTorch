from collections import defaultdict
import json
import os

from pprint import pprint


class Tokenizer:
    """
    A class to tokenize text into subwords using a pre-trained Byte-Pair Encoding model.
    """

    def __init__(self):
        self.vocab = {index: bytes([index]) for index in range(256)}
        self.bpe_map = self.load_bpe_mapping()

        for (p0, p1), index in self.bpe_map.items():
            self.vocab[index] = self.vocab[p0] + self.vocab[p1]

    def load_bpe_mapping(self, file_path):
        map = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            map = json.load(f)

        new_map = {}
        for k, v in map.items():
            tokens = k.split('+')
            key = (int(tokens[0]), int(tokens[1]))
            new_map[key] = v

        return new_map

    def encode(self, text):
        tokens = list(text.encode('utf-8'))

        while len(tokens) > 1:
            pairs = self.get_pair_counter(tokens)
            if not pairs:
                break

            # find candidate pairs that exist in the BPE map
            candidate_pairs = [p for p in pairs.keys() if p in self.bpe_map]
            if not candidate_pairs:
                break

            # choose the pair with the lowest bpe id (earliest merge)
            pair = min(candidate_pairs, key=lambda p: self.bpe_map[p])
            tokens = Tokenizer.merge(tokens, pair, self.bpe_map[pair])

        return tokens

    def decode(self, tokens: list) -> str:
        tokens = b"".join(self.vocab[token] for token in tokens)
        return tokens.decode('utf-8', errors='replace')

    @staticmethod
    def generate_bpe_map(vocab_file: str, VOCABULARY_SIZE=1000) -> dict:
        all_tokens = []
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = list(line.encode('utf-8'))
                all_tokens.extend(tokens)

        bpe_map = Tokenizer.merge_common_pairs(all_tokens, VOCABULARY_SIZE)

        with open('result.json', 'w') as fp:
            json.dump(bpe_map, fp)

    @staticmethod
    def get_pair_counter(tokens):
        counter = defaultdict(int)
        for pair in zip(tokens, tokens[1:]):
            counter[pair] += 1

        return counter

    @staticmethod
    def merge_common_pairs(tokens, vocab_size):
        """
        Merge the most common pair of tokens until it reaches vocabulary size or
        there are no more pairs which occurs more than once to merge.

        return tokens and BPE map.
        """

        if not tokens:
            return

        new_id = 256  # Because maximum utf-8 codepoint ends in 255
        compression_steps_left = vocab_size - 256

        pair_counter = Tokenizer.get_pair_counter(tokens)
        pair_to_be_merged = max(pair_counter, key=pair_counter.get)

        bpe_map = {}

        while pair_counter[pair_to_be_merged] > 1 and compression_steps_left > 0:
            print(
                f"Merging pair: {pair_to_be_merged} with id: {new_id}. Occured: {pair_counter[pair_to_be_merged]}")

            # Merge it
            tokens = Tokenizer.merge(tokens, pair_to_be_merged, new_id)
            bpe_map[f"{pair_to_be_merged[0]}+{pair_to_be_merged[1]}"] = new_id

            # Recompute pair counter
            pair_counter = Tokenizer.get_pair_counter(tokens)
            pair_to_be_merged = max(pair_counter, key=pair_counter.get)

            new_id += 1
            compression_steps_left -= 1

        return bpe_map

    @staticmethod
    def merge(tokens, pair_to_be_merged, new_id):
        """
        Merge the most common pair of tokens.
        merge([1,2,6,1,2,9], (1,2), 99) => output: [99, 6, 99, 9]
        """
        new_ids = []
        i = 0
        length = len(tokens)
        while i < length:
            if i < length - 1 and (tokens[i], tokens[i + 1]) == pair_to_be_merged:
                new_ids.append(new_id)
                i += 2  # Skip the next token as it is merged
            else:
                new_ids.append(tokens[i])
                i += 1
        return new_ids
