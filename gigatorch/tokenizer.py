from collections import defaultdict, Counter
import json
import regex as re


class Tokenizer:
    """
    A class to tokenize text into subwords using a pre-trained Byte-Pair Encoding model.
    """

    def __init__(self):
        self.vocab = {index: bytes([index]) for index in range(256)}
        self.bpe_map = self.load_bpe_mapping()
        # ignore the casing

        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE)

        for (p0, p1), index in self.bpe_map.items():
            self.vocab[index] = self.vocab[p0] + self.vocab[p1]

    ################ INFERENCE PART ##########################
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

    ################## TRAINING PART ####################
    @staticmethod
    def train(vocab_file: str, output_file: str, VOCABULARY_SIZE=50256) -> dict:
        """
        Generate a BPE mapping from a vocabulary file and write it to output_file.
        This implementation reads the input once (binary) and uses efficient
        Counter-based pair counting for faster merges.
        """
        # Read file as binary once (faster than per-line encode)
        with open(vocab_file, 'rb') as f:
            data = f.read()
        all_tokens = list(data)

        bpe_map = Tokenizer.merge_common_pairs(all_tokens, VOCABULARY_SIZE)

        with open(output_file, 'w') as fp:
            json.dump(bpe_map, fp)

        return bpe_map

    @staticmethod
    def get_pair_counter(tokens):
        return Counter(zip(tokens, tokens[1:]))

    @staticmethod
    def merge_common_pairs(tokens, vocab_size):
        """
        Merge the most common pair of tokens until it reaches vocabulary size or
        there are no more pairs which occurs more than once to merge.

        Uses Counter.most_common (C-optimized) and a single-pass merge per step
        to improve performance vs. manual Python loops.
        """
        if not tokens:
            return {}

        new_id = 256  # Because maximum utf-8 codepoint ends in 255
        compression_steps_left = vocab_size - 256

        bpe_map = {}

        # Initialize pair counter efficiently
        pair_counter = Tokenizer.get_pair_counter(tokens)

        counter = 0
        while compression_steps_left > 0 and pair_counter:
            pair_to_be_merged, freq = pair_counter.most_common(1)[0]
            if freq <= 1:
                break

            if counter % 100 == 0:
                print("Step left", compression_steps_left, freq)

            counter += 1

            # Single-pass merge to produce the new token list
            new_tokens = []
            i = 0
            length = len(tokens)
            while i < length:
                if i < length - 1 and (tokens[i], tokens[i + 1]) == pair_to_be_merged:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            bpe_map[f"{pair_to_be_merged[0]}+{pair_to_be_merged[1]}"] = new_id

            # Prepare for next iteration
            tokens = new_tokens
            pair_counter = Tokenizer.get_pair_counter(tokens)

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
