from gigatorch.tokenizer import Tokenizer


def _make_minimal_tokenizer():
    """
    Create a Tokenizer instance without running __init__ and provide a
    minimal runtime-only vocab and empty bpe_map so tests don't depend on files.
    """
    tk = object.__new__(Tokenizer)
    tk.vocab = {i: bytes([i]) for i in range(256)}
    tk.bpe_map = {}
    return tk


def test_merge_appends_last_token():
    # The merge implementation should include the final leftover token.
    res = Tokenizer.merge([1, 2, 6, 1, 2, 9], (1, 2), 99)
    assert res == [99, 6, 99, 9]


def test_get_pair_counter_counts_pairs():
    # Verify pair counting on a short token sequence.
    tokens = [1, 2, 3, 2, 3]
    counter = Tokenizer.get_pair_counter(tokens)
    assert counter[(1, 2)] == 1
    assert counter[(2, 3)] == 2
    assert counter[(3, 2)] == 1


def test_merge_common_pairs_merges_most_frequent_pair():
    # For tokens [97,98,97,98] the pair (97,98) occurs twice and should be merged.
    tokens = [97, 98, 97, 98]  # bytes for 'a', 'b', 'a', 'b'
    bpe_map = Tokenizer.merge_common_pairs(tokens, vocab_size=258)
    assert "97+98" in bpe_map
    assert bpe_map["97+98"] == 256


def test_encode_merges_pairs_and_keeps_last_byte():
    # Use a minimal tokenizer to test encode/decode without requiring external files.
    tk = _make_minimal_tokenizer()

    # Make (ord('a'), ord('b')) -> 256 and ensure vocab for 256 exists
    pair = (ord('a'), ord('b'))
    tk.bpe_map = {pair: 256}
    tk.vocab[256] = tk.vocab[pair[0]] + tk.vocab[pair[1]]

    # "abxab" -> bytes: [97,98,120,97,98]
    # Expect merge of (97,98) -> 256 producing [256, 120, 256]
    encoded = tk.encode("abxab")
    assert encoded == [256, 120, 256]

    # Decode should reconstruct the original string
    decoded = tk.decode(encoded)
    assert decoded == "abxab"
