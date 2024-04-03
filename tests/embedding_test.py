
def test_word_embedding():
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules called a program. 
    People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_index = {word: ix for ix, word in enumerate(vocab)}
    index_to_word = {ix: word for ix, word in enumerate(vocab)}
