from gigatorch.embedding import Embedding, prepare_data, make_context_vector


def main():
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules called a program. 
    People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells."""
    data, word_to_index, index_to_word = prepare_data(raw_text)
    model = Embedding(100, 10)

    for target, context in data:
        context_vector = make_context_vector(context, word_to_index)
        print(model(context_vector))


if __name__ == "__main__":
    main()
