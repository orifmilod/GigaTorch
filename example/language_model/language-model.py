import torch
from torch.nn.functional import cross_entropy

CONTEXT_SIZE = 3
EMBEDDING_SIZE = 2

def generate_mapping(data):
    data = sorted(list(set(''.join(data))))
    data = {char: index + 1 for index, char in enumerate(data)}
    # marks beginning or end of a word
    data['.'] = 0
    return data

def load_data():
    # How much tokens to keep as context when making the prediction for the next one

    x, y = [], []
    words = open('./names.txt').read().splitlines()
    stoi = generate_mapping(words)
    itos = {v: k for k, v in stoi.items()}

    for w in words[:1]:
       context = [0] * CONTEXT_SIZE
       print("word", w)
       for ch in w + '.':
           index_x = stoi[ch]
           print(''.join([itos[i] for i in context]), "with label: ", itos[index_x])
           x.append(context)
           y.append(index_x)

           context = context[1:] + [index_x]

    x = torch.tensor(x)
    y = torch.tensor(y)

    return x, y


def main():
    data, label = load_data()
    # look up table for the characters
    #Embedding our words in a 2d vector
    C = torch.rand((27, EMBEDDING_SIZE))
    print(f"{C=}")
    print(C.shape)
    print(f"{data=}")
    print(data.shape)

    # Creating an embedding from our data with each token being embedding represented by a vector of length "EMBEDDING_SIZE"
    embedding = C[data]
    # concatenated = torch.cat(torch.unbind(embedding, 1), 1) # Too slow, makes copy's of data
    # It happens that view'ing the tensor in 32, 6 is same as unbinding and concatenating it,
    # and view is a faster operation as it does not copy paste stuff. 
    print(f"{embedding=}")
    print(embedding.shape)
    embedding = embedding.view(embedding.shape[0], embedding.shape[1] * embedding.shape[2])
    print(f"{embedding=}")
    print(embedding.shape)

    #Creating hidden layer
    number_of_nuerons = 100
    hidden_layer_weights = torch.rand((CONTEXT_SIZE * EMBEDDING_SIZE, number_of_nuerons))
    hidden_layer_bias = torch.rand(100)
    print("hidden_layer_weights", hidden_layer_weights.shape)

    # output of the hidden layer
    h = torch.tanh(embedding @ hidden_layer_weights + hidden_layer_bias)

    # Creating the output layer
    output_layer_weights = torch.rand((number_of_nuerons, 27))
    output_layer_bias = torch.rand(27)
    logits = h @ output_layer_weights + output_layer_bias
    print(f"{logits=}")
    print(logits.shape)
    counts = logits.exp()
    print(f"{counts=}")
    print(counts.shape)
    probability = counts / counts.sum(1, keepdim=True)
    print(f"{probability=}")
    print(probability.shape)

    # Calculate the loss by calculating log mean of the correct labels; ideally they are all 1, meaning that we are correctly predicting the label
    # loss = -probability[torch.arange(embedding.shape[0]), label].log().mean()
    # print(loss)
    loss = cross_entropy(logits, label)
    print(loss)



if __name__ == "__main__":
    main()
