import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# How much tokens to keep as context when making the prediction for the next one
CONTEXT_SIZE = 3
# Size of the vector to represent a single token
EMBEDDING_SIZE = 3
MINIBATCH_SIZE = 32
EPOCHS = 10000

def generate_mapping(data):
    data = sorted(list(set(''.join(data))))
    data = {char: index + 1 for index, char in enumerate(data)}
    # marks beginning or end of a word
    data['.'] = 0
    return data

def generate_learning_rates(size):
    lre = torch.linspace(-6, 0, size)
    return 10 ** lre # we want the learning rates to be spaced exponentially


def load_data():
    data, label = [], []
    words = open('./names.txt').read().splitlines()
    stoi = generate_mapping(words)
    itos = {v: k for k, v in stoi.items()}

    for w in words:
       context = [0] * CONTEXT_SIZE
       # print("word", w)
       for ch in w + '.':
           index_x = stoi[ch]
           # print(''.join([itos[i] for i in context]), "with label: ", itos[index_x])
           data.append(context)
           label.append(index_x)

           context = context[1:] + [index_x]

    data = torch.tensor(data)
    label = torch.tensor(label)
    return data, label, itos


def main():
    data, label, itos = load_data()
    # look up table for the characters
    # Embedding our words in a 2d vector
    C = torch.rand((27, EMBEDDING_SIZE))
    # print(f"{C=}")
    # print(C.shape)
    # print(f"{data=}")
    # print(data.shape)

    # Creating an embedding from our data with each token being embedding represented by a vector of length "EMBEDDING_SIZE"

    # concatenated = torch.cat(torch.unbind(embedding, 1), 1) # Too slow, makes copy's of data
    # It happens that view'ing the tensor in 32, 6 is same as unbinding and concatenating it,
    # and view is a faster operation as it does not copy paste stuff. 
    # print(f"{embedding=}")
    # print(embedding.shape)

    # print(f"{embedding=}")
    # print(embedding.shape)

    #Creating hidden layer
    number_of_nuerons = 200
    hidden_layer_weights = torch.rand((CONTEXT_SIZE * EMBEDDING_SIZE, number_of_nuerons))
    hidden_layer_bias = torch.rand(number_of_nuerons)


    # Creating the output layer
    output_layer_weights = torch.rand((number_of_nuerons, 27))
    output_layer_bias = torch.rand(27)


    # print(logits.shape)
    # print(f"{logits=}")

    # print(f"{counts=}")
    # print(counts.shape)

    # print(f"{probability=}")
    # print(probability.shape)

    # Calculate the loss by calculating log mean of the correct labels; ideally they are all 1, meaning that we are correctly predicting the label
    # loss = -probability[torch.arange(embedding.shape[0]), label].log().mean()
    # print(loss)

    parametes = [C, hidden_layer_bias, output_layer_weights, output_layer_bias]
    print("Number of parameters:", sum(p.nelement() for p in parametes))

    for p in parametes:
        p.requires_grad = True

    for p in parametes:
        p.grad = None


    # lrs = generate_learning_rates(EPOCHS)
    used_lrs = []
    losses = []

    print(data.shape)
    for i in range(EPOCHS):
        # Minibatching 
        minibatch_indexes = torch.randint(0, data.shape[0], (MINIBATCH_SIZE,))
        embedding = C[data[minibatch_indexes]]

        # Forward pass
        h = torch.tanh(embedding.view(-1, embedding.shape[1] * embedding.shape[2]) @ hidden_layer_weights + hidden_layer_bias)
        logits = h @ output_layer_weights + output_layer_bias

        # counts = logits.exp()
        # probability = counts / counts.sum(1, keepdim=True)
        loss = F.cross_entropy(logits, label[minibatch_indexes])
        loss.backward()
        # print(loss.item())

        used_lrs.append(i)
        losses.append(loss.item())

        for p in parametes:
            p.data += -0.001 * p.grad



    # Plot the characters
    plt.figure(figsize=(8,8))
    plt.scatter(C[:,0].data, C[:,1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
    plt.grid('minor')

    # plt.plot(used_lrs, losses)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
