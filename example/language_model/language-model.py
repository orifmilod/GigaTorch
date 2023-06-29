import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# How much tokens to keep as context when making the prediction for the next one
CONTEXT_SIZE = 3
# Size of the vector to represent a single token
EMBEDDING_SIZE = 10
MINIBATCH_SIZE = 32
EPOCHS = 100000

def generate_mapping(data):
    chars = sorted(list(set(''.join(data))))
    stoi = {char: index + 1 for index, char in enumerate(chars)}
    # marks beginning or end of a word
    stoi['.'] = 0
    return stoi

def generate_learning_rates(size):
    lre = torch.linspace(-6, 0, size)
    return 10 ** lre # we want the learning rates to be spaced exponentially

def load_data():
    data, label = [], []
    words = open('./names.txt', 'r').read().splitlines()
    stoi = generate_mapping(words)
    # itos = {v: k for k, v in stoi.items()}

    for w in words:
        context = [0] * CONTEXT_SIZE
        for ch in w + '.':
          ix = stoi[ch]
          data.append(context)
          label.append(ix)
          context = context[1:] + [ix] # crop and append

    data = torch.tensor(data)
    label = torch.tensor(label)
    return data, label

def main():
    data, label = load_data()
    # Creating an embedding from our data with each token being embedding represented 
    # by a vector of length "EMBEDDING_SIZE"
    C = torch.rand((27, EMBEDDING_SIZE))

    # Creating hidden layer
    number_of_nuerons = 200
    w1 = torch.rand((CONTEXT_SIZE * EMBEDDING_SIZE, number_of_nuerons)) * 0.01
    b1 = torch.rand(number_of_nuerons) * 0

    # Creating the output layer
    w2 = torch.rand((number_of_nuerons, 27)) * 0.01
    b2 = torch.rand(27) * 0

    parameters = [C, w1, b1, w2, b2]
    print("Number of parameters:", sum(p.nelement() for p in parameters))

    for p in parameters:
        p.requires_grad = True

    used_lrs = []
    losses = []
    avgs = []

    for i in range(EPOCHS):
        # Minibatching 
        minibatch_indexes = torch.randint(0, data.shape[0], (MINIBATCH_SIZE,))
        embedding = C[data[minibatch_indexes]]

        # Forward pass
        h = torch.tanh(embedding.view(-1, EMBEDDING_SIZE * CONTEXT_SIZE) @ w1 + b1)
        logits = h @ w2 + b2

        loss = F.cross_entropy(logits, label[minibatch_indexes])
        for p in parameters:
            p.grad = None

        loss.backward()

        # track stats
        if i % 1000 == 0: # print every once in a while
          print(f'{i:7d}/{EPOCHS:7d}: {loss.item():.4f}')
          if i > EPOCHS / 2:
              avgs.append(loss.item())

        used_lrs.append(i)
        losses.append(loss.item())

        lr = 0.1 if i < EPOCHS / 2 else 0.01 # step learning rate decay
        for p in parameters:
            p.data -= lr * p.grad


    print("Average loss", sum(avgs) / len(avgs))
    plt.plot(used_lrs, losses)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
