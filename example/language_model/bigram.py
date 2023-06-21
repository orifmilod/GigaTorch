from math import dist
from numpy import float32, int32
from gigatorch.tensor import Tensor
import torch
from torch.nn.functional import one_hot

def generate_mapping(data):
    data = sorted(list(set(''.join(data))))
    data = {char: index for index, char in enumerate(data)}
    # marks beginning or end of a word
    data['.'] = 26
    return data

def main():
    words = open('./names.txt').read().splitlines()
    counter = {}
    stoi = generate_mapping(words)
    value = Tensor.zeros((28,28), dtype=int32)

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            # bigram = (ch1, ch2)
            # counter[bigram] = counter.get(bigram, 0) + 1

            index1 = stoi[ch1]
            index2 = stoi[ch2]
            value[index1, index2] += 1


    value.sum()
    print(value.to(float32))
    print(value[1, :])

    counter = sorted(counter.items(), key = lambda kv: -kv[1])

def sample_word(distribution, itos):
    index = 0
    word = ''
    while True:
        index = torch.multinomial(distribution[index], num_samples=1, replacement=True).item()
        # print("Sampled index", index, itos[index])
        word += itos[index]
        if(index == 0):
            break
    return word


def pytorch_version():
    words = open('names.txt', 'r').read().splitlines()

    probabilities = torch.zeros((27, 27), dtype=torch.int32)
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {v: k for k, v in stoi.items()}

    for w in words:
      chs = ['.'] + list(w) + ['.']
      for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        probabilities[ix1, ix2] += 1


    # Converting N so each row is a probablity distribution
    probabilities = probabilities.float()
    probabilities /= probabilities.sum(1, keepdim=True)
    print(probabilities)

    words = []
    for _ in range(10):
        word = sample_word(probabilities, itos)
        words.append(word)

    print(words)

def Neural_net():
    words = open('names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {v: k for k, v in stoi.items()}

    xs, ys = [], []

    for w in words:
      chs = ['.'] + list(w) + ['.']
      for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        xs.append(ix1)
        ys.append(ix1)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    print(xs)

    x_encoded = one_hot(xs, num_classes=27)
    y_encoded = one_hot(ys, num_classes=27)
    print(x_encoded)


if __name__ == "__main__":
    Neural_net()
    # pytorch_version()
    # main()
