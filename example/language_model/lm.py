import torch
import torch.nn as nn
import numpy as np

# Define the characters in the vocabulary
text = open('./names.txt').read().splitlines()
characters = sorted(list(set(text)))  # 'text' is the training data
sequence_length = 10
learning_rate = 0.1
num_epochs = 1000
max_length = 10

# Create character-to-index and index-to-character mappings
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for idx, char in enumerate(characters)}

# Convert the text into sequences of indices
sequences = []
sequence_length = 3
for i in range(len(text) - sequence_length):
    sequence = text[i:i+sequence_length]
    target = text[i+sequence_length]
    sequences.append((sequence, target))

# Generate input and target data
X = np.zeros((len(sequences), sequence_length, len(characters)), dtype=np.float32)
y = np.zeros((len(sequences), len(characters)), dtype=np.float32)
for i, (sequence, target) in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_idx[char]] = 1.0
    y[i, char_to_idx[target]] = 1.0

# Convert numpy arrays to PyTorch tensors
X = torch.from_numpy(X)
y = torch.from_numpy(y)

# Define the model architecture
class CharFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharFFN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = sequence_length * len(characters)
hidden_size = 128
output_size = len(characters)
model = CharFFN(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, torch.argmax(y, dim=1))

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1):
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate sample text using the trained model
start_sequence = 'The '
generated_text = start_sequence
with torch.no_grad():
    input_seq = torch.from_numpy(np.array([[char_to_idx[ch] for ch in start_sequence]]))
    for _ in range(max_length):
        output = model(input_seq)
        predicted_idx = torch.argmax(output, dim=1)
        generated_text += idx_to_char[predicted_idx.item()]
        input_seq = torch.cat((input_seq[:, 1:], predicted_idx.unsqueeze(0)), dim=1)

print(generated_text)
