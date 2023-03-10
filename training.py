# Importing required libraries
import json # To work with Json files
from utils import tokenize, stem, b_o_w #Utils
import numpy as np #numpy
import torch #Pytorch
import torch.nn as nn #nn module of Pytorch
from torch.utils.data import Dataset, DataLoader 
from nn import NeuralNet

# Load the intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Prepare the data
xy = []    
all_words = []
tags = []
for x in intents['intents']:
    tag = x['tag']
    tags.append(tag)
    for pattern in x['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
ignored = ['?', '!', '.', ',', '/', ';', ':', '{', '}', '[', ']', '*', '^', "$", '#']
all_words = [stem(word) for word in all_words if word not in ignored]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

train_x = []
train_y = []
for (sentence, tag) in xy:
    bow = b_o_w(sentence, all_words)
    train_x.append(bow)
    label = tags.index(tag)
    train_y.append(label)    
train_x = np.array(train_x)
train_y = np.array(train_y)

# Create a chat dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(train_x)
        self.x_data = train_x
        self.y_data = train_y     
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 
    def __len__(self):
        return self.n_samples    

# Set the hyperparameters
batch_size = 20
hidden_size = 20
output_size = len(tags)
input_size = len(train_x[0])
learning_rate = 0.001
num_epochs = 1000

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a DataLoader object
dataset = ChatDataset()
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Create a neural network object
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch-{epoch+1}/{num_epochs}, loss = {loss.item():.4f}')   
print(f'final loss.loss={loss.item():.4f}')     

# Save the trained model
data = {"model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags}
FILE = "data.pth"
torch.save(data, FILE) 
print(f'training completed stored at {FILE}')
