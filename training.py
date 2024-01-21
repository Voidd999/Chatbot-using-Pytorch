# Importing required libraries
import json
from utils import tokenize, stem, b_o_w
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn import NeuralNet
import streamlit as st


def train(intents, batch_size, hidden_size, learning_rate, num_epochs):
    xy = []
    all_words = []
    tags = []
    for x in intents["intents"]:
        tag = x["tag"]
        tags.append(tag)
        for pattern in x["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))
    ignored = ["!",".",",","/",";",":","{","}","[","]","*","^","$","#","@","&","-","_"]
    all_words = [stem(word) for word in all_words if word not in ignored]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    train_x = []
    train_y = []
    for sentence, tag in xy:
        bow = b_o_w(sentence, all_words)
        train_x.append(bow)
        label = tags.index(tag)
        train_y.append(label)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # chat dataset class
    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(train_x)
            self.x_data = train_x
            self.y_data = train_y

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    # hyperparameters
    output_size = len(tags)
    input_size = len(all_words)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChatDataset()
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    #  neural network object
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bar = st.progress(0, text=f"Training with {num_epochs} epochs...")
    # Train 
    for epoch in range(num_epochs):
        for words, labels in loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            st.caption(f"epoch-{epoch+1}/{num_epochs}, loss = {loss.item():.4f}")
            bar.progress(
                (epoch / num_epochs),
                text=f"{str(epoch+1)}/{str(num_epochs)} iterations done... loss={loss.item():.4f}",
            )

    st.subheader(f"final loss.loss={loss.item():.4f}")
    bar.progress(100, text=f"Done..final loss = {loss.item():.4f}")

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags,
    }
    return data
