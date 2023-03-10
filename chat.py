# Importing required libraries
import random #Random
import json  #To work with Json files
import torch #PyTorch for our NN
from nn import NeuralNet #our model
from utils import b_o_w, tokenize, checkSpelling, correct #Utils

# Set device to CUDA if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load pre-trained model data from file
FILE = 'data.pth'
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize model and load pre-trained weights
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Set Chatbot name
name = 'Void'

# Define function to generate response to user input
def get_response(msg):
    # Tokenize user input
    sentence = tokenize(msg)
    # Convert tokenized sentence to bag of words vector
    X = b_o_w(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get output from model and determine predicted tag
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Determine probability of predicted tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If probability is high enough, return a random response from the corresponding intent
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    # Otherwise, return an "I do not understand" response
    else:
        return "I do not understand..."

# Main loop for chatbot interaction
if __name__ == "__main__":
    print("What can I help you with?\nType 'quit' to exit.")
    while True:
        sentence = input('You: ')
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(f'{name}: {resp}')
