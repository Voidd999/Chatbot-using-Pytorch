import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # define first linear layer
        self.l2 = nn.Linear(hidden_size, hidden_size) # define second linear layer
        self.l3 = nn.Linear(hidden_size, num_classes) # define third linear layer
        self.relu = nn.ReLU() # define activation function as ReLU
        
    def forward(self, x):
        out = self.l1(x) # input x to first linear layer
        out = self.relu(out) # apply ReLU activation function
        out = self.l2(out) # input output of first layer to second linear layer
        out = self.relu(out) # apply ReLU activation function
        out = self.l3(out) # input output of second layer to third linear layer
        return out



#This code defines a class NeuralNet that inherits from the nn.Module class of PyTorch. 
#The class defines a fully connected neural network with three linear layers, each followed by a ReLU activation function. 
#The constructor takes three arguments: input_size, hidden_size, and num_classes
#which represent the number of input features, the number of hidden units, and the number of output classes, respectively.
#The forward method takes an input tensor x and passes it through the three linear layers and ReLU activation functions, returning the output tensor.