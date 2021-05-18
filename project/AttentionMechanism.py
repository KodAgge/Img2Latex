import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

class AttentionMechanism(nn.Module): 
    def __init__(self, beta_size, hidden_size, v_length):
        super().__init__()

        # Weights for the hidden layer h
        self.W_h = nn.Linear(hidden_size, beta_size)

        # Weights for the encoded image V
        self.W = nn.Linear(v_length, beta_size)

        # To sum up after activation function (tanh)
        self.beta = nn.Linear(beta_size, 1)

    def forward(self, V, h_t):
        # Find dimensions
        batch_size, H_prime, W_prime, C = V.shape

        # Reshape to batch_size x H' * W' x C
        V_new = torch.reshape(V, (batch_size, H_prime * W_prime, C))

        # Matrix multiplocation
        U_T = self.W_h(h_t) + self.W(V_new)

        # Activation function and weighted summing
        E_t = self.beta(torch.tanh(U_T))

        # Applying softmax
        A_t = torch.transpose(torch.softmax(E_t, dim = 1), 1, 2)

        # Final weighted summing
        C_T = torch.matmul(A_t, V_new)
        
        return C_T


def main():
    H_prime = 3; W_prime = 3; C = 30; batch_size = 3; hidden_size=8
    h_t = torch.rand(hidden_size).float()

    V = torch.rand((batch_size, H_prime, W_prime, C)).float()
    beta_size = 10; 

    model = AttentionMechanism(beta_size, hidden_size, v_length=C)

    context = model(V, h_t)



if __name__=='__main__':
    main()