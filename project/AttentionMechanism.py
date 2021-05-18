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
        self.W_h = nn.Linear(hidden_size, beta_size, bias=False)

        # Weights for the encoded image V
        self.W = nn.Linear(v_length, beta_size, bias=False)

        # To sum up after activation function (tanh)
        self.beta = nn.Linear(beta_size, 1, bias=False)

    def forward(self, V, h_t):
        # Change dimensions of the input vector
        V = V.permute(0, 2, 3, 1)

        # Find dimensions
        batch_size, H_prime, W_prime, C = V.shape

        # Reshape to batch_size x H' * W' x C
        V_new = torch.reshape(V, (batch_size, H_prime * W_prime, C))

        # Matrix multiplocation
        U_t = self.W_h(h_t) + self.W(V_new)

        # Activation function and weighted summing
        E_t = self.beta(torch.tanh(U_t))

        # Applying softmax
        A_t = torch.transpose(torch.softmax(E_t, dim = 1), 1, 2)

        # Final weighted summing
        C_t = torch.matmul(A_t, V_new)

        return C_t


def main():
    H_prime = 3; W_prime = 3; C = 30; batch_size = 3; hidden_size=8
    h_t = torch.rand(hidden_size).float()

    V = torch.rand((batch_size, C, H_prime, W_prime)).float()

    # ordinary_dimensions = True

    # if ordinary_dimensions:
    #     # The dimensions this program needs
    #     V = torch.rand((batch_size, H_prime, W_prime, C)).float()
    # else:
    #     # The dimensions V will come in from the CNN
    #     V = torch.rand((batch_size, C, H_prime, W_prime)).float()

    #     # Changing the dimensions to the ones the program need
    #     V = V.permute(0, 2, 3, 1)

    beta_size = 10; 

    model = AttentionMechanism(beta_size, hidden_size, v_length=C)

    context = model(V, h_t)



if __name__=='__main__':
    main()