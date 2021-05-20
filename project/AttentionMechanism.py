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
        self.W_h = nn.Linear(hidden_size, beta_size, bias=False).double()

        # Weights for the encoded image V
        self.W = nn.Linear(v_length, beta_size, bias=False).double()

        # To sum up after activation function (tanh)
        # self.beta = nn.Linear(beta_size, 1, bias=False).double()
        self.beta = nn.Parameter(torch.Tensor(beta_size))
        nn.init.uniform_(self.beta, -1e-2, 1e-2)

        self.init_weights()

    def init_weights(self): 
        # Initialisez the wights with Xavier Normalization
        torch.nn.init.xavier_normal_(self.W_h.weight)
        torch.nn.init.xavier_normal_(self.W.weight)

    def forward(self, V_new, h_t):
        # Matrix multiplocation
        U_t = torch.tanh(self.W_h(h_t).unsqueeze(1) + self.W(V_new)) # [B, H' * W', C]
        
        # Activation function and weighted summing
        E_t = torch.sum(self.beta * U_t, dim=-1) # [B, H' * W']
        
        # Applying softmax
        A_t = torch.softmax(E_t, dim = 1).unsqueeze(1) # [B, 1, H' * W']
  

        # Final weighted summing
        C_t = torch.matmul(A_t, V_new).squeeze()

        C_t = torch.transpose(C_t, 0, 1) # [C, B]

        return C_t, A_t


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