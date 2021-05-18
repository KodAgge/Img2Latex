import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class AttentionMechanism(nn.Module): 
    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()

    def forward(self, V, h_t):
        pass


def main():
    H_prime = 10, W_prime = 20, C = 30
    V = torch.ones((H_prime, W_prime, C)
    print(V)
    

if __name__=='__main__':
    main()