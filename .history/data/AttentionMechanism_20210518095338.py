import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class AttentionMechanism(nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self, V, h_t):
        
        pass


def main():
    H_prime = 10; W_prime = 20; C = 30; batch_size = 5; hidden_size=8
    V = torch.ones((H_prime, W_prime, C, batch_size))
    h_t = torch.ones(hidden_size)
    print(V.shape)
    print(h_t.shape)
    model = AttentionMechanism()
    model(V, h_t)


if __name__=='__main__':
    main()