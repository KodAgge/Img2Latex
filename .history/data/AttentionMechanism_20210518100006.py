import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class AttentionMechanism(nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self, V, h_t):
        H_prime, W_prime, C, batch_size = V.shape
        print(V[0,0,:,:])
        V = torch.reshape(V, (H_prime * W_prime, C, batch_size))
        print(V[0,:,:])
        print(torch.eq(V[0,:,:], V[0,0,:,:]))

def main():
    H_prime = 10; W_prime = 20; C = 30; batch_size = 5; hidden_size=8
    V = torch.rand((H_prime, W_prime, C, batch_size))
    h_t = torch.rand(hidden_size)
    print(V.shape)
    print(h_t.shape)
    model = AttentionMechanism()
    model(V, h_t)


if __name__=='__main__':
    main()