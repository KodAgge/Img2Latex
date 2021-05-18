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
        V_new = torch.reshape(V, (H_prime * W_prime, C, batch_size))
        print(V_new.shape)
        print(torch.all(torch.eq(V_new[H_prime+1,:,:], V[0,1,:,:]))) # check to see that it's correct

def main():
    H_prime = 10; W_prime = 20; C = 30; batch_size = 1; hidden_size=8
    V = torch.rand((H_prime, W_prime, C, batch_size))
    h_t = torch.rand(hidden_size)
    print(V.shape)
    print(h_t.shape)
    model = AttentionMechanism()
    model(V, h_t)


if __name__=='__main__':
    main()