import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class AttentionMechanism(nn.Module): 
    def __init__(self, beta_size, hidden_size, v_length):
        super().__init__()

        self.W_h = nn.Linear(hidden_size, beta_size)
        self.W = nn.Linear(v_length, beta_size)
        self.beta = nn.Parameter(torch.zeros(beta_size))

    def forward(self, V, h_t):
        H_prime, W_prime, C, batch_size = V.shape
        V_new = torch.reshape(V, (H_prime * W_prime, C, batch_size))
        print(V_new.shape)
        print(torch.all(torch.eq(V_new[2*W_prime,:,:], V[2,0,:,:]))) # check to see that it's correct

def main():
    H_prime = 10; W_prime = 20; C = 30; batch_size = 3; hidden_size=8
    V = torch.rand((H_prime, W_prime, C, batch_size))
    h_t = torch.rand(hidden_size)
    print(V.shape)
    #print(h_t.shape)
    model = AttentionMechanism()
    model(V, h_t)


if __name__=='__main__':
    main()