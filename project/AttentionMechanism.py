import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

class AttentionMechanism(nn.Module): 
    def __init__(self, beta_size, hidden_size, v_length):
        
        super().__init__()

        self.W_h = nn.Linear(hidden_size, beta_size)
        self.W = nn.Linear(v_length, beta_size)
        # self.beta = nn.Parameter(torch.zeros(beta_size))
        self.beta = nn.Linear(beta_size, 1)

    def forward(self, V, h_t):
        H_prime, W_prime, C, batch_size = V.shape
        V_new = torch.reshape(V, (H_prime * W_prime, C, batch_size))
        c_t = torch.zeros(C, batch_size)

        for n in range(V.shape[3]):
            e = torch.zeros(V_new.shape[0])

            for i in range(V_new.shape[0]):
                e[i] = self.beta(torch.tanh(self.W_h(h_t) + self.W(V_new[i, :, n])))

            a = torch.softmax(e, dim = 0)

            c_t[:, n] = torch.matmul(a, V_new[:, :, n])

        return c_t


    def forward_3D_WIP(self, V, h_t):
        H_prime, W_prime, C, batch_size = V.shape
        V_new = torch.reshape(V, (H_prime * W_prime, C, batch_size))
        c_t = torch.zeros(C, batch_size)
        A = torch.zeros(H_prime * W_prime, batch_size)
        for n in range(V.shape[3]):
            e = torch.zeros(V_new.shape[0])

            for i in range(h_t.shape[0]):
                e[i] = self.beta(torch.tanh(self.W_h(h_t) + self.W(V_new[i, :, n])))

            # a = np.exp(e)/sum(np.exp(e))
            # a = torch.softmax(e)
            a = e
            A[:, n] = a
            c_t[:, n] = torch.matmul(a, V_new[:, :, n])
            # print(c_t.shape)
            
        print(A.shape)
        print(V_new.shape)
        c_t_2 = torch.matmul(A, V_new)
        print(c_t.shape)
        print(c_t_2.shape)
            # print(V_new.shape)
            # print(torch.all(torch.eq(V_new[2*W_prime,:,:], V[2,0,:,:]))) # check to see that it's correct

def main():
    H_prime = 10; W_prime = 20; C = 30; batch_size = 3; hidden_size=8
    V = torch.rand((H_prime, W_prime, C, batch_size)).float()
    h_t = torch.rand(hidden_size).float()


    beta_size = 50; 
    print(V.shape)
    print(h_t.shape)
    model = AttentionMechanism(beta_size, hidden_size, v_length=C)
    # model = AttentionMechanism(beta_size, hidden_size, v_length=H_prime * W_prime)
    print(model(V, h_t))


if __name__=='__main__':
    main()