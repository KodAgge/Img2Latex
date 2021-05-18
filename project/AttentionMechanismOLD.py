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
        print("2D")
        for n in range(V.shape[3]):

            e_t = torch.transpose(self.beta(torch.tanh(self.W_h(h_t) + self.W(V_new[:, :, n]))), 0, 1)

            a_t = torch.softmax(e_t, dim = 1)

            c_t[:, n] = torch.matmul(a_t, V_new[:, :, n])


        print("3D")
        V = V.permute(3, 0, 1, 2)

        V_new = torch.reshape(V, (batch_size, H_prime * W_prime, C))

        U_T = self.W_h(h_t) + self.W(V_new)

        E_t = self.beta(torch.tanh(U_T))

        A_t = torch.transpose(torch.softmax(E_t, dim = 1), 1, 2)

        C_T = torch.matmul(A_t, V_new)

        print(c_t)
        
        
        return C_T

    def forward1D(self, V, h_t):
        H_prime, W_prime, C, batch_size = V.shape
        V_new = torch.reshape(V, (H_prime * W_prime, C, batch_size))
        c_t = torch.zeros(C, batch_size)
        x = self.W(V_new)
        print(x)
        E_t = self.beta(torch.tanh(self.W_h(h_t) + self.W(V_new)))

        print(E_t)

        for n in range(V.shape[3]):
            # e = torch.zeros(V_new.shape[0])

            e = torch.transpose(self.beta(torch.tanh(self.W_h(h_t) + self.W(V_new[:, :, n]))), 0, 1)

            a = torch.softmax(e, dim = 1)

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


class AttentionMechanism3D(nn.Module): 
    def __init__(self, beta_size, hidden_size, v_length):
        
        super().__init__()

        self.W_h = nn.Linear(hidden_size, beta_size)
        self.W = nn.Linear(v_length, beta_size)
        self.beta = nn.Linear(beta_size, 1)

    def forward(self, V, h_t):
        batch_size, H_prime, W_prime, C  = V.shape
        # print(V.shape)
        V_new = torch.reshape(V, (batch_size, H_prime * W_prime, C))
        # c_t = torch.zeros(C, batch_size)

        # print(self.W(V_new).shape)
        # print(self.W_h(h_t).shape)
        # print(torch.tanh(self.W_h(h_t) + self.W(V_new)))
        E_t = self.beta(torch.tanh(self.W_h(h_t) + self.W(V_new)))
        print()
        A_t = torch.transpose(torch.softmax(E_t, dim = 1), 1, 2)
        # print(A_t.shape)
        # print(V_new.shape)
        C_T = torch.matmul(A_t, V_new)
        # print(E_t.shape)


        return C_T


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
    H_prime = 3; W_prime = 3; C = 30; batch_size = 3; hidden_size=8
    V = torch.rand((H_prime, W_prime, C, batch_size)).float()
    h_t = torch.rand(hidden_size).float()
    # print(V)

    # V2 = V
    # V2 = V2.permute(3, 0, 1, 2)
    # print(V2)
    # V2 = torch.transpose(V, )
    # V2 = torch.rand((batch_size, H_prime, W_prime, C)).float()
    beta_size = 50; 
    # print(V.shape)
    # print(h_t.shape)
    model = AttentionMechanism(beta_size, hidden_size, v_length=C)
    model2 = AttentionMechanism3D(beta_size, hidden_size, v_length=C)
    # model = AttentionMechanism(beta_size, hidden_size, v_length=H_prime * W_prime)
    # print(model(V, h_t))
    # print(model2(V2, h_t))
    print(model(V, h_t))
    # model2(V2, h_t)


if __name__=='__main__':
    main()