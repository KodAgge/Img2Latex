import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt

from CNN import Net as CNN
from paper_LSTM import paper_LSTM_Module
from AttentionMechanism import AttentionMechanism

import sys
sys.path.insert(0, '..\data')
from CROHME_Datasets import CROHME_Training_Set



class EncoderDecoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, batch_size, sequence_length, vocab_size, o_layer_size, v_length=512):
        super().__init__()

        # Static params
        self.v_length = v_length
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.o_layer_size = o_layer_size

        input_size = embedding_size + o_layer_size

        # Network Modules
        self.CNN = CNN()
        self.LSTM_module = paper_LSTM_Module(input_size, hidden_size, batch_size)
        self.AttentionMechanism = AttentionMechanism(beta_size=512, hidden_size=hidden_size, v_length=v_length) # TODO: Change these hard-coded values

        # The other layers
        self.E = nn.Parameter(torch.zeros(embedding_size, vocab_size)).double()
        self.O = nn.Linear(v_length + hidden_size, o_layer_size, bias=False).double()  # TODO: ADD BIAS?
        self.W_out = nn.Linear(o_layer_size, vocab_size, bias=False).double()  # TODO: ADD BIAS?
        self.softmax = nn.Softmax(1).double()


    def init_parameters(self):
        """Function to initialize parameters that are NOT initialized in the modules (which should take care of themselves"""
        pass

    def forward(self, X_batch): 
        # 1) CNN, aka "HyperCube Creation" :) 
        #x = self.CNN(X_batch)
        V = self.CNN(X_batch)

        # Initialize Y and O 
        # Y_pred = torch.zeros((self.batch_size, self.sequence_length)).int()
        output = torch.zeros(self.batch_size, self.sequence_length, self.vocab_size).double()

        Y_0 = torch.zeros(self.vocab_size, self.batch_size).double()
        Y_0[141,:] = 1
        O_0 = torch.zeros(self.o_layer_size, self.batch_size).double()
        X_t = torch.cat((self.E @ Y_0, O_0), 0)

        
        # output = torch.zeros(self.sequence_length, self.vocab_size, self.batch_size)   
        # print(X_t.shape)
        for i in range(self.sequence_length):
            print(i)
            H_t = self.LSTM_module(X_t)         # 2) LSTM 

            # 3) Attention Mechanism
            C_t, A_t = self.AttentionMechanism(V, torch.transpose(H_t, 0, 1))  
            # C_t = torch.ones (self.v_length, self.batch_size)

            concat = torch.cat((H_t, C_t), 0)
            concat = torch.transpose(concat, 0, 1)
            linear_O = self.O(concat)
            O_t = torch.tanh(linear_O)
            A_t = self.W_out(O_t) # This is the wanted output for the cross-entropy, that is un-softmaxed probabilities

            output[:, i, :] = A_t
            Y_distr = self.softmax(A_t)
            
            # Greedy approach
            max_indices = torch.argmax(Y_distr, dim=1)
            Y_onehot = torch.zeros(self.vocab_size, self.batch_size).double()
            Y_onehot[max_indices, :] = 1
            # Y_pred[:, i] = max_indices + 1
            #print(Y_onehot[int(max_indices[0])-2:int(max_indices[0])+2,:])
            O_t = torch.transpose(O_t, 0, 1)
            X_t = torch.cat((self.E @ Y_onehot, O_t), 0)

            # Store output distribution (OR SHOULD WE STORE THE GREEDY?)   
            # output[i,:,:] = Y_onehot

        # return Y_pred
        # return output # Y_s -> [seq_length, vocab_size, batch_size]
        return output


def MGD(net, train_dataloader, learning_rate, momentum, n_epochs):
    criterion = nn.CrossEntropyLoss() # Ändra denna?
    
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [images, labels]
            images, labels = data["image"], data["label"] - 1 # Labels måste börja på 0
            
            # Can use to decrease learning rate
            # for g in optimizer.param_groups:
            #     g['lr'] /= 2

            #print(images.shape)
            #print(labels)
            
            # forward-pass
            outputs = net(images)

            input('---Klar med FORWARD PASSET---')

            
            # backwards pass + gradient step
            optimizer.zero_grad() # zero the parameter gradients
            loss = criterion(outputs.view(-1, 144), labels.view(-1))
            print(loss)
            loss.backward(retain_graph=True) # Fullösning för att den verkar behöva förra iterationen
            # Kolla här: https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795
            
            # loss.backward()
            optimizer.step()

            input('---Klar med BACKWARD PASSET---')

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    return net



def main():
    train_set = CROHME_Training_Set()

    #image = train_set[0]['image']
    #label = train_set[0]['label']
    #plt.imshow(image.permute(1, 2, 0), cmap='gray')
    #plt.show()
    
    embedding_size = 80; # number of rows in the E-matrix
    o_layer_size = 100;  # size of o-vektorn TODO: What should this be?
    hidden_size = 512; 
    sequence_length = 110; vocab_size = 144; 

    batch_size = 3

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    ED = EncoderDecoder(embedding_size=embedding_size, hidden_size=hidden_size, batch_size=batch_size, sequence_length=sequence_length, vocab_size=vocab_size, o_layer_size = o_layer_size)

    ED_Trained = MGD(ED, train_loader, learning_rate=0.001, momentum=0.9, n_epochs=10)



if __name__=='__main__':
    main()
    
