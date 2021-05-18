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
        self.AttentionMechanism = AttentionMechanism(beta_size=50, hidden_size=hidden_size, v_length=v_length) # TODO: Change these hard-coded values

        # The other layers
        self.E = nn.Parameter(torch.zeros(embedding_size, vocab_size))
        self.O = nn.Linear(v_length + hidden_size, o_layer_size, bias=False)  # TODO: ADD BIAS?
        self.W_out = nn.Linear(o_layer_size, vocab_size, bias=False)  # TODO: ADD BIAS?
        self.softmax = nn.Softmax(0)


    def init_parameters(self):
        """Function to initialize parameters that are NOT initialized in the modules (which should take care of themselves"""
        pass

    def forward(self, X_batch): 
        # 1) CNN & "Cube Creation"
        #x = self.CNN(X_batch)
        #print(X_batch.shape)
        V = self.CNN(X_batch)
        #input(V.shape)

        # 2) LSTM 

        # Initialize Y and O 
        Y_0 = torch.zeros(self.vocab_size, self.batch_size)
        Y_0[141,:] = 1
        O_0 = torch.zeros(self.o_layer_size, self.batch_size)
        X_1 = torch.cat((self.E @ Y_0, O_0), 0)

        for i in range(self.sequence_length):
            H_t = self.LSTM_module(X_1) 
            #C_t = self.AttentionMechanism(V, H_t)
            C_t = torch.ones (self.v_length, self.batch_size)
            concat = torch.cat((H_t, C_t), 0)
            concat = torch.transpose(concat, 0, 1)
            O_t = torch.tanh(self.O(concat))
            print(O_t.shape)
            Y_distr = self.softmax(W_out(O_t))
            print(Y_distr.shape)


            input('Attention')
            

        # Attention

        # The Rest

        return 0


def MGD(net, train_dataloader, learning_rate, momentum, n_epochs):
    criterion = nn.CrossEntropyLoss() # Ändra denna?
    
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [images, labels]
            images, labels = data["image"], data["label"]
            
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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

    batch_size = 20

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    ED = EncoderDecoder(embedding_size=embedding_size, hidden_size=hidden_size, batch_size=batch_size, sequence_length=sequence_length, vocab_size=vocab_size, o_layer_size = o_layer_size)

    ED_Trained = MGD(ED, train_loader, learning_rate=0.001, momentum=0.9, n_epochs=10)


    """ n_batches = len(train_loader)
    print(n_batches)
    print(len(train_set))
    for epoch in range(num_epochs):
        for (i, batch) in enumerate(train_loader):
            images = batch['image'] # [batch_size, height, width]
            labels = batch['label']
            #print(images)
            #print(labels)
            print(images.shape)
            print(len(labels))

            # Forward-pass


            # Backward-pass and gradient descent


            input('---BATCH IS OVER---') """

if __name__=='__main__':
    main()
    
