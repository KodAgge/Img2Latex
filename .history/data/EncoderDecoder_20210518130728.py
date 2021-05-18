import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt 

from paper_LSTM import paper_LSTM_Module

import sys
sys.path.insert(0, '..\data')
from CROHME_Datasets import CROHME_Training_Set



class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()


        # Network Architecture
        self.CNN = None
        self.LSTM_module = paper_LSTM_Module(input_size, hidden_size, batch_size)
        self.AttentionMechanism = None

        # ... rest of layers

    def init_parameters(self):
        """Initializes parameters that are not initialized in the modules"""
        pass

    def forward(self, X_batch): 
        # CNN & "Cube Creation"

        # LSTM f

        pass




def main():
    train_set = CROHME_Training_Set()

    #image = train_set[0]['image']
    #label = train_set[0]['label']
    #plt.imshow(image.permute(1, 2, 0), cmap='gray')
    #plt.show()
    
    embedding_size = 80; # number of rows in the E-matrix
    o_size = 100;  # size of o-vektorn
    input_size = embedding_size + o_size
    hidden_size = 512; 

    batch_size = 20; num_epochs = 1

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = EncoderDecoder(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)


    n_batches = len(train_loader)
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


            input('---BATCH IS OVER---')

if __name__=='__main__':
    main()
    
