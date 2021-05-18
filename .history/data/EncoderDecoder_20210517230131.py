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
        self.attentionMechanism = None

    def init_parameters(self):
        """Initializes parameters that are not initialized in the modules"""
        pass

    def forward(self, X_batch): 


        pass




def main():
    train_set = CROHME_Training_Set()

    #image = train_set[0]['image']
    #label = train_set[0]['label']
    #plt.imshow(image.permute(1, 2, 0), cmap='gray')
    #plt.show()
    
    n1 = 80; n2 = 100; # n1 = antalet rader i E-matrisen, n2 = storleken på o-vektorn
    hidden_size = 512; batch_size = 20;
    num_epochs = 1

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = EncoderDecoder(input_size=n1+n2, hidden_size=hidden_size, batch_size=batch_size)


    n_total_steps = len(train_loader)
    print(n_total_steps)
    print(len(train_set))
    for epoch in range(num_epochs):
        for (i, batch) in enumerate(train_loader):
            images = batch['image'] # [batch_size, height, width]
            labels = batch['label']
            #print(images)
            #print(labels)
            print(images.shape)
            print(len(labels))



            input('---BATCH IS OVER---')

if __name__=='__main__':
    main()
    
