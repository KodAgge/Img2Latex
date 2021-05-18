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
        self.CNN = None
        self.LSTM_module = paper_LSTM_Module(input_size, hidden_size, batch_size)
        self.attentionMechanism = None

    def init_parameters(self):
        pass

    def forward(self, X_batch): 


        pass






def main():
    train_set = CROHME_Training_Set()
    image = train_set[0]['image']
    label = train_set[0]['label']
    #plt.imshow(image.permute(1, 2, 0), cmap='gray')
    #plt.show()
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    
    n1 = 80; n2 = 100; # n1 = antalet rader i E-matrisen, n2 = storleken på o-vektorn
    hidden_size = 512; batch_size = 20;
    num_epochs = 1
    model = EncoderDecoder(input_size=n1+n2, hidden_size=hidden_size, batch_size=batch_size)

    #train_samples = train_set[0:10]
    #X = train_samples.keys()
    #print(X)

    n_total_steps = len(train_loader)
    #print(n_total_steps)
    #print(len(train_set))
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            print(images)
            print(labels)
            input('hej')


if __name__=='__main__':
    main()
    
