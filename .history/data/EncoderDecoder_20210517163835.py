import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt 

import sys
sys.path.insert(0, '..\data')
from CROHME_Datasets import CROHME_Training_Set



class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

    def init_parameters(self):
        pass

    def forward(self, X_batch): 
        pass

def main():
    train_set = CROHME_Training_Set()
    image = train_set[0]['image']
    label = train_set[0]['label']
    plt.imshow(image.permute(1, 2, 0))



    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    
if __name__=='__main__':
    main()
    
