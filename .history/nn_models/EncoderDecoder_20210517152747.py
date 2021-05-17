import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
#sys.path.append("./data/")
sys.path.insert(0, 'C:\Users\EMIL\Documents\Previous Desktop\Skolan år 4\Deep Learning in Data Science\Project\DD2424_Img2Latex\data')
from CROHME_Datasets import CROHME_Training_Set
#from CROHME_Datasets import CROHME_Training_Set

class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

    def init_parameters(self):
        pass

    def forward(self, X_batch): 
        pass

def main():
    train_set = CROHME_Training_Set()
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)

if __name__=='__main__':
    main()
    
