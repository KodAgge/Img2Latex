import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../data/')

import CROHME_Datasets

class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

    def init_parameters(self):
        pass

    def forward(self, X_batch): 
        pass

def main():
    pass

if __name__=='__main__':
    main()
    train_set = CROHME_Training_Set()
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
