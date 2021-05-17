import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from CNN import Net as CNN
# import LSTM

import sys

sys.path.insert(0, '..\data')
from CROHME_Datasets import CROHME_Training_Set



class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.CNN = CNN()
        

    def init_parameters(self): # Needed if networks initalize themselves?
        pass

    def forward(self, X_batch): 
        x = self.CNN(X_batch)

def MGD(net, train_dataloader, learning_rate, momentum, n_epochs):
    criterion = nn.CrossEntropyLoss() # Ändra denna?

    
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["image"], data["label"]

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)

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
    print(train_set[0]['image'])
    print(train_set[0]['label'])
    train_dataloader = DataLoader(train_set, batch_size=20, shuffle=True)
    ED = EncoderDecoder(1, 1)

    ED_Trained = MGD(ED, train_dataloader, learning_rate=0.001, momentum=0.9, n_epochs=10)

if __name__=='__main__':
    main()
    
