import torch
from torch.cuda import init
# import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from skimage.io import imread
from skimage.io import imshow
# import cv2
# import math
# import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class Net(nn.Module):
  # The one used in the original paper
  def __init__(self):
    super().__init__()
    # First row (from end)
    self.conv1 = nn.Conv2d(1, 64, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=(1, 1)).double() # size_of_pool, stride, can't have (2,2) padding

    # Second row (from end)
    self.conv2 = nn.Conv2d(64, 128, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2)).double() # size_of_pool, stride

    # Third row (from end)
    self.conv3 = nn.Conv2d(128, 256, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.bn3 = nn.BatchNorm2d(256).double()

    # Fourth row (from end)
    self.conv4 = nn.Conv2d(256, 256, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.pool4 = nn.MaxPool2d((2, 1), stride=(2, 1)).double() # size_of_pool, stride

    # Fifth row (from end)
    self.conv5 = nn.Conv2d(256, 512, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.bn5 = nn.BatchNorm2d(512).double()
    self.pool5 = nn.MaxPool2d((1, 2), stride=(1, 2)).double() # size_of_pool, stride

    # Sixth row (from end)
    self.conv6 = nn.Conv2d(512, 512, 3).double() # num_input_channels, num_filters, filter_size
    self.bn6 = nn.BatchNorm2d(512).double()

    # Initialize weights
    self.init_weights()

    # # Optional flattening for testing
    # self.fc = nn.Linear(148480, 2).double()


  def init_weights(self): 
    # Initialisez the wights with Xavier Normalization
    torch.nn.init.xavier_normal_(self.conv1.weight)
    torch.nn.init.xavier_normal_(self.conv2.weight)
    torch.nn.init.xavier_normal_(self.conv3.weight)
    torch.nn.init.xavier_normal_(self.conv4.weight)
    torch.nn.init.xavier_normal_(self.conv5.weight)
    torch.nn.init.xavier_normal_(self.conv6.weight)


  def forward(self, x):
    # First row (from end)
    x = F.relu(self.conv1(x))
    x = self.pool1(x)

    # Second row (from end)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)

    # Third row (from end)
    x = F.relu(self.conv3(x))
    x = self.bn3(x)

    # Fourth row (from end)
    x = F.relu(self.conv4(x))
    x = self.pool4(x)

    # Fifth row (from end)
    x = F.relu(self.conv5(x))
    x = self.bn5(x)
    x = self.pool5(x)

    # Sixth row (from end)
    x = F.relu(self.conv6(x))
    x = self.bn6(x)

    # # For testing
    # x = torch.flatten(x, 1) # flatten all dimensions except batch
    # x = F.relu(self.fc(x))

    return x