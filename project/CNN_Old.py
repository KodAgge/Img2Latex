import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.io import imread
from skimage.io import imshow
import cv2
import math
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



'''
To do

- Reshape images DONE
- Normalize data DONE
- Import labels
- Define loss function
- First conv layer to match image size DONE
- Define layers DONE
- Activation function?
- Batcg normalization?
- Train on small dataset

Mischenallous
- Clean up code
- Transform training images

'''

class NetExample(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):

    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


class NetTest(nn.Module):
  # Original + 1D output
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 512, 3).double() # num_input_channels, num_filters, filter_size
    self.bn1 = nn.BatchNorm2d(512).double()

    self.conv2 = nn.Conv2d(512, 512, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.bn2 = nn.BatchNorm2d(512).double()

    self.pool1 = nn.MaxPool2d((1, 2), stride=(1, 2)).double() # size_of_pool, stride
    self.conv3 = nn.Conv2d(512, 256, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.pool2 = nn.MaxPool2d((2, 1), stride=(2, 1)).double() # size_of_pool, stride
    self.conv4 = nn.Conv2d(256, 256, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.bn3 = nn.BatchNorm2d(256).double()

    self.conv5 = nn.Conv2d(256, 128, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2)).double() # size_of_pool, stride
    self.conv6 = nn.Conv2d(128, 64, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size

    self.fc1 = nn.Linear(95232, 2).double()

  def forward(self, x):
    (batch_size, height, width) = x.shape
    x = x.reshape(batch_size, 1, height, width)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.bn2(self.pool1(self.conv2(x)))
    x = self.pool2(self.conv3(x))
    x = self.bn3(self.conv4(x))
    x = self.pool3(self.conv5(x))
    x = self.conv6(x)

    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.fc1(x)

    return x


class NetOld(nn.Module):
  # The one used in the original paper
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 512, 3).double() # num_input_channels, num_filters, filter_size
    self.bn1 = nn.BatchNorm2d(512).double()

    self.conv2 = nn.Conv2d(512, 512, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.bn2 = nn.BatchNorm2d(512).double()

    self.pool1 = nn.MaxPool2d((1, 2), stride=(1, 2)).double() # size_of_pool, stride
    self.conv3 = nn.Conv2d(512, 256, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.pool2 = nn.MaxPool2d((2, 1), stride=(2, 1)).double() # size_of_pool, stride
    self.conv4 = nn.Conv2d(256, 256, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.bn3 = nn.BatchNorm2d(256).double()

    self.conv5 = nn.Conv2d(256, 128, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2)).double() # size_of_pool, stride
    self.conv6 = nn.Conv2d(128, 64, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size
    self.pool4 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=(1, 1)).double() # size_of_pool, stride, can't have (2,2) padding

  def forward(self, x):
    (batch_size, height, width) = x.shape
    x = x.reshape(batch_size, 1, height, width)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.bn2(self.pool1(self.conv2(x)))
    x = self.pool2(self.conv3(x))
    x = self.bn3(self.conv4(x))
    x = self.pool3(self.conv5(x))
    # x = self.conv6(x)
    x = self.pool4(self.conv6(x))

    return x


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

    # Optional flattening for testing
    self.fc = nn.Linear(148480, 2).double()

  def forward(self, x):
    # Reshape vector
    (batch_size, height, width) = x.shape
    x = x.reshape(batch_size, 1, height, width)

    # First row (from end)
    x = self.conv1(x)
    x = self.pool1(x)

    # Second row (from end)
    x = self.conv2(x)
    x = self.pool2(x)

    # Third row (from end)
    x = self.conv3(x)
    x = self.bn3(x)

    # Fourth row (from end)
    x = self.conv4(x)
    x = self.pool4(x)

    # Fifth row (from end)
    x = self.conv5(x)
    x = self.bn5(x)
    x = self.pool5(x)

    # Sixth row (from end)
    x = self.conv6(x)
    x = self.bn6(x)

    # For testing
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc(x))

    return x


class NetStanford(nn.Module):
  # The one used in the original paper
  def __init__(self):
    super().__init__()
    # First row (from end)
    self.conv1 = nn.Conv2d(1, 64, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size

    # Second row (from end)
    self.conv2 = nn.Conv2d(64, 128, 3, padding=(1, 1)).double() # num_input_channels, num_filters, filter_size

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

    # Optional flattening for testing
    self.fc = nn.Linear(148480, 2).double()

  def forward(self, x):
    # Reshape vector
    (batch_size, height, width) = x.shape
    x = x.reshape(batch_size, 1, height, width)

    # First row (from end)
    x = self.conv1(x)

    # Second row (from end)
    x = self.conv2(x)

    # Third row (from end)
    x = self.conv3(x)
    x = self.bn3(x)

    # Fourth row (from end)
    x = self.conv4(x)
    x = self.pool4(x)

    # Fifth row (from end)
    x = self.conv5(x)
    x = self.bn5(x)
    x = self.pool5(x)

    # Sixth row (from end)
    x = self.conv6(x)
    x = self.bn6(x)

    # For testing
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc(x))

    return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def loadImages(path, width, height, n_test, scale = 255):
  # Preallocate memory
  data = np.zeros([height, width, n_test])

  for i in range(n_test):
    image_path = path + '/Image' + str(i) + '.png'
    img = imread(image_path) # Read image

    img_gray_scale = img[:,:,0] # Remove unessecary dimensions

    (img_height, img_width) = img_gray_scale.shape # Current shape

    # If the picture is to short compared to the required format
    
    if img_height / img_width < height / width:
      new_height = height / width * img_width
      pad = (new_height - img_height) / 2
      img_padded= cv2.copyMakeBorder(img_gray_scale, math.ceil(pad), math.floor(pad), 0, 0, cv2.BORDER_CONSTANT, value=255)
    
    # If the picture is to narrow compared to the required format
    elif img_height / img_width > height / width:
      new_width = img_height * width / height
      pad = (new_width - img_width) / 2
      img_padded= cv2.copyMakeBorder(img_gray_scale, 0, 0, math.ceil(pad),math.floor(pad), cv2.BORDER_CONSTANT, value=255)

    img_rescaled = img_padded / (scale / 2) - 1 # Rescale to [-1, 1]

    #FIXME Crashes when using [width, height]. Changed to tuple (width, height)
    #img_resize = cv2.resize(img_rescaled, [width, height]) # Resize image 
    img_resize = cv2.resize(img_rescaled, (width, height)) # Resize image

    data[:,:,i] = img_resize # Save transformed image data

    # plt.imshow(img_resize)
    # plt.show()

  return data


def normalizeData(test_images, train_images, val_images):
  # Find mean and std
  all_data = np.concatenate([train_images, test_images, val_images], axis = 2)
  mean = np.mean(all_data)
  std = np.std(all_data)

  # Normalize
  test_images = (test_images - mean) / std
  train_images = (train_images - mean) / std
  val_images = (val_images - mean) / std

  return test_images, train_images, val_images


def addLabels(images, labels, random=False):
  n = images.shape[2]
  data = []

  if random:
    for i in range(n):
      if i < n / 2:
        k = 0
      else:
        k = 1

      data.append([images[:,:,i], k])

  else:
    for i in range(n):
      data.append([images[:,:,i], labels[i]])

  return data


#Number of datapoints to include in training
n_test = 10
n_train = 5
n_val = 20

#Average image dimensions on testset: 198.7316715542522 x 502.2697947214076
height = 100
width = 250

# Load and transform images
scale = 255 # RGB from 0-255 (black - white)

#path_test = 'C:/Users/TheBeast/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/TestTransformed'
path_test = '/Users/carlhoggren/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/TestTransformed' 

#path_train = 'C:/Users/TheBeast/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/TrainTransformed'
path_train = '/Users/carlhoggren/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/TrainTransformed'

#path_val = 'C:/Users/TheBeast/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/Validation_Transformed'
path_val = '/Users/carlhoggren/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/Validation_Transformed'

#Load images and rescale to required dimensions 
test_images = loadImages(path_test, width, height, n_test, scale)
train_images = loadImages(path_train, width, height, n_train, scale)
val_images = loadImages(path_val, width, height, n_val, scale)

# Normalize data
test_images, train_images, val_images = normalizeData(test_images, train_images, val_images)

# Append labels
test_data = addLabels(test_images, 0, True)
train_data = addLabels(train_images, 0, True)
val_data = addLabels(val_images, 0, True)

# Init network
net = Net()

#REVIEW START 
config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}
net = Net(config["l1"], config["l2"]) 
#REVIEW END

batch_size = 2
n_epochs = 2

#Load data into .torch format
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

# Dummy classes for testing the network
classes = ('0', '1')

#ANCHOR Start
# # Kod som tar fram några bilder och plottar en av dem
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# plt.imshow(images[0])
# plt.show()
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
#ANCHOR End


# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#REVIEW START
optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9) 
#REVIEW END

# Training cycle
start_time = time.perf_counter()

for epoch in range(n_epochs):
    print("Epoch:", epoch + 1)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        print("\tBatch:", i + 1)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

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

end_time = time.perf_counter()
print('Finished Training')
print('It took', end_time - start_time, 'seconds')



'''CALCULATE ACCURACY'''
correct = 0
total = 0
predictions = []
with torch.no_grad():
    for data in testloader: # testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# Calculate accuracy per class

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader: #testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))