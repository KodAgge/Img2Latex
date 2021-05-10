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


'''
To do

1. Reshape images DONE
  1.1 Crop images NOT DONE
2. Normalize data DONE
3. Import labels
4. Define loss function
5. First conv layer to match image size
6. Train on small dataset

Mischenallous
- Clean up code
- Transform training images

'''

class Net(nn.Module):
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


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    # img = (img + 1) * (255 / 2)    # unnormalize
    npimg = img.numpy()
    # print(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.imshow(npimg[0,:,:])
    plt.show()


def loadData(path, width, height, n_test, scale = 255):
  # Preallocate memory
  data = np.zeros([width, height, n_test])

  for i in range(n_test):
    image_path = path + '/Image' + str(i) + '.png'
    img = imread(image_path) # Read image
    plt.imshow(img)
    plt.show()

    img_gray_scale = img[:,:,0] # Remove unessecary dimensions
    # TODO
    # Add code that automatically crops out whitespace around equation
    plt.imshow(img_gray_scale)
    plt.show()

    img_rescaled = img_gray_scale / (scale / 2) - 1 # Rescale to [-1, 1]

    plt.imshow(img_rescaled)
    plt.show()

    img_resize = cv2.resize(img_rescaled, [width, height]) # Resize image
    data[:,:,i] = img_resize # Save transformed image data

    plt.imshow(img_resize)
    plt.show()

    # print(img_rescaled.shape)
    # print(img_resize.shape)

  return data

def normalizeData(test_data, train_data, val_data):
  # Find mean and std
  all_data = np.array([train_data, test_data, val_data])
  mean = np.mean(all_data)
  std = np.std(all_data)

  # Normalize
  test_data = (test_data - mean) / std
  train_data = (train_data - mean) / std
  val_data = (val_data - mean) / std

  return test_data, train_data, val_data




n_test = 1
n_train = 1
n_val = 1
width = 100
height = 100
scale = 255
path_test = 'C:/Users/TheBeast/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/TestTransformed'
path_train = 'C:/Users/TheBeast/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/TrainTransformed'
path_val = 'C:/Users/TheBeast/Documents/GitHub/DD2424_Img2Latex/data/CROHME DATA/TestTransformed'

test_data = loadData(path_test, width, height, n_test, scale)
train_data = loadData(path_train, width, height, n_train, scale)
val_data = loadData(path_val, width, height, n_val, scale)

# test_data, train_data, val_data = normalizeData(test_data, train_data, val_data)

net = Net()

batch_size = 4

# Laddar dem här in labels? Hur ska labels komma med?

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Assuming that we are on a CUDA machine, this should print a CUDA device:

# print(device)

# net = Net()

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



# batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# get some random training images
# dataiter = iter(trainloader)
# print(dataiter.next().shape)
# images = dataiter.next()
# images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')

# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# net = Net()
# net.load_state_dict(torch.load(PATH))

# outputs = net(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

# # prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
#                                                    accuracy))