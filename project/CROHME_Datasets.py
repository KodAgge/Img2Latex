import torch
from torch._C import dtype
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
from ast import literal_eval
pd.options.display.width = 0

from DataLoaderHelpers import loadImages, normalizeData

# TRAIN_IMG_FOLDER = '/CROHME DATA/TrainTransformed'
# VAL_IMG_FOLDER = './CROHME DATA/Validation_Transformed'
# TEST_IMG_FOLDER = './CROHME DATA/TestTransformed'
# TRAIN_ANNOTATIONS = './TRAIN_Tokenized_Normalized.txt'
TRAIN_IMG_FOLDER = './project/CROHME DATA/TRAIN_DATA'
VAL_IMG_FOLDER = './project/CROHME DATA/VALIDATION_DATA'
TEST_IMG_FOLDER = './project/CROHME DATA/TEST_DATA'
OWN_IMG_FOLDER = './project/CROHME DATA/OWN_DATA'
# TRAIN_ANNOTATIONS = './project/TRAIN_Tokenized_Normalized - kopia.txt'
TRAIN_ANNOTATIONS = './project/TRAIN_NORMALIZED_TOKENIZED.txt'
TEST_ANNOTATIONS = './project/TEST_NORMALIZED_TOKENIZED.txt'
VALIDATION_ANNOTATIONS = './project/VALIDATION_NORMALIZED_TOKENIZED.txt'
ANNOTATION_HEADERS = ['ImageFile','InkmlFile','LatexCode','InkmlFolder']

# N_TEST = 2120
N_TRAIN = 5 #7170
# N_VAL = 667
N_TEST = 5
# N_TRAIN = 7
N_VAL = 5
THRESHOLD = 165 # 0 = Black, 255 = White

N_OWN = 11

SCALE = 255 # RGB from 0-255 (black - white)
HEIGHT = 100
WIDTH = 250

ending = ".png"
TEST_IMG = loadImages(TEST_IMG_FOLDER, WIDTH, HEIGHT, N_TEST, SCALE, ending)
print("Loading of test images complete")
TRAIN_IMG = loadImages(TRAIN_IMG_FOLDER, WIDTH, HEIGHT, N_TRAIN, SCALE, ending)
print("Loading of train images complete")
VAL_IMG = loadImages(VAL_IMG_FOLDER, WIDTH, HEIGHT, N_VAL, SCALE, ending)
print("Loading of validation images complete")
OWN_IMG = loadImages(OWN_IMG_FOLDER, WIDTH, HEIGHT, N_OWN, SCALE, ".jpg", THRESHOLD)
print("Loading of own images complete")

_, _, OWN_IMGS = normalizeData(TEST_IMG, TRAIN_IMG, OWN_IMG)
TEST_IMGS, TRAIN_IMGS, VAL_IMGS = normalizeData(TEST_IMG, TRAIN_IMG, VAL_IMG)

class CROHME_Training_Set(Dataset):
    def __init__(self, annotations_file=TRAIN_ANNOTATIONS, img_dir=TRAIN_IMG_FOLDER, img_transform=ToTensor(), target_transform=None):
        self.annotations_df = pd.read_csv(TRAIN_ANNOTATIONS, names = ANNOTATION_HEADERS, sep= ';', engine='python', encoding='UTF-8')
        self.img_dir = img_dir
        self.images = TRAIN_IMGS       
        self.img_transform = img_transform


    def __len__(self):
        return self.images.shape[2] #len(self.annotations_df)


    def __getitem__(self, idx):
        label = literal_eval(self.annotations_df.iloc[idx].LatexCode)[1:] # '[1, 2, 3]' to [1, 2, 3]

        label = torch.Tensor(label).long() # Make it into a int-tensor

        image = self.images[:,:,idx]
        
        if self.img_transform:
            image = self.img_transform(image)

        return {"image": image, "label": label}


class CROHME_Testing_Set(Dataset):
    def __init__(self, annotations_file=TEST_ANNOTATIONS, img_dir=TEST_IMG_FOLDER, img_transform=ToTensor(), target_transform=None):
        self.annotations_df = pd.read_csv(TEST_ANNOTATIONS, names = ANNOTATION_HEADERS, sep= ';', engine='python', encoding='UTF-8')
        self.img_dir = img_dir
        self.images = TEST_IMGS       
        self.img_transform = img_transform


    def __len__(self):
        return self.images.shape[2] #len(self.annotations_df)


    def __getitem__(self, idx):
        label = literal_eval(self.annotations_df.iloc[idx].LatexCode)[1:] # '[1, 2, 3]' to [1, 2, 3]

        label = torch.Tensor(label).long() # Make it into a int-tensor

        image = self.images[:,:,idx]
        
        if self.img_transform:
            image = self.img_transform(image)

        return {"image": image, "label": label}


class CROHME_Validation_Set(Dataset):
    def __init__(self, annotations_file=VALIDATION_ANNOTATIONS, img_dir=VAL_IMG_FOLDER, img_transform=ToTensor(), target_transform=None):
        self.annotations_df = pd.read_csv(VALIDATION_ANNOTATIONS, names = ANNOTATION_HEADERS, sep= ';', engine='python', encoding='UTF-8')
        self.img_dir = img_dir
        self.images = VAL_IMGS       
        self.img_transform = img_transform


    def __len__(self):
        return self.images.shape[2] #len(self.annotations_df)


    def __getitem__(self, idx):
        label = literal_eval(self.annotations_df.iloc[idx].LatexCode)[1:] # '[1, 2, 3]' to [1, 2, 3]

        label = torch.Tensor(label).long() # Make it into a int-tensor

        image = self.images[:,:,idx]
        
        if self.img_transform:
            image = self.img_transform(image)

        return {"image": image, "label": label}


class CROHME_Own_Set(Dataset):
    def __init__(self, img_dir=OWN_IMG_FOLDER, img_transform=ToTensor(), target_transform=None):
        self.img_dir = img_dir
        self.images = OWN_IMGS       
        self.img_transform = img_transform


    def __len__(self):
        return self.images.shape[2] #len(self.annotations_df)


    def __getitem__(self, idx):
        label = torch.zeros(150)

        label = torch.Tensor(label).long() # Make it into a int-tensor

        image = self.images[:,:,idx]
        
        if self.img_transform:
            image = self.img_transform(image)

        return {"image": image, "label": label}

def main():
    train_set = CROHME_Training_Set()
    #print(len(train_set))
    #train_set[0]


if __name__=='__main__':
    main()