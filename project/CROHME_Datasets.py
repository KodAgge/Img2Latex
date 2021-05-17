import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
pd.options.display.width = 0

from DataLoaderHelpers import loadImages, normalizeData

# TRAIN_IMG_FOLDER = '/CROHME DATA/TrainTransformed'
# VAL_IMG_FOLDER = './CROHME DATA/Validation_Transformed'
# TEST_IMG_FOLDER = './CROHME DATA/TestTransformed'
# TRAIN_ANNOTATIONS = './TRAIN_Tokenized_Normalized.txt'
TRAIN_IMG_FOLDER = './project/CROHME DATA/TrainTransformed'
VAL_IMG_FOLDER = './project/CROHME DATA/Validation_Transformed'
TEST_IMG_FOLDER = './project/CROHME DATA/TestTransformed'
TRAIN_ANNOTATIONS = './project/TRAIN_Tokenized_Normalized.txt'
ANNOTATION_HEADERS = ['ImageFile','InkmlFile','LatexCode','InkmlFolder']

N_TEST = 10
N_TRAIN = 6495
N_VAL = 20

SCALE = 255 # RGB from 0-255 (black - white)
HEIGHT = 100
WIDTH = 250

TEST_IMGS = loadImages(TEST_IMG_FOLDER, WIDTH, HEIGHT, N_TEST, SCALE)
TRAIN_IMGS = loadImages(TRAIN_IMG_FOLDER, WIDTH, HEIGHT, N_TRAIN, SCALE)
VAL_IMGS = loadImages(VAL_IMG_FOLDER, WIDTH, HEIGHT, N_VAL, SCALE)
TEST_IMGS, TRAIN_IMGS, VAL_IMGS = normalizeData(TEST_IMGS, TRAIN_IMGS, VAL_IMGS)

class CROHME_Training_Set(Dataset):
    def __init__(self, annotations_file=TRAIN_ANNOTATIONS, img_dir=TRAIN_IMG_FOLDER, img_transform=ToTensor(), target_transform=None):
        self.annotations_df = pd.read_csv(TRAIN_ANNOTATIONS, names = ANNOTATION_HEADERS, sep= '§', engine='python', encoding='ISO-8859-1')
        self.img_dir = img_dir
        self.images = TRAIN_IMGS       
        
        self.img_transform = img_transform

        #print(self.images.shape)
        #print(type(self.annotations_df))
        #print(self.annotations_df.iloc[1])
        #print(self.annotations_df.head())

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        label = self.annotations_df.iloc[idx].LatexCode
        print(self.images)
        print(self.images.shape)
        image = self.images[:,:,idx]
        
        if self.img_transform:
            image = self.img_transform(image)
        
        #print(label)
        #print(image)

        return {"image": image, "label": label}



def main():
    train_set = CROHME_Training_Set()
    #print(len(train_set))
    #train_set[0]

if __name__=='__main__':
    main()