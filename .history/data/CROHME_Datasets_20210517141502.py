import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
pd.options.display.width = 0

from DataLoaderHelpers import loadImages, normalizeData

TRAIN_IMG_FOLDER = './CROHME DATA/TrainTransformed'
VAL_IMG_FOLDER = './CROHME DATA/Validation_Transformed'
TEST_IMG_FOLDER = './CROHME DATA/TestTransformed'
TRAIN_ANNOTATIONS = './TRAIN_Tokenized_Normalized.txt'
ANNOTATION_HEADERS = ['ImageFile','InkmlFile','LatexCode','InkmlFolder']

N_TEST = 10
N_TRAIN = 5
N_VAL = 20

SCALE = 255 # RGB from 0-255 (black - white)
HEIGHT = 100
WIDTH = 250

TEST_IMGS = loadImages(TEST_IMG_FOLDER, WIDTH, HEIGHT, N_TEST, SCALE)
TRAIN_IMGS = loadImages(TRAIN_IMG_FOLDER, WIDTH, HEIGHT, N_TRAIN, SCALE)
VAL_IMGS = loadImages(VAL_IMG_FOLDER, WIDTH, HEIGHT, N_VAL, SCALE)
TEST_IMGS, TRAIN_IMGS, VAL_IMGS = normalizeData(TEST_IMGS, TRAIN_IMGS, VAL_IMGS)

class CROHME_Training_Set(Dataset):
    def __init__(self, annotations_file=TRAIN_ANNOTATIONS, img_dir=TRAIN_IMG_FOLDER, transform=None, target_transform=None):
        self.annotations_df = pd.read_csv(TRAIN_ANNOTATIONS, names = ANNOTATION_HEADERS, sep= '§', engine='python')
        self.img_dir = img_dir
        self.TRAIN_IMGS = TRAIN_IMGS        
        print(type(self.TRAIN_IMGS))
        
        #print(type(self.annotations_df))
        #print(self.annotations_df.iloc[1])
        #print(self.annotations_df.head())



    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        pass



def main():
    train_set = CROHME_Training_Set()
    print(len(train_set))


if __name__=='__main__':
    main()