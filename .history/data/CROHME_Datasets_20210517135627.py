import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd

from DataLoaderHelpers import loadImages, normalizeData

TRAIN_IMG_FOLDER = './CROHME DATA/TrainTransformed'
VAL_IMG_FOLDER = './CROHME DATA/Validation_Transformed'
TEST_IMG_FOLDER = './CROHME DATA/TestTransformed'

TRAIN_ANNOTATIONS = './TRAIN_Tokenized_Normalized.txt'
ANNOTATION_HEADERS = ['ImageFile','InkmlFile','LatexCode','InkmlFolder']

pd.options.display.width = 0

TEST_IMGS = loadImages(TEST_IMG_FOLDER, width, height, n_test, scale)
TRAIN_IMGS = loadImages(TRAIN_IMG_FOLDER, width, height, n_train, scale)
VAL_IMGS = loadImages(VAL_IMG_FOLDER, width, height, n_val, scale)
TEST_IMGS, TRAIN_IMGS, VAL_IMGS = normalizeData(test_images, train_images, val_images)

class CROHME_Training_Set(Dataset):
    def __init__(self, annotations_file=TRAIN_ANNOTATIONS, img_dir=TRAIN_IMG_FOLDER, transform=None, target_transform=None):
        self.annotations_df = pd.read_csv(TRAIN_ANNOTATIONS, names = ANNOTATION_HEADERS, sep= '§', engine='python')
        self.img_dir = img_dir
        self.TRAIN_IMGS = TRAIN_IMGS        
    
        
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