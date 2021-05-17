import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd

TRAIN_IMG_FOLDER = './CROHME DATA/TrainTransformed'
TRAIN_ANNOTATIONS = './TRAIN_Tokenized_Normalized.txt'
ANNOTATION_HEADERS = ['ImageFile','InkmlFile','LatexCode','InkmlFolder']

pd.options.display.width = 0



class CROHME_Training_Set(Dataset):
    def __init__(self, annotations_file=TRAIN_ANNOTATIONS, img_dir=TRAIN_IMG_FOLDER, transform=None, target_transform=None):
        self.annotations_df = pd.read_csv(TRAIN_ANNOTATIONS, names = ANNOTATION_HEADERS, sep= '§', engine='python')
        self.img_dir = img_dir
        
        #Load images and rescale to required dimensions 
        test_images = loadImages(path_test, width, height, n_test, scale)
        train_images = loadImages(path_train, width, height, n_train, scale)
        val_images = loadImages(path_val, width, height, n_val, scale)

        # Normalize data
        test_images, train_images, val_images = normalizeData(test_images, train_images, val_images)
        
        
        
        #print(type(self.annotations_df))
        #print(self.annotations_df.iloc[1])
        #print(self.annotations_df.head())

        train_images = loadImages(path_train, width, height, n_train, scale)


    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        pass



def main():
    train_set = CROHME_Training_Set()
    print(len(train_set))


if __name__=='__main__':
    main()