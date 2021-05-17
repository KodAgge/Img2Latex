import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd

TRAIN_ANNOTATIONS = './TRAIN_Tokenized_Normalized.txt'
TRAIN_IMG_FOLDER = './CROHME DATA/TrainTransformed'

pd.options.display.width = 0

class CROHME_Training_Set(Dataset):
    def __init__(self, annotations_file=TRAIN_ANNOTATIONS, img_dir=TRAIN_IMG_FOLDER, transform=None, target_transform=None):
        self.annotations_df = pd.read_csv(TRAIN_ANNOTATIONS, sep= '§', engine='python')
        self.img_dir = img_dir
        print('hoj')
        print(type(self.annotations_df))
        #print(self.annotations_df.iloc[0])
        print(self.annotations_df.iloc[0][0])

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass

def main():
    train_set = CROHME_Training_Set()



if __name__=='__main__':
    main()