import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

TRAIN_ANNOTATIONS = './TRAIN_Tokenized_Normalized'
TRAIN_IMG_FOLDER = './CROHME DATA/TrainTransformed'

class CROHME_Training_Set(Dataset):
    def __init__(self, annotations_file=TRAIN_ANNOTATIONS, img_dir=TRAIN_IMG_FOLDER, transform=None, target_transform=None):
        self.annotations_file = annotations_file
        self.img_dir = img_dir

    def __len__(self):
        return 0

    def __getitem__(self, idx):


def main():
    pass



if __name__=='__main__':
    main()