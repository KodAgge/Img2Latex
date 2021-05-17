import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor




class CROHME_Training_Set(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.annotations_file = annotations_file


    def __len__(self):
        return 0

    def __getitem__(self, idx):


def main():
    pass



if __name__=='__main__':
    main()