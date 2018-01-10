"""
 Arguments:
    A CSV file path
    Path to image folder
    PIL transforms

"""


import pandas as pd
from torch import np

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.preprocessing import MultiLabelBinarizer

IMG_PATH = 'train/'
TRAIN_DATA = 'train/out.csv'

class Cats_Dogs_Dataset(Dataset):

    def __init__(self, csv_path, img_path, transform=None):

        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x)).all()

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tag'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


