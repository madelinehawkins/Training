"""
 Arguments:
    A CSV file path
    Path to image folder
    PIL transforms

"""

import pandas as pd
from pandas import Series
from torch import np

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset

from sklearn.preprocessing import MultiLabelBinarizer

class Pokemon_Dataset(Dataset):

    def __init__(self, csv_path, img_path, transform=None):

        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x)).all()

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tag'].astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.y_train[index]
        return img, label

    def __len__(self):
        return len(self.X_train.index)



