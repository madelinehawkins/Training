import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

from src.model import net
from src.train import train
from src.test import test
from src.dataset import dataloader
from src import test

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TRAIN_PATH = 'src/dataset/training/'
TEST_PATH = 'src/dataset/test/'
TRAIN_DATA = 'src/dataset/train_out.csv'
TEST_DATA = 'src/dataset/test/test_out.csv'


transformations = transforms.Compose([transforms.ToTensor()])
dset_train = dataloader.Pokemon_Dataset(TRAIN_DATA, TRAIN_PATH, transformations)


# Checking my model by grabbing only two pokemon and seeing how well it overfits
# dset_train.X_train = dset_train.X_train[:60]
# dset_train.y_train = dset_train.y_train[:60]


dset_test = dset_train


train_loader = DataLoader(dset_train, batch_size=20, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=30, shuffle=False, num_workers=4)

model = net.LeNet()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 30
loss, n_model = train(epochs, train_loader, model, optimizer)

# Get individual class results.


test.test(n_model, test_loader)

plt.plot(loss)
plt.show()
