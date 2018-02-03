import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

from src.model import net
from src.train import train
from src.dataset import dataloader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TRAIN_PATH = 'src/dataset/training/'
#TEST_PATH = 'src/dataset/test/'
TRAIN_DATA = 'src/dataset/train_out.csv'
#TEST_DATA = 'src/dataset/test/test_out.csv'


transformations = transforms.Compose([transforms.ToTensor()])
dset_train = dataloader.Pokemon_Dataset(TRAIN_DATA, TRAIN_PATH, transformations)
# dset_test = dataloader.Pokemon_Dataset(TEST_DATA, TEST_PATH, transformations)


train_loader = DataLoader(dset_train, batch_size=10, shuffle=True, num_workers=4)
# test_loader = DataLoader(dset_test, batch_size=24, shuffle=False, num_workers=4)

model = net.LeNet()
lr = 1e-5
optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
losses = []
epochs = 2
loss, n_model = train(epochs, train_loader, model, optimizer)
plt.plot(loss)
plt.show()