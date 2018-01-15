import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import net
from dataset import dataloader

import matplotlib.pyplot as plt

IMG_PATH = 'dataset/train/'
TRAIN_DATA = 'dataset/train/out.csv'

# Center crop is shit
transformations = transforms.Compose([transforms.ToTensor()])
dset_train = dataloader.Cats_Dogs_Dataset(TRAIN_DATA, IMG_PATH, transformations)




train_loader = DataLoader(dset_train, batch_size=20, shuffle=True, num_workers=4)


model = net.LeNet()

optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []

def train(epoch):
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Always have requires_grad set to false since we don't need gradients.
        data, target = Variable(data, requires_grad=False), Variable(target.type(torch.LongTensor), requires_grad=False)
        optimizer.zero_grad()
        output = model(data)
        print(output, target)
        output_loss = loss(output, target[:, 0])
        output_loss.backward()
        optimizer.step()
        print(output_loss.data)
        # if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), output_loss.data[0]))

        losses.append(output_loss.data[0])
    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

for epoch in range(1, 2):
    train(epoch)








