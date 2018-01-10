from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import net
from dataset import dataloader

IMG_PATH = 'dataset/train/'
TRAIN_DATA = 'dataset/train/out.csv'

transformations = transforms.Compose([transforms.Scale(32), transforms.ToTensor()])
dset_train = dataloader.Cats_Dogs_Dataset(TRAIN_DATA, IMG_PATH, transformations)


train_loader = DataLoader(dset_train, batch_size=256, shuffle=True, num_workers=4)


model = net.AlexNet()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


for epoch in range(1, 2):
    train(epoch)








