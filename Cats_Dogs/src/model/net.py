__author__ = 'Madeline Hawkins'

from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.fc1 = nn.Linear(12*2*2, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, x):
        batch_size = x.size(0)
        # x = 20, 3, 32, 32
        # x = F.max_pool2d(F.relu(self.conv1(x)), 3)
        x = F.relu(self.conv1(x))
        # x = 20, 6, 32, 32
        x = F.max_pool2d(x, 3)
        # x = 20, 6, 10, 10
        x = F.max_pool2d(F.relu(self.conv2(x)), 3)
        # x = 20, 12, 2, 2
        x = x.view(batch_size, -1)
        # x = 20, 48
        # Since Just cats and dogs
        x = self.fc1(x)
        # x = 20, 24
        x = F.relu(x)
        # x = 20, 24 no negatives
        x = F.relu(self.fc2(x))
        # x = 20, 12 no negatives
        x = self.fc3(x)
        # x = 20, 2
        return x






# class AlexNet(nn.Module):
#
#     def __init__(self, num_classes=1000):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         return x







