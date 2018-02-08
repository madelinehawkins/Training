from torch import nn
import torch.nn.functional as F

# TODO: Create more layers in your network



class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12*50*50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        soft_max = nn.Softmax()
        batch_size = x.size(0)
        # 10x6x202x202
        x = self.pool(F.relu(self.conv1(x)))
        # x = 10x12x50x50
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batch_size, -1)
        # x =  10x64
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = 10 x 20
        x = self.fc3(x)
        # x = 10 x 20
        x = soft_max(x)
        return x


# Go through and figure out if padding is ok.

class AlexNet(nn.Module):

    def __init__(self, num_classes=20):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(12,24, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(24 * 24 * 24, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        soft_max = nn.Softmax()
        # 20 x 24 x 24 x 24
        x = self.features(x)
        x = x.view(x.size(0), 24 * 24 * 24)
        x = self.classifier(x)
        # x = soft_max(x)
        return x
