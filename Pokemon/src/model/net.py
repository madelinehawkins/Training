from torch import nn
import torch.nn.functional as F

# TODO: Create more layers in your network
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.fc1 = nn.Linear(24*198*198, 64)
        #self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(64, 20)

    def forward(self, x):
        soft_max = nn.Softmax()
        batch_size = x.size(0)
        # 10x6x202x202
        x = F.relu(self.conv1(x))
        # x = 10x12x200x200
        x = F.relu(self.conv2(x))
        # x = 10x24x198x198
        x = F.relu(self.conv3(x))
        # x = 10x940896
        x = x.view(batch_size, -1)
        # x =  10x64
        x = F.relu(self.fc1(x))
        # x = 10 x 20
        x = self.fc3(x)
        # x = 10 x 20
        x = soft_max(x)
        return x