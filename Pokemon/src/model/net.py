from torch import nn
import torch.nn.functional as F

# TODO: Create more layers in your network
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.fc1 = nn.Linear(24*222*222, 64)
        #self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        soft_max = nn.Softmax()
        batch_size = x.size(0)
        # 56 x 6 x 224 x 224
        x = F.relu(self.conv1(x))
        # x = 56x12x224x224
        x = F.relu(self.conv2(x))
        # x = 56x24x222x222
        x = F.relu(self.conv3(x))
        # x = 56x1182816
        x = x.view(batch_size, -1)
        # x = 56 x 64
        x = self.fc1(x)
        # x = 56 x 64
        x = F.relu(x)
        # x = 56 x 2
        x = self.fc3(x)
        # x = 56 x 2
        x = soft_max(x)
        return x