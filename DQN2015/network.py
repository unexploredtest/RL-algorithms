import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariCNN(nn.Module):
    def __init__(self, nb_actions, image_shape=(84, 84), frame_skip=4):
        super(AtariCNN, self).__init__()
        self.image_shape = image_shape
        
        self.conv1 = nn.Conv2d(in_channels=frame_skip, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, nb_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x