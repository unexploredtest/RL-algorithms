import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariCNN(nn.Module):
    def __init__(self, nb_actions, image_shape=(84, 84), frame_skip=4):
        super(AtariCNN, self).__init__()
        self.image_shape = image_shape
        
        self.conv1 = nn.Conv2d(in_channels=frame_skip, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, nb_actions)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = x.flatten() 
        # Why we don't use flatten: No support for batch size
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x