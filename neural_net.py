import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 5) # 3 input, 12 output(feature maps), kernel of 5
        self.conv2 = nn.Conv2d(12, 24, 5) # 12 input, 24 output, kernel of 5
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    
    def forward(self, x):
        # (3 channel 32x32 feature map) -> (12 channels 28x28 feature map) -> after pool(12 channel 14x14 feature map)
        x = self.pool(F.relu(self.conv1(x))) 

        # (12 channel 14x14 feature map) -> (24 channels 10x10 feature map) -> after pool(24 channel 5x5 feature map)
        x = self.pool(F.relu(self.conv2(x)))

        # (24 channel 5x5 feature map) to a list of pixel value (1 x 5*5*24=600)
        x = torch.flatten(x, 1)

        # (1x600) -> (1X120)
        x = F.relu(self.fc1(x))

        # (1x120) -> (1X84)
        x = F.relu(self.fc2(x))

        # (1x84) -> (1X10)
        x = self.fc3(x)

        return x

        
                
