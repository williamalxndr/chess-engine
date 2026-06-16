from torch import nn
import math
import torch.nn.functional as F

class PolicyValueNetwork(nn.Module):
    def __init__(self, body_channels=16):
        super().__init__()
        self.body = ResidualBlock(1, body_channels, 2, 1, 1)
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(body_channels, 9),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(body_channels, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Args:
            x (np.ndarray): input data (board) with shape (3,3)
        Returns policy_head, value_head
        """
        features = self.body(x)
        return self.policy_head(features), self.value_head(features)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Automatically creates a downsample
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if x.shape[-2:] != out.shape[-2:]:
            x = F.adaptive_avg_pool2d(x, out.shape[-2:])
        if self.in_channels != self.out_channels:
            x = self.channel_proj(x)
        
        out += x
        out = self.relu2(out)
        return out
