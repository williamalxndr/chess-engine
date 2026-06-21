from types import SimpleNamespace

from torch import nn
import torch
import math
import torch.nn.functional as F
from pathlib import Path

from game.rules import Rules


class PolicyValueNetwork(nn.Module):
    def __init__(self, rules: Rules, body_channels=16):
        super().__init__()
        self.action_space_size = rules.action_space_size
        self.body_channels = body_channels

        self.body = ResidualBlock(1, body_channels, 2, 1, 1)
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(body_channels, rules.action_space_size),
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

    def save(self, game: str, path: str):
        """
        Saves weights together with the architecture metadata needed to
        reconstruct this network, so load() never needs an external rules
        argument. Only plain ints go into the checkpoint -- safe to load
        back with weights_only=True.
        """
        fixed_path = f"checkpoints/{game}/{path}.pt"

        # mkdir if not exists
        dir_path = Path(fixed_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict": self.state_dict(),
            "action_space_size": self.action_space_size,
            "body_channels": self.body_channels,
        }, fixed_path)

    @staticmethod
    def load(game: str, path: str) -> "PolicyValueNetwork":
        checkpoint = torch.load(f"checkpoints/{game}/{path}.pt", weights_only=True)

        rules_stub = SimpleNamespace(action_space_size=checkpoint["action_space_size"])
        network = PolicyValueNetwork(rules=rules_stub, body_channels=checkpoint["body_channels"])
        network.load_state_dict(checkpoint["state_dict"])
        return network


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
        self.norm1 = nn.GroupNorm(out_channels, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.norm2 = nn.GroupNorm(out_channels, out_channels)

        # Automatically creates a downsample
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if x.shape[-2:] != out.shape[-2:]:
            x = F.adaptive_avg_pool2d(x, out.shape[-2:])
        if self.in_channels != self.out_channels:
            x = self.channel_proj(x)

        out += x
        out = self.relu2(out)
        return out