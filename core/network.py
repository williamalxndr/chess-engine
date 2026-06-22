from types import SimpleNamespace

from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np

from game.rules import Rules, TicTacToeRules, ChessRules


class PolicyValueNetwork(nn.Module, ABC):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """
        Register every subclass by name so load() can rebuild it later.

        Args:
            cls (type): the subclass being defined.
            **kwargs: forwarded to the parent __init_subclass__.
        """
        super().__init_subclass__(**kwargs)
        PolicyValueNetwork._registry[cls.__name__] = cls

    def __init__(self, rules: Rules):
        """
        Build the shared body plus the policy and value heads.

        Args:
            rules (Rules): game rules; supplies action_space_size and
                encoded_channels (the number of input channels).
        """
        super().__init__()
        self.rules = rules
        self.action_space_size = rules.action_space_size
        self.in_channels = rules.encoded_channels
        self.body = self._build_body()

        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.body_channels, rules.action_space_size)
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.body_channels, 1),
            nn.Tanh()
        )

    @abstractmethod
    def _build_body(self):
        """
        Define the body layers (defined by subclasses).

        Returns:
            nn.Module: the feature-extraction body the heads run on top of.
        """
        pass

    @property
    @abstractmethod
    def body_channels(self):
        """
        Number of output channels the body produces (feeds both heads).

        Returns:
            int: the body's output channel count.
        """
        pass

    def forward(self, x):
        """
        Run the body and both heads on a batch of encoded states.

        Args:
            x (torch.Tensor): input batch of shape (B, in_channels, H, W).

        Returns:
            policy (torch.Tensor): action probabilities, shape (B, action_space_size).
            value (torch.Tensor): value estimates in [-1, 1], shape (B, 1).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(next(self.parameters()).device)

        features = self.body(x)
        return self.policy_head(features), self.value_head(features)

    def save(self, game: str, path: str):
        """
        Save the weights plus the architecture metadata, so load() can
        rebuild the network without an external rules argument. The
        checkpoint stores only plain ints, so weights_only=True is safe.

        Args:
            game (str): game name; selects the checkpoints/<game> folder.
            path (str): file name (without extension) to save under.
        """
        dir_path = f"checkpoints/{game}"
        fixed_path = f"{dir_path}/{path}.pt"

        Path(dir_path).mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict": self.state_dict(),
            "action_space_size": self.action_space_size,
            "in_channels": self.in_channels,
            "class_name": self.__class__.__name__,
        }, fixed_path)

    @staticmethod
    def load(game: str, path: str) -> "PolicyValueNetwork":
        """
        Rebuild a saved network from its checkpoint and load its weights.

        Args:
            game (str): game name; selects the checkpoints/<game> folder.
            path (str): file name (without extension) to load.

        Returns:
            PolicyValueNetwork: the reconstructed network with weights loaded.

        Raises:
            ValueError: if the checkpoint's class is not in the registry.
        """
        checkpoint = torch.load(f"checkpoints/{game}/{path}.pt", weights_only=True, map_location="cpu")

        rules_stub = SimpleNamespace(
            action_space_size=checkpoint["action_space_size"],
            encoded_channels=checkpoint["in_channels"],
        )

        class_name = checkpoint["class_name"]

        if class_name not in PolicyValueNetwork._registry:
            raise ValueError(f"Network class '{class_name}' is not registered in registry.")

        network_class = PolicyValueNetwork._registry[class_name]

        network = network_class(rules=rules_stub)
        network.load_state_dict(checkpoint["state_dict"])

        return network

class ChessNetwork:
    LATEST = "ChessNetworkV1"

    def __new__(cls, rules: Rules, version: str = None):
        if version is not None:
            target = f"ChessNetwork{version.upper()}"
        else:
            target = cls.LATEST
        
        if target not in PolicyValueNetwork._registry:
            raise ValueError(f"Network version '{target}' is not registered.")
        return PolicyValueNetwork._registry[target](rules)    

class ChessNetworkV1(PolicyValueNetwork):
    # Legacy
    # 10 Residual blocks
    def __init__(self, rules: Rules):
        super().__init__(rules)

    def _build_body(self):
        return nn.Sequential(
            ResidualBlock(self.in_channels, 32, padding=1),
            ResidualBlock(32, 64, padding=1),
            ResidualBlock(64, 128, padding=1),
            ResidualBlock(128, 256, padding=1),
            ResidualBlock(256, 256, padding=1),
            ResidualBlock(256, 256, padding=1),
            ResidualBlock(256, 256, padding=1),
            ResidualBlock(256, 256, padding=1),
            ResidualBlock(256, 64, padding=1),
            ResidualBlock(64, 32, padding=1)
        )

    @property
    def body_channels(self):
        return 32


class TicTacToeNetwork(PolicyValueNetwork):
    def __init__(self, rules: Rules):
        super().__init__(rules)

    def _build_body(self):
        """
        Small residual body for the 3x3 board (1 -> 16 channels).

        Returns:
            nn.Sequential: the residual body.
        """
        return nn.Sequential(
            ResidualBlock(self.in_channels, 8, kernel_size=2),
            ResidualBlock(8, 16, kernel_size=1)
        )

    @property
    def body_channels(self):
        return 16



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        Two-conv residual block with a skip connection.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): convolution kernel size.
            stride (int): stride of the first convolution.
            padding (int): padding applied to both convolutions.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.norm2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Run the two-conv block and add the (shape-matched) skip connection.

        Args:
            x (torch.Tensor): input batch, shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: output batch, shape (B, out_channels, H', W').
        """
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


class NetworkFactory:
    @staticmethod
    def create(game):
        """
        Build the right network for a game from its default rules.

        Args:
            game (str): "tictactoe" or "chess" (case-insensitive).

        Returns:
            PolicyValueNetwork: a fresh network for that game.
        """
        if game.lower() == "tictactoe":
            return TicTacToeNetwork(TicTacToeRules())
        elif game.lower() == "chess":
            return ChessNetwork(ChessRules())
