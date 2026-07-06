from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import re

from core.node import Node
from core.utils import resolve_latest
from game.rules import Rules
from core.encoder import Encoder

class ValueHead(nn.Module):
    def __init__(self, in_channels, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(in_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 73, kernel_size=1)  

    def forward(self, x):                   # (N, in_channels, 8, 8)
        x = self.conv(x)                    # (N, 73, 8, 8)
        x = x.flip(dims=[2])                # (N, 73, 8, 8), flips the board vertically (rank mirror)
        x = x.permute(0, 2, 3, 1)           # (N, 8, 8, 73)  -> (N, rank, file, plane)
        return x.reshape(x.size(0), -1)     # (N, 4672), index = rank*8*73 + file*73 + plane_idx == from_sq*73 + plane_idx
    

class PolicyValueNetwork(nn.Module, ABC):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        PolicyValueNetwork._registry[cls.__name__] = cls

    def __init__(self, rules: Rules, encoder=None):
        super().__init__()
        self.rules             = rules
        self.encoder           = encoder
        self.action_space_size = rules.action_space_size
        self.in_channels       = encoder.channels
        self.body              = self._build_body()
        self.device            = torch.device(
            torch.accelerator.current_accelerator()
            if torch.accelerator.is_available() else "cpu"
        )

        self.policy_head = PolicyHead(self.body_channels).to(self.device)
        self.value_head = ValueHead(self.body_channels).to(self.device)
        self.to(self.device)

    @abstractmethod
    def _build_body(self) -> nn.Sequential:
        pass

    @property
    @abstractmethod
    def body_channels(self) -> int:
        pass

    def forward(self, x: torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        features = self.body(x)
        return self.policy_head(features), self.value_head(features)

    def forward_nodes(self, nodes: list[Node]):
        encode = self.encoder.encode
        batch  = torch.stack([encode(n.state) for n in nodes])
        return self.forward(batch)

    def save(self, game: str = "chess", version: str = "v2", file_name: str = "example", parent_dir: str = "checkpoints", path: str=None):
        """Save weights + metadata for load()."""
        file_path = Path(path) if path else Path(f"{parent_dir}/{game}/{version}") / f"{file_name}.pt"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict":  self.state_dict(),
            "in_channels": self.in_channels,
            "class_name":  self.__class__.__name__,
            "game":        game,
            "version":     version,
        }, file_path)

        print(f"Network saved at {file_path}")

    @staticmethod
    def load(game: str = None, version: str = None, file_name: str=None, parent_dir: str = "checkpoints", path: str = None) -> "PolicyValueNetwork":
        """Reconstruct network from checkpoint via config."""        
        path       = path or f"{parent_dir}/{game}/{version}/{file_name}.pt"
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        game       = checkpoint.get("game", game)
        version    = checkpoint.get("version", version)
        network    = NetworkFactory.create(game=game, version=version)
        network.load_state_dict(checkpoint["state_dict"])

        return network

    @staticmethod
    def create(game: str, version: str = "latest") -> "PolicyValueNetwork":
        return NetworkFactory.create(game, version)


# ── Chess ─────────────────────────────────────────────────────────────────────

class ChessNetwork(PolicyValueNetwork):
    encoder = None
    game="chess"

    def __init__(self):
        super().__init__(rules=Rules.get("chess"), encoder=self.__class__.encoder)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not cls.__name__.startswith("ChessNetwork"):
            raise ValueError(f"Class name must start with 'ChessNetwork', got '{cls.__name__}'")

        # Extract version suffix 
        version = cls.__name__[len("ChessNetwork"):]
        if not version:
            return 

        cls.version = version 
        cls.encoder = Encoder.get(game="chess", version=version)


class ChessNetworkV1(ChessNetwork):
    """10-block residual network, 21 input channels (ChessEncoderV1)."""

    def __init__(self):
        super().__init__()

    def _build_body(self) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(21,  32,  padding=1),
            ResidualBlock(32,  64,  padding=1),
            ResidualBlock(64,  128, padding=1),
            ResidualBlock(128, 256, padding=1),
            ResidualBlock(256, 256, padding=1),
            ResidualBlock(256, 256, padding=1),
            ResidualBlock(256, 256, padding=1),
            ResidualBlock(256, 256, padding=1),
            ResidualBlock(256, 64,  padding=1),
            ResidualBlock(64,  32,  padding=1),
        )

    @property
    def body_channels(self) -> int:
        return 32


class ChessNetworkV2(ChessNetwork):
    """20-block uniform residual network, 30 input channels (ChessEncoderV2). ~24.3M params."""

    def __init__(self):
        super().__init__()

    def _build_body(self) -> nn.Sequential:
        layers = [ResidualBlock(30, 256, padding=1)]
        for _ in range(19):
            layers.append(ResidualBlock(256, 256, padding=1))
        return nn.Sequential(*layers)

    @property
    def body_channels(self) -> int:
        return 256


# ── TicTacToe ─────────────────────────────────────────────────────────────────

class TicTacToeNetwork(PolicyValueNetwork):
    """Small residual network for TicTacToe."""

    def __init__(self, rules, encoder=None):
        super().__init__(rules, encoder)

    def _build_body(self) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(self.in_channels, 8, kernel_size=2),
            ResidualBlock(8, 16, kernel_size=1),
        )

    @property
    def body_channels(self) -> int:
        return 16


# ── Residual block ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if x.shape[-2:] != out.shape[-2:]:
            x = F.adaptive_avg_pool2d(x, out.shape[-2:])
        if self.in_channels != self.out_channels:
            x = self.channel_proj(x)
        return self.relu2(out + x)


# ── Factory ────────────────────────────────────────────────────────────────────

class NetworkFactory:
    _GAME_PREFIX: dict[str, str] = {
        "chess": "ChessNetwork",
        "tictactoe": "TicTacToeNetwork",
    }

    @classmethod
    def create(cls, game: str, version: str = "latest") -> PolicyValueNetwork:
        version = "latest" if version is None else version

        game_key = game.lower()

        if game_key not in cls._GAME_PREFIX:
            raise ValueError(f"Unknown game '{game}'. Available: {list(cls._GAME_PREFIX)}")

        prefix = cls._GAME_PREFIX[game_key]

        if version.lower() == "latest":
            resolved = resolve_latest(prefix, PolicyValueNetwork._registry)
        elif version.startswith(prefix):
            resolved = version
        else:
            v = version.upper() if version.upper().startswith("V") else f"V{version}"
            resolved = f"{prefix}{v}"

        if resolved not in PolicyValueNetwork._registry:
            available = [k for k in PolicyValueNetwork._registry if k.startswith(prefix)]
            raise ValueError(f"Network '{resolved}' not found. Available for '{game}': {available}")

        return PolicyValueNetwork._registry[resolved]()
    

if __name__ == "__main__": 
    obj = NetworkFactory.create("chess")
    print(type(obj).__name__)