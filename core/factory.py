import time
import torch
from torch import nn

from core.encoder import Encoder
from core.network import NetworkFactory, PolicyValueNetwork


# ── Constants ─────────────────────────────────────────────────────────────────

LISTED_GAMES = {"chess", "tictactoe"}


_network_cache: dict[str, PolicyValueNetwork] = {}


# ── Versioning ────────────────────────────────────────────────────────────────

def get_encoder(game: str, version: str = "latest") -> Encoder:
    return Encoder.get(game=game, version=version)

def build_network(game: str, version: str = "latest", dp=True) -> PolicyValueNetwork:
    """
    Build or retrieve a cached (encoder, network) pair for a game + version.
    """
    cache_key = f"{game}/{version}"

    if cache_key not in _network_cache:
        network = NetworkFactory.create(game, version)
        if torch.cuda.device_count() > 1 and dp:
            network = nn.DataParallel(network)
        _network_cache[cache_key] = network


    if cache_key not in _network_cache:
        _network_cache[cache_key] = NetworkFactory.create(game, version)
    
    return _network_cache[cache_key]

def load_network(
    path: str = None,
    game: str = None,
    version: str = None,
    file_name: str = None,
    parent_dir: str = None,
) -> PolicyValueNetwork:
    """
    Load or retrieve a cached network

    Usage:
        load(path="checkpoints/chess/V2/example.pt")
        load(game="chess", version="V2", file_name="example")
    """
    # If not load from .pt, build fresh
    if path is None and file_name is None:
        if game is None:
            raise ValueError("'game' is required.")
        return build_network(game=game, version=version or "latest")

    # Not both
    if path is not None and any(v is not None for v in [game, version, file_name]):
        raise ValueError("Use either 'path' or 'game'/'version'/'file_name', not both.")
    
    # Load
    if path is None and any(v is None for v in [game, version, file_name]):
        raise ValueError("Either 'path' or all of 'game', 'version', 'file_name' must be provided.")
    
    parent_dir = "checkpoints" if parent_dir is None else parent_dir
    cache_key = path or f"{parent_dir}/{game}/{version}/{file_name}"

    if cache_key not in _network_cache:
        _network_cache[cache_key] = PolicyValueNetwork.load(path=path, game=game, version=version, file_name=file_name, parent_dir=parent_dir)

    return _network_cache[cache_key]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    network = build_network("chess")
    print(type(network).__name__)