import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel
import traceback

from core.encoder import Encoder
from core.network import NetworkFactory, PolicyValueNetwork
from selfplay.replay_buffer import ReplayBuffer


# ── Constants ─────────────────────────────────────────────────────────────────

LISTED_GAMES = {"chess", "tictactoe"}


# ── Cache ─────────────────────────────────────────────────────────────────────

_network_cache: dict[str, PolicyValueNetwork] = {}
_buffer_cache:  dict[str, ReplayBuffer] = {}


# ── Versioning ────────────────────────────────────────────────────────────────

def get_encoder(game: str, version: str = "latest") -> Encoder:
    return Encoder.get(game=game, version=version)

def build_network(game: str, version: str = "latest") -> PolicyValueNetwork:
    """
    Build or retrieve a cached (encoder, network) pair for a game + version.
    """
    cache_key = f"{game}/{version}"

    if cache_key not in _network_cache:
        _network_cache[cache_key] = NetworkFactory.create(game, version)
    
    return _network_cache[cache_key]

def build_ddp(
    device_id:  int,
    path:       str  = None,
    game:       str  = None,
    version:    str  = None,
    file_name:  str  = None,
    parent_dir: str  = None,
):
    network = load_or_build_network(path=path, game=game, version=version, file_name=file_name, parent_dir=parent_dir).to(device_id)

    cache_key = f"{game}/{version}/{device_id}"
    _network_cache[cache_key] = DDP(network, device_ids=[device_id])

    return _network_cache[cache_key]

def load_network(
    path:       str  = None,
    game:       str  = None,
    version:    str  = None,
    file_name:  str  = None,
    parent_dir: str  = None,
) -> PolicyValueNetwork:
    if path is None and file_name is None:
        raise ValueError("Must provide 'path', or 'file_name' (with 'game'/'version'), to load a checkpoint.")

    # Not both addressing schemes at once.
    if path is not None and any(v is not None for v in [game, version, file_name]):
        raise ValueError("Use either 'path' or 'game'/'version'/'file_name', not both.")

    # Path-less load needs the full (game, version, file_name) triple.
    if path is None and any(v is None for v in [game, version, file_name]):
        raise ValueError("Either 'path' or all of 'game', 'version', 'file_name' must be provided.")

    parent_dir = "checkpoints" if parent_dir is None else parent_dir
    cache_key = path or f"{parent_dir}/{game}/{version}/{file_name}"

    if cache_key not in _network_cache:
        _network_cache[cache_key] = PolicyValueNetwork.load(path=path, game=game, version=version, file_name=file_name, parent_dir=parent_dir)

    return _network_cache[cache_key]

def load_or_build_network(
    path:       str  = None,
    game:       str  = None,
    version:    str  = None,
    file_name:  str  = None,
    parent_dir: str  = None,
) -> PolicyValueNetwork:
    if path is None and file_name is None:
        if game is None:
            raise ValueError("'game' is required.")
        return build_network(game=game, version=version or "latest")

    return load_network(path=path, game=game, version=version, file_name=file_name, parent_dir=parent_dir)


# ── Buffer ────────────────────────────────────────────────────────────────────

def build_buffer(max_size: int = 10000, seed: int = 42) -> ReplayBuffer:
    cache_key = f"{max_size}/{seed}"

    if cache_key not in _buffer_cache:
        _buffer_cache[cache_key] = ReplayBuffer(max_size=max_size, seed=seed)

    return _buffer_cache[cache_key]

def load_buffer(
    path:       str  = None,
    game:       str  = None,
    version:    str  = None,
    file_name:  str  = None,
    parent_dir: str  = None,
) -> ReplayBuffer:
    if path is None and file_name is None:
        raise ValueError("Must provide 'path', or 'file_name' (with 'game'/'version'), to load a buffer.")

    if path is not None and any(v is not None for v in [game, version, file_name]):
        raise ValueError("Use either 'path' or 'game'/'version'/'file_name', not both.")

    if path is None and any(v is None for v in [game, version, file_name]):
        raise ValueError("Either 'path' or all of 'game', 'version', 'file_name' must be provided.")

    parent_dir = "checkpoints" if parent_dir is None else parent_dir
    cache_key = path or f"{parent_dir}/{game}/{version}/{file_name}_buffer"

    if cache_key not in _buffer_cache:
        _buffer_cache[cache_key] = ReplayBuffer.load(
            path=path, game=game, version=version,
            file_name=file_name, parent_dir=parent_dir,
        )

    return _buffer_cache[cache_key]

def load_or_build_buffer(
    path:       str  = None,
    game:       str  = None,
    version:    str  = None,
    file_name:  str  = None,
    parent_dir: str  = None,
    max_size:   int  = 10000,
    seed:       int  = 42,
) -> ReplayBuffer:
    # Nothing to load from → build a fresh empty buffer.
    if path is None and file_name is None:
        return build_buffer(max_size=max_size, seed=seed)

    return load_buffer(path=path, game=game, version=version, file_name=file_name, parent_dir=parent_dir)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    network = build_network("chess")
    print(type(network).__name__)