from core.encoder import Encoder
from core.network import NetworkFactory, PolicyValueNetwork


# ── Constants ─────────────────────────────────────────────────────────────────

LISTED_GAMES = {"chess", "tictactoe"}


_network_cache: dict[str, PolicyValueNetwork] = {}


# ── Versioning ────────────────────────────────────────────────────────────────

def build(game: str, version: str = "latest") -> tuple[Encoder, PolicyValueNetwork]:
    """
    Build or retrieve a cached (encoder, network) pair for a game + version.
    """
    cache_key = f"{game}/{version}"

    if cache_key not in _network_cache:
        _network_cache[cache_key] = NetworkFactory.create(game, version)

    network = _network_cache[cache_key]
    return network.encoder, network

def load(
    path: str = None,
    game: str = None,
    version: str = None,
    file_name: str = None,
    parent_dir: str = "checkpoints",
) -> tuple[Encoder, PolicyValueNetwork]:
    """
    Load or retrieve a cached (encoder, network) pair from a checkpoint.

    Usage:
        load(path="checkpoints/chess/V2/example.pt")
        load(game="chess", version="V2", file_name="example")
    """
    if path is not None and any(v is not None for v in [game, version, file_name]):
        raise ValueError("Use either 'path' or 'game'/'version'/'file_name', not both.")

    if path is None and any(v is None for v in [game, version, file_name]):
        raise ValueError("Either 'path' or all of 'game', 'version', 'file_name' must be provided.")

    cache_key = path or f"{parent_dir}/{game}/{version}/{file_name}"

    if cache_key not in _network_cache:
        _network_cache[cache_key] = PolicyValueNetwork.load(path=path, game=game, version=version, file_name=file_name, parent_dir=parent_dir)

    network = _network_cache[cache_key]

    return network.encoder, network


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    encoder, network = build("chess")
    print(type(encoder).__name__)
    print(type(network).__name__)