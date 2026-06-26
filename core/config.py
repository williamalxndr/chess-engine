from core.encoder import Encoder
from core.network import NetworkFactory, PolicyValueNetwork

# ------ Constant config --------------------------

LISTED_GAMES = {"chess", "tictactoe"}


# ------- Versioning interface ------------------------------

def build(game: str, version: str = "latest") -> tuple[Encoder, PolicyValueNetwork]:
    """
    Build a network from scratch
    Returns (encoder, network) pair for a given game + version.
    encoder is singleton, network is fresh instance.
    """
    network = NetworkFactory.create(game, version)
    encoder = network.encoder
    return encoder, network

def load_network(path: str = None, game: str = None, version: str = None, file_name: str = None, parent_dir: str = "checkpoints") -> PolicyValueNetwork:
    """
    Load network

    Usage:
        load(path="checkpoints/chess/V2/best.pt")
        load(game="chess", version="V2", file_name="best")
    """
    if path is not None and any(v is not None for v in [game, version, file_name]):
        raise ValueError("Use either 'path' or 'game'/'version'/'file_name', not both.")
    
    if path is None and any(v is None for v in [game, version, file_name]):
        raise ValueError("Either 'path' or all of 'game', 'version', 'file_name' must be provided.")

    return PolicyValueNetwork.load(
        path=path,
        game=game,
        version=version,
        file_name=file_name,
        parent_dir=parent_dir,
    )

if __name__ == "__main__":
    encoder, network = build("chess")
    print(type(encoder).__name__)
    print(type(network).__name__)