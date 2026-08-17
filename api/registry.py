from api.config import LISTED_GAMES
from api.service import GameService, GameServiceFactory

_services: dict[str, GameService] = {}


def get_service(game: str) -> GameService:
    """
    Services are built on first request, not at import, so importing the app
    does not load a checkpoint.

    Raises:
        KeyError: if `game` is not listed.
    """
    key = game.lower()

    if key not in LISTED_GAMES:
        raise KeyError(game)

    if key not in _services:
        _services[key] = GameServiceFactory.make(key)

    return _services[key]
