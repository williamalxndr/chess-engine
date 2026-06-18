from api.service import GameServiceFactory
from api.config import LISTED_GAMES


GAME_REGISTRY = { name: GameServiceFactory.make(name) for name in LISTED_GAMES }
