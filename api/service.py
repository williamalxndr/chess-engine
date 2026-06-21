from api import config
from game.env import TicTacToe
from core.network import *
from core.tree import *
from abc import ABC, abstractmethod

import numpy as np

class GameService(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_bot_move(self, board):
        """
        ## Args:
            board(list)

        ## Returns:
            dict of move(int), game_over(int), result(int)
        """
        return NotImplemented


class TicTacToeService(GameService):
    def __init__(self, network_path=config.DEFAULT_NETWORK_PATH, num_rollout=config.NUM_ROLLOUT):
        super().__init__()
        self.network = PolicyValueNetwork.load(network_path)
        self.bot = NetworkMCTS(self.network, num_rollout=num_rollout, seed=config.SEED)

    def check_terminal(self, board):
        return TicTacToe.is_terminal(board)

    def get_bot_move(self, board):
        board = np.array(board)
    
        if self.check_terminal(board):
            move = None
            game_over = True
            result = TicTacToe.get_result(board)
        else:
            self.bot.reset(board)
            move = self.bot.search()
            game_over, result = self.bot.advance(move)
        
        return {
            "move": move,
            "game_over": game_over, 
            "result": result
            }

class GameServiceFactory:
    @staticmethod
    def make(game: str) -> GameService:
        if game.lower() == "tictactoe":
            return TicTacToeService()




