from abc import ABC, abstractmethod
import random

import chess

from core.network import PolicyValueNetwork
from core.tree import NetworkMCTS, VanillaMCTS
from game.rules import RULES_REGISTRY, int_to_move, move_to_int
from core.encoder import *
from game.env import Environment

class Player(ABC):
    def __init__(self):
        super().__init__()

    def set_side(self, side):
        self.side = side
    
    @abstractmethod
    def reset(self):
        return NotImplemented

    @abstractmethod
    def select_action(self):
        return NotImplemented
    
    @abstractmethod
    def advance(self):
        """
        Should return game_over (bool), result (None/int)
        """
        return NotImplemented


class HumanPlayer(Player):
    def __init__(self, env: Environment):
        super().__init__()
        self.is_bot = False
        self.env = env
   
    def reset(self):
        self.env.reset()

    def _format_legal_actions(self, legal_action, state):
        """
        Human-readable display for the legal action list. Chess gets UCI
        notation (e.g. "e2e4") via int_to_move; any other game (e.g.
        TicTacToe) falls back to the raw integers, since int_to_move only
        makes sense for chess's move-encoding scheme.
        """
        if isinstance(state, chess.Board):
            return [str(int_to_move(a, state)) for a in legal_action]
        return legal_action

    def _parse_input(self, raw, state):
        """
        Converts what the human typed back into an action int. Chess
        accepts UCI notation ("e2e4") via move_to_int; any other game
        expects a raw integer, same as before.
        """
        if isinstance(state, chess.Board):
            move = chess.Move.from_uci(raw)
            return move_to_int(move)
        return int(raw)

    def select_action(self):
        legal_action = self.env.get_legal_actions()
        state = self.env.state
        display = self._format_legal_actions(legal_action, state)

        legal = False
        while not legal:
            print(f"Legal moves: {display}")
            raw = input(f"Your turn: ")
            try:
                action = self._parse_input(raw, state)
                legal = action in legal_action
            except (ValueError, chess.InvalidMoveError):
                legal = False
            if not legal:
                print("Invalid input, try again.")

        return action

    def advance(self, action):
        _, reward, terminated, _ = self.env.step(action)
        return terminated, reward

class NetworkMCTSPlayer(Player):
    def __init__(self, network: PolicyValueNetwork, game: str):
        super().__init__()
        self.mcts = NetworkMCTS(network, epsilon=0)
        self.is_bot = True

    def reset(self):
        self.mcts.reset()

    def select_action(self):
        return self.mcts.search()
    
    def advance(self, action):
        return self.mcts.advance(action)
    
class VanillaMCTSPlayer(Player):
    def __init__(self):
        super().__init__()
        self.mcts = VanillaMCTS()
        self.is_bot = True

    def reset(self):
        self.mcts.reset()

    def advance(self, action):
        return self.mcts.advance(action)
    
    def select_action(self):
        return self.mcts.search()
    

class RandomPlayer(Player):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.env.reset()

    def advance(self, action):
        _, reward, terminated, _ = self.env.step(action)
        return terminated, reward

    def select_action(self):
        legal_action = self.env.get_legal_actions()
        random_action = random.choice(legal_action)

        return random_action