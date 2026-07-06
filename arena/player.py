from abc import ABC, abstractmethod
import random

import chess

from core.network import PolicyValueNetwork
from core.tree import NetworkMCTS, VanillaMCTS
from game.rules import int_to_move, move_to_int
from core.encoder import *
from game.env import Environment

class Player(ABC):
    def __init__(self, name=None):
        super().__init__()
        self.name = name or self.__class__.__name__

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
        return NotImplemented


class HumanPlayer(Player):
    def __init__(self, env: Environment, name="Human"):
        super().__init__(name=name)
        self.is_bot = False
        self.env = env

    def reset(self):
        self.env.reset()

    def _format_legal_actions(self, legal_action, state):
        if isinstance(state, chess.Board):
            return [str(int_to_move(a, state)) for a in legal_action]
        return legal_action

    def _parse_input(self, raw, state):
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
    def __init__(self, network: PolicyValueNetwork, game: str, num_rollout=2000, name="NetworkMCTS"):
        super().__init__(name=name)
        self.mcts = NetworkMCTS(game=game, num_rollout=num_rollout, network=network, epsilon=0)
        self.is_bot = True

    def reset(self):
        self.mcts.reset()

    def select_action(self):
        return self.mcts.search()

    def advance(self, action):
        return self.mcts.advance(action)

class VanillaMCTSPlayer(Player):
    def __init__(self, name="VanillaMCTS"):
        super().__init__(name=name)
        self.mcts = VanillaMCTS()
        self.is_bot = True

    def reset(self):
        self.mcts.reset()

    def advance(self, action):
        return self.mcts.advance(action)

    def select_action(self):
        return self.mcts.search()


class RandomPlayer(Player):
    def __init__(self, env: Environment, name="Random"):
        super().__init__(name=name)
        self.is_bot = True
        self.env = env

    def reset(self):
        self.env.reset()

    def advance(self, action):
        _, reward, terminated, _ = self.env.step(action)
        return terminated, reward

    def select_action(self):
        legal_action = self.env.get_legal_actions()
        return random.choice(legal_action)