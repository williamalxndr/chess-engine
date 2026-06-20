from abc import ABC, abstractmethod
import random

from core.network import PolicyValueNetwork
from core.tree import NetworkMCTS, VanillaMCTS
from game.rules import RULES_REGISTRY

class Player(ABC):
    def __init__(self):
        super().__init__()

    def set_side(self, side):
        self.side = side

    @abstractmethod
    def set_env(self, env):
        return NotImplemented
    
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
    def __init__(self):
        super().__init__()
        self.is_bot = False
   
    def set_env(self, env):
        self.env = env

    def reset(self):
        self.env.reset()

    def select_action(self):
        legal = False
        legal_action = self.env.get_legal_actions()

        while not legal:
            action = int(input(f"Your turn({'X' if self.side == -1 else 'O'})! Input your action({legal_action}): "))

            legal = action in legal_action

        return action

    def advance(self, action, do_step):
        if do_step:
            _, reward, terminated, _ = self.env.step(action)
            return terminated, reward
        return None

class NetworkMCTSPlayer(Player):
    def __init__(self, network: PolicyValueNetwork):
        super().__init__()
        self.mcts = NetworkMCTS(network, rules=RULES_REGISTRY["tictactoe"], epsilon=0)
        self.is_bot = True

    def reset(self):
        self.mcts.env.reset()
        self.mcts.reset()

    def set_env(self, env):
        self.mcts.set_env(env)

    def select_action(self):
        return self.mcts.search()
    
    def advance(self, action, do_step):
        return self.mcts.advance(action, do_step)
    
class VanillaMCTSPlayer(Player):
    def __init__(self):
        super().__init__()
        self.mcts = VanillaMCTS()
        self.is_bot = True

    def reset(self):
        self.mcts.env.reset()
        self.mcts.reset()

    def set_env(self, env):
        self.mcts.set_env(env)

    def advance(self, action, do_step):
        return self.mcts.advance(action, do_step)
    
    def select_action(self):
        return self.mcts.search()
    

class RandomPlayer(Player):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.env.reset()

    def set_env(self, env):
        self.env = env

    def advance(self, action, do_step):
        if do_step:
            _, reward, terminated, _ = self.env.step(action)
            return terminated, reward
        return None

    def select_action(self):
        legal_action = self.env.get_legal_actions()
        random_action = random.choice(legal_action)

        return random_action
