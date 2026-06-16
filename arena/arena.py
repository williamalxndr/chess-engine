from abc import ABC, abstractmethod
import numpy as np
import random

from core.network import PolicyValueNetwork
from core.tree import NetworkMCTS, VanillaMCTS
from game.env import TicTacToe


class Arena:
    def __init__(self, env: TicTacToe = None, player_1: "Player" = None, player_2: "Player" = None, verbose=False):
        self.verbose = verbose
        if env is None:
            self.env = TicTacToe()
        if player_1 is not None and player_2 is not None:
            self.set_player(player_1, player_2)
            self.set_env(env)

    def set_player(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.reset_result()

    def reset_result(self):
        self.results = {
            id(self.player_1): 0,
            id(self.player_2): 0,
            "Draw": 0
        }

    def reset_env(self):
        self.player_1.reset()
        self.player_2.reset()
    
    def set_env(self, env: TicTacToe):
        self.player_1.set_env(env)
        self.player_2.set_env(env)
        self.env = env

    def play(self, times):
        for _ in range(times):
            self.reset_env()

            game_over = False
            turn = -1

            current, other = random.sample([self.player_1, self.player_2], 2)
            x_player = current
            o_player = other

            x_player.set_side(turn)
            o_player.set_side(-turn)

            while not game_over:
                action = current.select_action()
                game_over, result = current.advance(action, do_step=True)
                other.advance(action, do_step=False)

                self.log(f"Player {'X' if turn == -1 else 'O'} plays {action}")
                if self.verbose:
                    self.env.render()

                current, other = other, current
                turn *= -1

            if result == -1:
                self.results[id(x_player)] += 1
            elif result == 1: 
                self.results[id(o_player)] += 1
            elif result == 0:
                self.results["Draw"] += 1
            
            if result == -1:
                result_msg = "X wins."
            elif result == 1:
                result_msg = "O wins."
            elif result == 0:
                result_msg = "Draw."
            self.log(f"Game over! {result_msg}")

        self.log(f"Results: {self.results}")
        return self.results

    def log(self, msg):
        if self.verbose:
            print(msg)


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
   
    def set_env(self, env):
        self.env = env

    def reset(self):
        self.env.reset()

    def select_action(self):
        legal = False
        legal_action = self.env.get_legal_actions_self()

        while not legal:
            action = int(input(f"{'X' if self.side == -1 else 'O'}'s turn! Input your action({legal_action}): "))

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
        self.mcts = NetworkMCTS(network)

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

    def reset(self):
        self.mcts.env.reset()
        self.mcts.reset()

    def set_env(self, env):
        self.mcts.set_env(env)

    def advance(self, action, do_step):
        return self.mcts.advance(action, do_step)
    
    def select_action(self):
        return self.mcts.search()
    


    

if __name__ == "__main__":
    env = TicTacToe()
    network = PolicyValueNetwork()
    net_mcts = NetworkMCTSPlayer(network)
    vanilla_mcts = VanillaMCTSPlayer()
    human_player = HumanPlayer()

    arena = Arena(env, human_player, vanilla_mcts)
    arena.play(10)