from abc import ABC, abstractmethod

from network import PolicyValueNetwork
from tree import NetworkMCTS
from env import TicTacToe


class Arena:
    def __init__(self, env, player_x: "Player", player_o: "Player", verbose=True):
        self.player_x = player_x
        self.player_o = player_o
        self.verbose = verbose
        self.set_env(env)
        self.results = {
            "x": 0,
            "o": 0,
            "d": 0
        }
        self.player_dict = {
            -1: self.player_x,
            1: self.player_o
        }

    def reset_result(self):
        self.results = {
            "x": 0,
            "o": 0,
            "d": 0
        }
    def reset_env(self):
        self.player_x.reset()
        self.player_o.reset()
    
    def set_env(self, env: TicTacToe):
        self.player_x.set_env(env)
        self.player_o.set_env(env)
        self.env = env

    def play(self, times):
        for _ in range(times):
            self.reset_env()

            game_over = False
            turn = -1
            while not game_over:
                current = self.player_dict[turn]
                other   = self.player_dict[-turn]

                action = current.select_action()
                game_over, result = current.advance(action, do_step=True)
                other.advance(action, do_step=False)

                self.log(f"Player {'X' if turn == -1 else 'O'} plays {action}")
                self.env.render()

                turn *= -1

            if result == -1:
                self.results["x"] += 1
            elif result == 1: 
                self.results["o"] += 1
            elif result == 0:
                self.results["d"] += 1

        self.log(f"Results: {self.results}")
        return self.results

    def log(self, msg):
        if self.verbose:
            print(msg)


class Player(ABC):
    def __init__(self, side):
        self.side = side
        super().__init__()

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
    def __init__(self, side):
        super().__init__(side)
        self.name = input(f"Welcome to Arena! You played as {'X' if self.side == -1 else 'O'}. What is your name? \n")
   
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
    def __init__(self, side, network: PolicyValueNetwork):
        super().__init__(side)
        self.mcts = NetworkMCTS(None, network)

    def reset(self):
        self.mcts.env.reset()
        self.mcts.reset()

    def set_env(self, env):
        self.mcts.set_env(env)

    def select_action(self):
        return self.mcts.search()
    
    def advance(self, action, do_step):
        return self.mcts.advance(action, do_step)
    

if __name__ == "__main__":
    env = TicTacToe()
    network = PolicyValueNetwork()
    player_1 = NetworkMCTSPlayer(-1, network)
    player_2 = HumanPlayer(1)

    arena = Arena(env, player_1, player_2)
    arena.play(1)