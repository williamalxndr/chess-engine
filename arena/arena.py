from abc import ABC, abstractmethod
import numpy as np
import random
import time
import threading

from core.network import PolicyValueNetwork
from core.tree import NetworkMCTS, VanillaMCTS
from game.env import TicTacToe
from arena.player import *
from game.helper import *


class Arena:
    def __init__(self, env: TicTacToe = None, player_1: "Player" = None, player_2: "Player" = None, delay=5, num_games=1, verbose=False):
        self.verbose = verbose
        self.delay = delay
        self.num_games = num_games
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

    def sleep(self, current):
        if not self.verbose or not current.is_bot:
            return
        
        dots = ["   ", ".  ", ".. ", "..."]
        i = 0
        start = time.time()
        while (time.time() - start) < self.delay:
            print(f"\rBot is thinking{dots[i % len(dots)]}", end="", flush=True)
            i += 1
            time.sleep(0.4)
        
        print("\r" + " " * 25 + "\r", end="", flush=True)
        
    def play(self):
        for _ in range(self.num_games):
            self.reset_env()

            game_over = False
            turn = -1

            current, other = random.sample([self.player_1, self.player_2], 2)

            x_player = current
            o_player = other

            current.set_side(turn)
            other.set_side(-turn)

            self.log(f"Good luck! \n{'Bot' if current.is_bot else 'Human'} moves first")
            self.render_board()

            while not game_over:
                self.sleep(current)
                action = current.select_action()

                game_over, result = current.advance(action, do_step=True)
                other.advance(action, do_step=False)

                self.log(f"{'Bot' if current.is_bot else 'Human'} plays {str(int_to_move(action))}")
                self.render_board()

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

        return self.results

    def log(self, msg):
        if self.verbose:
            print(msg)

    def render_board(self):
        if self.verbose:
            self.env.render()


if __name__ == "__main__":
    env = TicTacToe()
    network = PolicyValueNetwork()
    net_mcts = NetworkMCTSPlayer(network)
    vanilla_mcts = VanillaMCTSPlayer()
    human_player = HumanPlayer()

    arena = Arena(env, human_player, vanilla_mcts)
    arena.play(10)