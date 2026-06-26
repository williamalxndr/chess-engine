from abc import ABC, abstractmethod
import numpy as np
import random
import time
import threading

from core.network import PolicyValueNetwork
from core.tree import NetworkMCTS, VanillaMCTS
from game.env import TicTacToe, Chess
from arena.player import *
from game.encoder import *


class Arena:
    def __init__(self, player_1: "Player" = None, player_2: "Player" = None, num_games=1, verbose=False):
        self.verbose = verbose
        self.num_games = num_games
        if player_1 is not None and player_2 is not None:
            self.set_player(player_1, player_2)

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

    def reset_game(self):
        self.player_1.reset()
        self.player_2.reset()
    
    def set_env(self, env: TicTacToe):
        self.player_1.set_env(env)
        self.player_2.set_env(env)
        self.env = env

    def _animate_thinking(self, stop_event):
        dots = ["   ", ".  ", ".. ", "..."]
        i = 0
        while not stop_event.is_set():
            print(f"\rBot is thinking{dots[i % len(dots)]}", end="", flush=True)
            i += 1
            stop_event.wait(0.4) 
        
        print("\r" + " " * 25 + "\r", end="", flush=True)
        
    def play(self):
        for _ in range(self.num_games):
            self.reset_game()

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
                if self.verbose and current.is_bot:
                    stop_event = threading.Event()
                    anim_thread = threading.Thread(target=self._animate_thinking, args=(stop_event,))
                    anim_thread.start()
                    
                    action = current.select_action()
                    
                    stop_event.set()
                    anim_thread.join()
                else:
                    action = current.select_action()

                game_over, result = current.advance(action)
                other.advance(action)

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
        if isinstance(self.player_1, HumanPlayer):
            player_1_state = self.player_1.env.state
        elif isinstance(self.player_1, NetworkMCTSPlayer) or isinstance(self.player_1, VanillaMCTSPlayer):
            player_1_state = self.player_1.mcts.root.state

        if isinstance(self.player_2, HumanPlayer):
            player_2_state = self.player_2.env.state
        elif isinstance(self.player_2, NetworkMCTSPlayer) or isinstance(self.player_2, VanillaMCTSPlayer):
            player_2_state = self.player_2.mcts.root.state

        assert player_1_state == player_2_state

        if self.verbose:
            print(player_1_state)
            

if __name__ == "__main__":
    network = PolicyValueNetwork()
    net_mcts = NetworkMCTSPlayer(network)
    vanilla_mcts = VanillaMCTSPlayer()
    human_player = HumanPlayer(env=Chess())

    arena = Arena(human_player, vanilla_mcts, verbose=True, num_games=10)
    
    arena.play()