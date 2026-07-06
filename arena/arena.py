from abc import ABC, abstractmethod
import numpy as np
import random
import time
import threading
import uuid
import os
import chess

from core.network import PolicyValueNetwork
from core.tree import NetworkMCTS, VanillaMCTS
from game.env import TicTacToe, Chess
from arena.player import *
from core.encoder import *


class Arena:
    def __init__(self, player_1: "Player" = None, player_2: "Player" = None, num_games=1, verbose=False, pgn_dir="history"):
        self.verbose = verbose
        self.num_games = num_games
        self.pgn_dir = pgn_dir
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
            moves = []

            current, other = random.sample([self.player_1, self.player_2], 2)

            x_player = current
            o_player = other

            current.set_side(turn)
            other.set_side(-turn)

            self.log(f"Good luck! \n{current.name} moves first")
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

                move_str = str(int_to_move(action))
                moves.append(move_str)

                self.log(f"{current.name} plays {move_str}")
                self.render_board()

                current, other = other, current
                turn *= -1

            if result == -1:
                self.results[id(x_player)] += 1
                pgn_result = "1-0"
            elif result == 1:
                self.results[id(o_player)] += 1
                pgn_result = "0-1"
            elif result == 0:
                self.results["Draw"] += 1
                pgn_result = "1/2-1/2"

            if result == -1:
                result_msg = "X wins."
            elif result == 1:
                result_msg = "O wins."
            elif result == 0:
                result_msg = "Draw."
            self.log(f"Game over! {result_msg}")

            self._write_pgn(moves, x_player, o_player, pgn_result)

        return self.results

    def _to_san(self, moves):
        board = chess.Board()
        san_moves = []
        for uci in moves:
            move = chess.Move.from_uci(uci)
            san_moves.append(board.san(move))
            board.push(move)
        return san_moves

    def _write_pgn(self, moves, x_player, o_player, pgn_result):
        random_id = uuid.uuid4().hex[:8]
        os.makedirs(self.pgn_dir, exist_ok=True)
        path = os.path.join(self.pgn_dir, f"{random_id}.pgn")

        san_moves = self._to_san(moves)

        headers = [
            '[Event "Casual Game"]',
            '[Site "?"]',
            f'[Date "{time.strftime("%Y.%m.%d")}"]',
            '[Round "1"]',
            f'[White "{x_player.name}"]',
            f'[Black "{o_player.name}"]',
            f'[Result "{pgn_result}"]',
        ]

        movetext = ""
        for i in range(0, len(san_moves), 2):
            move_num = i // 2 + 1
            white_move = san_moves[i]
            black_move = san_moves[i + 1] if i + 1 < len(san_moves) else ""
            movetext += f"{move_num}. {white_move} {black_move} ".rstrip() + " "

        movetext += pgn_result

        with open(path, "w") as f:
            f.write("\n".join(headers) + "\n\n" + movetext.strip() + "\n")

        self.log(f"PGN written to {path}")

    def log(self, msg):
        if self.verbose:
            print(msg)

    def render_board(self):
        if isinstance(self.player_1, HumanPlayer):
            player_1_state = self.player_1.env.state
        elif isinstance(self.player_1, RandomPlayer):
            player_1_state = self.player_1.env.state
        elif isinstance(self.player_1, NetworkMCTSPlayer) or isinstance(self.player_1, VanillaMCTSPlayer):
            player_1_state = self.player_1.mcts.root.state

        if isinstance(self.player_2, HumanPlayer):
            player_2_state = self.player_2.env.state
        elif isinstance(self.player_2, RandomPlayer):
            player_2_state = self.player_2.env.state
        elif isinstance(self.player_2, NetworkMCTSPlayer) or isinstance(self.player_2, VanillaMCTSPlayer):
            player_2_state = self.player_2.mcts.root.state

        assert player_1_state == player_2_state

        if self.verbose:
            print(player_1_state)


if __name__ == "__main__":
    network = PolicyValueNetwork()
    net_mcts = NetworkMCTSPlayer(network, game="chess")
    env = Chess()
    random_player = RandomPlayer(env)

    arena = Arena(random_player, net_mcts, verbose=True, num_games=10)

    arena.play()