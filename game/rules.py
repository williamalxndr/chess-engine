import numpy as np
import chess
import math
from abc import ABC, abstractmethod

from game.helper import *


class IllegalBoardError(Exception):
    def __init__(self, message="Illegal board"):
        self.message = message
        super().__init__(self.message)


class Rules(ABC):
    @property
    @abstractmethod
    def action_space_size(self):
        pass

    @property
    @abstractmethod
    def encoded_channels(self):
        pass

    @abstractmethod
    def base_state(self):
        pass

    @abstractmethod
    def get_legal_actions(self, state):
        pass

    @abstractmethod
    def get_whose_turn(self, state):
        pass

    @abstractmethod
    def is_terminal(self, state):
        pass

    @abstractmethod
    def get_result(self, state):
        pass

    @abstractmethod
    def transition_state(self, state, action):
        pass

    @abstractmethod
    def encode(self, state) -> np.ndarray:
        pass

    def rollout(self, state):
        state = state.copy()
        while not self.is_terminal(state):
            legal = self.get_legal_actions(state)
            action = np.random.choice(legal)
            state = self.transition_state(state, action)
        return self.get_result(state)


class TicTacToeRules(Rules):
    @property
    def action_space_size(self):
        return 9

    @property
    def encoded_channels(self):
        return 1

    def base_state(self):
        return np.zeros((3, 3), dtype=int)

    def get_whose_turn(self, state):
        unique, counts = np.unique(state, return_counts=True)
        counts_by_player = dict(zip(unique, counts))
        x = counts_by_player.get(-1, 0)
        o = counts_by_player.get(1, 0)
        if o == x:
            return -1
        elif o + 1 == x:
            return 1
        raise IllegalBoardError("Board is illegal: inconsistent turn counts")

    def get_legal_actions(self, state):
        return [i for i in range(9) if state[divmod(i, 3)] == 0]

    def transition_state(self, state, action, player=None):
        if player is None:
            player = self.get_whose_turn(state)
        row, col = divmod(action, 3)
        if state[row, col] != 0:
            raise ValueError("That cell is already occupied")
        new_state = state.copy()
        new_state[row, col] = player
        return new_state

    def check_winner(self, state, player):
        for i in range(3):
            if np.all(state[i, :] == player) or np.all(state[:, i] == player):
                return True
        if np.all(np.diag(state) == player) or np.all(np.diag(np.fliplr(state)) == player):
            return True
        return False

    def check_draw(self, state):
        return np.all(state != 0)

    def is_terminal(self, state):
        return self.check_winner(state, -1) or self.check_winner(state, 1) or self.check_draw(state)

    def get_result(self, state):
        if self.check_winner(state, -1):
            return -1
        if self.check_winner(state, 1):
            return 1
        if self.check_draw(state):
            return 0
        return None

    def encode(self, state):
        return state[np.newaxis, :, :].astype(np.float32)


class ChessRules(Rules):
    @property
    def action_space_size(self):
        return 4672

    @property
    def encoded_channels(self):
        return 21

    def base_state(self):
        return chess.Board()

    def get_whose_turn(self, board):
        return -1 if board.turn else 1

    def get_legal_actions(self, board):
        return [move_to_int(move) for move in board.legal_moves]

    def is_terminal(self, board):
        return board.is_game_over()

    def get_result(self, board):
        RESULT_MAP = {"1-0": -1, "0-1": 1, "1/2-1/2": 0, "*": None}
        return RESULT_MAP[board.result()]

    def transition_state(self, board, action: int, player=None):
        if action not in self.get_legal_actions(board):
            raise ValueError("Illegal action")
        board_copy = board.copy()
        move = int_to_move(action, board_copy)
        board_copy.push(move)
        return board_copy

    def encode(self, state):
        return encode_chess_state(state).astype(np.float32)


RULES_REGISTRY = {"tictactoe": TicTacToeRules(), "chess": ChessRules()}