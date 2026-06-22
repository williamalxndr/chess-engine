import numpy as np
import chess
import math
from abc import ABC, abstractmethod

from game.encoder import *


class IllegalBoardError(Exception):
    """Raised when a state cannot occur in a legal game."""

    def __init__(self, message="Illegal board"):
        self.message = message
        super().__init__(self.message)


class Rules(ABC):
    """
    Stateless game mechanics: legal moves, transitions, results, and
    state encoding. Subclasses implement one specific game.
    """

    @property
    @abstractmethod
    def action_space_size(self):
        """
        Total number of distinct actions in this game.

        Returns:
            int: size of the action space.
        """
        pass

    @property
    @abstractmethod
    def encoded_channels(self):
        """
        Number of channels produced by encode().

        Returns:
            int: channel count of the encoded state.
        """
        pass

    @abstractmethod
    def base_state(self):
        """
        Return the starting state of the game.

        Returns:
            state: the initial game state.
        """
        pass

    @abstractmethod
    def get_legal_actions(self, state):
        """
        Legal actions available in `state`.

        Args:
            state: the game state to query.

        Returns:
            list[int]: the legal actions.
        """
        pass

    @abstractmethod
    def get_whose_turn(self, state):
        """
        Which player is to move in `state`.

        Args:
            state: the game state to query.

        Returns:
            int: -1 for the first player, 1 for the second.
        """
        pass

    @abstractmethod
    def is_terminal(self, state):
        """
        Whether `state` ends the game.

        Args:
            state: the game state to query.

        Returns:
            bool: True if the game is over.
        """
        pass

    @abstractmethod
    def get_result(self, state):
        """
        Outcome of `state` from a fixed perspective.

        Args:
            state: the game state to query.

        Returns:
            int: -1 if the first player won, 1 if the second won, 0 for a
                draw, or None if the game is not over.
        """
        pass

    @abstractmethod
    def transition_state(self, state, action):
        """
        Return the new state after applying `action` to `state`.

        Args:
            state: the current game state.
            action (int): the action to apply.

        Returns:
            state: the resulting game state (the input is not mutated).
        """
        pass

    @abstractmethod
    def encode(self, state) -> np.ndarray:
        """
        Encode `state` into a tensor the network can consume.

        Args:
            state: the game state to encode.

        Returns:
            np.ndarray: encoded state of shape (encoded_channels, H, W).
        """
        pass

    def rollout(self, state):
        """
        Play uniformly random moves from `state` until the game ends.

        Args:
            state: the game state to roll out from (copied, not mutated).

        Returns:
            int: the terminal result of the finished game.
        """
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
        """
        Returns:
            np.ndarray: empty 3x3 board of ints (0 = empty).
        """
        return np.zeros((3, 3), dtype=int)

    def get_whose_turn(self, state):
        """
        Infer the player to move from the piece counts.

        Args:
            state (np.ndarray): the 3x3 board.

        Returns:
            int: -1 for X, 1 for O.

        Raises:
            IllegalBoardError: if the piece counts cannot occur in a real game.
        """
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
        """
        Args:
            state (np.ndarray): the 3x3 board.

        Returns:
            list[int]: indices (0-8) of the empty cells.
        """
        return [i for i in range(9) if state[divmod(i, 3)] == 0]

    def transition_state(self, state, action, player=None):
        """
        Place the current player's mark and return the new board.

        Args:
            state (np.ndarray): the 3x3 board.
            action (int): cell index (0-8) to play.
            player (int): mark to place, or None to use the player to move.

        Returns:
            np.ndarray: a copy of the board with `action` played.

        Raises:
            ValueError: if the target cell is already occupied.
        """
        if player is None:
            player = self.get_whose_turn(state)
        row, col = divmod(action, 3)
        if state[row, col] != 0:
            raise ValueError("That cell is already occupied")
        new_state = state.copy()
        new_state[row, col] = player
        return new_state

    def check_winner(self, state, player):
        """
        Whether `player` has three in a row.

        Args:
            state (np.ndarray): the 3x3 board.
            player (int): the player to check (-1 or 1).

        Returns:
            bool: True if that player has won.
        """
        for i in range(3):
            if np.all(state[i, :] == player) or np.all(state[:, i] == player):
                return True
        if np.all(np.diag(state) == player) or np.all(np.diag(np.fliplr(state)) == player):
            return True
        return False

    def check_draw(self, state):
        """
        Whether the board is full (no empty cells).

        Args:
            state (np.ndarray): the 3x3 board.

        Returns:
            bool: True if the board is full.
        """
        return np.all(state != 0)

    def is_terminal(self, state):
        """
        Args:
            state (np.ndarray): the 3x3 board.

        Returns:
            bool: True if either player has won or the board is full.
        """
        return self.check_winner(state, -1) or self.check_winner(state, 1) or self.check_draw(state)

    def get_result(self, state):
        """
        Args:
            state (np.ndarray): the 3x3 board.

        Returns:
            int: -1 if X won, 1 if O won, 0 for a draw, or None if unfinished.
        """
        if self.check_winner(state, -1):
            return -1
        if self.check_winner(state, 1):
            return 1
        if self.check_draw(state):
            return 0
        return None

    def encode(self, state):
        """
        Args:
            state (np.ndarray): the 3x3 board.

        Returns:
            np.ndarray: the board as float32 with shape (1, 3, 3).
        """
        return state[np.newaxis, :, :].astype(np.float32)


class ChessRules(Rules):
    @property
    def action_space_size(self):
        return 4672

    @property
    def encoded_channels(self):
        return 21

    def base_state(self):
        """
        Returns:
            chess.Board: a board in the standard starting position.
        """
        return chess.Board()

    def get_whose_turn(self, board):
        """
        Args:
            board (chess.Board): the current board.

        Returns:
            int: -1 if White is to move, 1 if Black is to move.
        """
        return -1 if board.turn else 1

    def get_legal_actions(self, board):
        """
        Args:
            board (chess.Board): the current board.

        Returns:
            list[int]: legal moves encoded as integer action indices.
        """
        return [move_to_int(move) for move in board.legal_moves]

    def is_terminal(self, board):
        """
        Args:
            board (chess.Board): the current board.

        Returns:
            bool: True if the game is over (mate, stalemate, draw, etc.).
        """
        return board.is_game_over()

    def get_result(self, board):
        """
        Args:
            board (chess.Board): the current board.

        Returns:
            int: -1 if White won, 1 if Black won, 0 for a draw, or None if
                the game is not over.
        """
        RESULT_MAP = {"1-0": -1, "0-1": 1, "1/2-1/2": 0, "*": None}
        return RESULT_MAP[board.result()]

    def transition_state(self, board, action: int, player=None):
        """
        Push the move for `action` and return the new board.

        Args:
            board (chess.Board): the current board.
            action (int): the integer action index of the move to play.
            player (int): unused; kept for interface compatibility.

        Returns:
            chess.Board: a copy of the board with the move played.

        Raises:
            ValueError: if the action is not legal on this board.
        """
        if action not in self.get_legal_actions(board):
            raise ValueError("Illegal action")
        board_copy = board.copy()
        move = int_to_move(action, board_copy)
        board_copy.push(move)
        return board_copy

    def encode(self, state):
        """
        Args:
            state (chess.Board): the board to encode.

        Returns:
            np.ndarray: float32 tensor of shape (21, 8, 8).
        """
        return encode_chess_state(state).astype(np.float32)


RULES_REGISTRY = {"tictactoe": TicTacToeRules(), "chess": ChessRules()}
