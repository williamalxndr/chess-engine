import numpy as np
import chess
import torch
from abc import ABC, abstractmethod

from core.encoder import move_to_int, int_to_move


class IllegalBoardError(Exception):
    """Raised when a state cannot occur in a legal game."""

    def __init__(self, message="Illegal board"):
        self.message = message
        super().__init__(self.message)


class Rules(ABC):
    _registry: dict[str, type["Rules"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Rules._registry[cls.__name__] = cls

    @classmethod
    def get(cls, game: str) -> "Rules":
        _GAME_PREFIX = {
            "chess":     "ChessRules",
            "tictactoe": "TicTacToeRules",
        }

        game_key = game.lower()
        if game_key not in _GAME_PREFIX:
            raise ValueError(f"Unknown game '{game}'. Available: {list(_GAME_PREFIX)}")

        key = _GAME_PREFIX[game_key]

        if key not in cls._registry:
            raise ValueError(f"Rules '{key}' not found in registry.")

        return cls._registry[key]()
    
        
    @property
    @abstractmethod
    def action_space_size(self) -> int:
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
    def get_legal_actions(self, state) -> list:
        """
        Legal actions available in `state`.

        Args:
            state: the game state to query.

        Returns:
            list[int]: the legal actions.
        """
        pass

    @abstractmethod
    def get_whose_turn(self, state) -> int:
        """
        Which player is to move in `state`.

        Args:
            state: the game state to query.

        Returns:
            int: -1 for the first player, 1 for the second.
        """
        pass

    @abstractmethod
    def is_terminal(self, state) -> bool:
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
    def transition_state(self, state, action: int):
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
    def action_space_size(self) -> int:
        return 9

    @property
    def avg_game_length(self) -> int:
        return 8

    def base_state(self):
        return np.zeros((3, 3), dtype=int)

    def get_whose_turn(self, state) -> int:
        unique, counts = np.unique(state, return_counts=True)
        counts_by_player = dict(zip(unique, counts))
        x = counts_by_player.get(-1, 0)
        o = counts_by_player.get(1, 0)
        if o == x:
            return -1
        elif o + 1 == x:
            return 1
        raise IllegalBoardError("Board is illegal: inconsistent turn counts")

    def get_legal_actions(self, state) -> list[int]:
        return [i for i in range(9) if state[divmod(i, 3)] == 0]

    def transition_state(self, state, action: int, player=None):
        if player is None:
            player = self.get_whose_turn(state)
        row, col = divmod(action, 3)
        if state[row, col] != 0:
            raise ValueError("That cell is already occupied")
        new_state = state.copy()
        new_state[row, col] = player
        return new_state

    def check_winner(self, state, player) -> bool:
        for i in range(3):
            if np.all(state[i, :] == player) or np.all(state[:, i] == player):
                return True
        if np.all(np.diag(state) == player) or np.all(np.diag(np.fliplr(state)) == player):
            return True
        return False

    def check_draw(self, state) -> bool:
        return np.all(state != 0)

    def is_terminal(self, state) -> bool:
        return self.check_winner(state, -1) or self.check_winner(state, 1) or self.check_draw(state)

    def get_result(self, state):
        if self.check_winner(state, -1): return -1
        if self.check_winner(state, 1):  return 1
        if self.check_draw(state):       return 0
        return None

    def encode(self, state):
        return torch.tensor(state[np.newaxis, :, :].astype(np.float32))


class ChessRules(Rules):
    @property
    def action_space_size(self) -> int:
        return 4672

    @property
    def avg_game_length(self) -> int:
        return 80

    def base_state(self):
        return chess.Board()

    def get_whose_turn(self, board) -> int:
        return -1 if board.turn else 1

    def get_legal_actions(self, board) -> list[int]:
        return [move_to_int(move) for move in board.legal_moves]

    def is_terminal(self, board) -> bool:
        return board.is_game_over() or board.is_repetition(3)

    def get_result(self, board):
        if board.is_repetition(3):
            return 0
        RESULT_MAP = {"1-0": -1, "0-1": 1, "1/2-1/2": 0, "*": None}
        return RESULT_MAP[board.result()]

    def transition_state(self, board, action: int, player=None):
        board_copy = board.copy(stack=8)
        move = int_to_move(action, board_copy)
        board_copy.push(move)
        return board_copy
    

if __name__ == "__main__":
    rules = Rules.get("chess")

    print(type(rules).__name__)