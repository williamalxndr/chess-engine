import numpy as np
from abc import ABC, abstractmethod
import chess
import math


class IllegalBoardError(Exception):
    def __init__(self, message="Illegal board"):
        self.message = message
        super().__init__(self.message)


class Rules(ABC):

    @property
    @abstractmethod
    def action_space_size(self):
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

    def is_legal_board(self, state):
        if not (isinstance(state, np.ndarray) and state.shape == (3, 3)):
            raise IllegalBoardError("Board has wrong type or shape")

        unique_vals, counts = np.unique(state, return_counts=True)
        freq = dict(zip(unique_vals, counts))
        x, o, empty = freq.get(-1, 0), freq.get(1, 0), freq.get(0, 0)

        if (empty + x + o) == 9 and (x == o or x - 1 == o):
            return True

        raise IllegalBoardError("Board has inconsistent piece counts")

    def encode(self, state):
        return state[np.newaxis, :, :].astype(np.float32)


class ChessRules(Rules):
    @property
    def action_space_size(self):
        return 4672

    def base_state(self):
        return chess.Board()

    def get_legal_actions(self, board):
        return [move_to_int(move) for move in board.legal_moves]

    def get_whose_turn(self, board):
        return -1 if board.turn else 1

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

    def encode(self, state: chess.Board):
        return board_to_state(state)[np.newaxis, :, :].astype(np.float32)


def board_to_state(board: chess.Board) -> np.ndarray:
    state = np.zeros((8, 8), dtype=int)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece.piece_type
            if piece.color == chess.BLACK:
                value = -value
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            state[row, col] = value
    return state


def move_to_int(move: chess.Move) -> int:
    from_sq = move.from_square; to_sq = move.to_square; promo = move.promotion
    from_file = chess.square_file(from_sq); from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq); to_rank = chess.square_rank(to_sq)
    dx = to_file - from_file; dy = to_rank - from_rank
    if promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        direction = 0 if dx == 0 else (1 if dx > 0 else 2)
        piece = 0
        if promo == chess.BISHOP: piece = 1
        elif promo == chess.ROOK: piece = 2
        plane_index = 64 + (direction * 3) + piece
    elif (abs(dx), abs(dy)) in [(1, 2), (2, 1)]:
        knight_moves = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]
        plane_index = 56 + knight_moves.index((dx, dy))
    else:
        sign_x = 0 if dx == 0 else int(math.copysign(1, dx))
        sign_y = 0 if dy == 0 else int(math.copysign(1, dy))
        queen_dir_map = {(0,1):0,(1,1):1,(1,0):2,(1,-1):3,(0,-1):4,(-1,-1):5,(-1,0):6,(-1,1):7}
        direction = queen_dir_map[(sign_x, sign_y)]
        distance = max(abs(dx), abs(dy))
        plane_index = (direction * 7) + (distance - 1)
    return (from_sq * 73) + plane_index


def int_to_move(action: int, board: chess.Board = None) -> chess.Move:
    from_sq = action // 73; plane_index = action % 73
    from_file = chess.square_file(from_sq); from_rank = chess.square_rank(from_sq)
    promo = None
    if plane_index < 56:
        direction = plane_index // 7; distance = (plane_index % 7) + 1
        dir_map = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        dx, dy = dir_map[direction]
        to_file = from_file + (dx*distance); to_rank = from_rank + (dy*distance)
        if board and board.piece_at(from_sq):
            if board.piece_at(from_sq).piece_type == chess.PAWN:
                if to_rank == 0 or to_rank == 7:
                    promo = chess.QUEEN
    elif plane_index < 64:
        knight_moves = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]
        dx, dy = knight_moves[plane_index - 56]
        to_file = from_file + dx; to_rank = from_rank + dy
    else:
        under_idx = plane_index - 64; direction = under_idx // 3; piece = under_idx % 3
        promo_map = {0: chess.KNIGHT, 1: chess.BISHOP, 2: chess.ROOK}
        promo = promo_map[piece]
        dx = 0 if direction == 0 else (1 if direction == 1 else -1)
        dy = 1 if from_rank == 6 else -1
        to_file = from_file + dx; to_rank = from_rank + dy
    to_sq = chess.square(to_file, to_rank)
    return chess.Move(from_sq, to_sq, promotion=promo)


RULES_REGISTRY = {"tictactoe": TicTacToeRules(), "chess": ChessRules()}