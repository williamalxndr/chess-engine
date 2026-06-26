import chess
import math
import numpy as np
import torch
from abc import ABC, abstractmethod

from core.utils import resolve_latest


device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

from abc import ABC, abstractmethod
import torch


class Encoder(ABC):
    _registry: dict[str, type["Encoder"]] = {}   # class registry
    _instances: dict[str, "Encoder"] = {}        # singleton cache

    _GAME_PREFIX: dict[str, str] = {
        "chess":     "ChessEncoder",
        "tictactoe": "TicTacToeEncoder",
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Encoder._registry[cls.__name__] = cls

    @classmethod
    def get(cls, game: str, version: str = "latest") -> "Encoder":
        game_key = game.lower()
        if game_key not in cls._GAME_PREFIX:
            raise ValueError(f"Unknown game '{game}'. Available: {list(cls._GAME_PREFIX)}")

        prefix = cls._GAME_PREFIX[game_key]

        if version.lower() == "latest":
            key = resolve_latest(prefix, cls._registry)
        elif version.startswith(prefix):
            key = version
        else:
            v = version.upper() if version.upper().startswith("V") else f"V{version}"
            key = f"{prefix}{v}"

        if key not in cls._registry:
            available = [k for k in cls._registry if k.startswith(prefix)]
            raise ValueError(f"Unknown encoder '{key}'. Available: {available}")

        if key not in cls._instances:
            cls._instances[key] = cls._registry[key]()

        return cls._instances[key]

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @abstractmethod
    def encode(self, state) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, state): ...

class TicTacToeEncoder(Encoder):
    pass

class ChessEncoder(Encoder):

    # ── encode helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _piece_planes(board: chess.Board, state: np.ndarray, offset: int = 0):
        """Fill channels [offset, offset+12) with piece bitboards."""
        for piece_type in chess.PIECE_TYPES:
            for color in [chess.WHITE, chess.BLACK]:
                channel = offset + (piece_type - 1) + (0 if color == chess.WHITE else 6)
                bb = int(board.pieces(piece_type, color))
                if bb == 0:
                    continue
                arr = np.unpackbits(
                    np.array([bb], dtype=np.uint64).view(np.uint8),
                    bitorder='little'
                ).reshape(8, 8)
                state[channel] = arr[::-1]

    @staticmethod
    def _ep_plane(board: chess.Board, state: np.ndarray, channel: int):
        if board.ep_square is not None:
            state[channel,
                  7 - chess.square_rank(board.ep_square),
                  chess.square_file(board.ep_square)] = 1.0

    @staticmethod
    def _castling_planes(board: chess.Board, state: np.ndarray, offset: int):
        """Fill 4 channels: W O-O, W O-O-O, B O-O, B O-O-O."""
        state[offset + 0, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        state[offset + 1, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        state[offset + 2, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        state[offset + 3, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    # ── decode helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _decode_base(state: np.ndarray) -> chess.Board:
        """Reconstruct pieces, ep, castling, turn from channels 0-17."""
        board = chess.Board(fen=None)
        board.clear()
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        for channel in range(12):
            piece_type = piece_types[channel % 6]
            color = chess.WHITE if channel < 6 else chess.BLACK
            for row in range(8):
                for col in range(8):
                    if state[channel, row, col] == 1.0:
                        board.set_piece_at(chess.square(col, 7 - row), chess.Piece(piece_type, color))
        board.ep_square = None
        for row in range(8):
            for col in range(8):
                if state[12, row, col] == 1.0:
                    board.ep_square = chess.square(col, 7 - row)
                    break
        castling = 0
        if state[13, 0, 0] == 1.0: castling |= chess.BB_H1
        if state[14, 0, 0] == 1.0: castling |= chess.BB_A1
        if state[15, 0, 0] == 1.0: castling |= chess.BB_H8
        if state[16, 0, 0] == 1.0: castling |= chess.BB_A8
        board.castling_rights = castling
        board.turn = chess.WHITE if state[17, 0, 0] == 1.0 else chess.BLACK
        return board

    @staticmethod
    def _to_numpy(state) -> np.ndarray:
        if isinstance(state, torch.Tensor):
            return state.cpu().numpy()
        return state

    # ── feature helpers ─────────────────────────────────────────────────────

    PIECE_VALUES = {
        chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.5,
        chess.ROOK: 5.0, chess.QUEEN: 9.0,
    }
    MAX_MATERIAL = 39.0
    MAX_LEGAL    = 218.0

    @staticmethod
    def _king_zone_squares(board: chess.Board, color: chess.Color) -> list[int]:
        """Squares in the 3x3 area around the king of `color`."""
        king_sq = board.king(color)
        if king_sq is None:
            return []
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        return [
            chess.square(kf + df, kr + dr)
            for df in [-1, 0, 1] for dr in [-1, 0, 1]
            if 0 <= kf + df <= 7 and 0 <= kr + dr <= 7
        ]

    @classmethod
    def _attacked_material_value(cls, board: chess.Board, attacker_color: chess.Color, target_color: chess.Color) -> float:
        """Total value of target_color pieces attacked by attacker_color (king excluded)."""
        return sum(
            value
            for piece_type, value in cls.PIECE_VALUES.items()
            for sq in board.pieces(piece_type, target_color)
            if board.is_attacked_by(attacker_color, sq)
        )

    @classmethod
    def _pieces_around_king(cls, board: chess.Board, king_color: chess.Color, piece_color: chess.Color) -> int:
        """Count piece_color pieces within the 3x3 zone around king_color's king."""
        zone = set(cls._king_zone_squares(board, king_color))
        return sum(
            1 for sq in zone
            if (p := board.piece_at(sq)) is not None and p.color == piece_color
        )

    @classmethod
    def _king_zone_attack_value(cls, board: chess.Board, king_color: chess.Color, attacker_color: chess.Color) -> float:
        """Total value of attacker_color pieces attacking any square in the 3x3 king zone. Each attacker counted once."""
        zone = set(cls._king_zone_squares(board, king_color))
        attackers_seen = set()
        total = 0.0
        for sq in zone:
            for piece_type, value in cls.PIECE_VALUES.items():
                for attacker_sq in board.attackers(attacker_color, sq):
                    piece = board.piece_at(attacker_sq)
                    if piece is not None and piece.piece_type == piece_type and attacker_sq not in attackers_seen:
                        attackers_seen.add(attacker_sq)
                        total += value
        return total

    @staticmethod
    def _pawn_shield(board: chess.Board, color: chess.Color) -> int:
        """Ally pawns in the 3 squares directly in front of the king. Max=3."""
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        shield_rank = kr + (1 if color == chess.WHITE else -1)
        if not (0 <= shield_rank <= 7):
            return 0
        return sum(
            1 for df in [-1, 0, 1]
            if 0 <= kf + df <= 7
            and (p := board.piece_at(chess.square(kf + df, shield_rank))) is not None
            and p.piece_type == chess.PAWN and p.color == color
        )

    @staticmethod
    def _open_files_near_king(board: chess.Board, color: chess.Color) -> int:
        """Files with no pawn of either color among king_file-1, king_file, king_file+1. Max=3."""
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        kf = chess.square_file(king_sq)
        return sum(
            1 for df in [-1, 0, 1]
            if 0 <= kf + df <= 7
            and not any(
                (p := board.piece_at(chess.square(kf + df, r))) is not None and p.piece_type == chess.PAWN
                for r in range(8)
            )
        )

    @staticmethod
    def _doubled_pawns(board: chess.Board, color: chess.Color) -> int:
        """Ally pawns on a file shared with at least one other ally pawn. Max=8."""
        file_counts = [0] * 8
        for sq in board.pieces(chess.PAWN, color):
            file_counts[chess.square_file(sq)] += 1
        return sum(c for c in file_counts if c >= 2)


# ─────────────────────────────────────────────
# V1 — 21 channels (legacy)
# ─────────────────────────────────────────────

class ChessEncoderV1(ChessEncoder):
    """
    21-channel encoder.

    Channels:
        0-5   White pieces  (P N B R Q K)
        6-11  Black pieces  (P N B R Q K)
        12    En passant square
        13    White kingside castling
        14    White queenside castling
        15    Black kingside castling
        16    Black queenside castling
        17    Turn (1 = White)
        18    In check
        19    Halfmove clock / 100
        20    Twofold repetition — is_repetition(2)
    """

    @property
    def channels(self) -> int:
        return 21

    def encode(self, board: chess.Board) -> torch.Tensor:
        state = np.zeros((21, 8, 8), dtype=np.float32)
        self._piece_planes(board, state, offset=0)
        self._ep_plane(board, state, channel=12)
        self._castling_planes(board, state, offset=13)
        state[17, :, :] = float(board.turn == chess.WHITE)
        state[18, :, :] = float(board.is_check())
        state[19, :, :] = min(board.halfmove_clock, 100) / 100.0
        state[20, :, :] = float(board.is_repetition(2))
        return torch.from_numpy(state)

    def decode(self, state) -> chess.Board:
        state = self._to_numpy(state)
        board = self._decode_base(state)
        board.halfmove_clock = int(round(state[19, 0, 0] * 100))
        return board


# ─────────────────────────────────────────────
# V2 — 30 channels
# ─────────────────────────────────────────────

class ChessEncoderV2(ChessEncoder):
    """
    30-channel encoder with strategic features.
    All scalar channels fill the entire 8x8 plane with one normalized value.

    Channels:
        0-5   White pieces  (P N B R Q K)
        6-11  Black pieces  (P N B R Q K)
        12    En passant square
        13    White kingside castling
        14    White queenside castling
        15    Black kingside castling
        16    Black queenside castling
        17    Turn (1 = White)
        18    In check
        19    Twofold repetition warning (is_repetition(1) and not is_repetition(2))
        20    Current player legal moves / 218
        21    Ally attacked material / 39
        22    Enemy attacked material / 39
        23    Ally pieces in 3×3 king zone / 8
        24    Enemy pieces in 3×3 king zone / 8
        25    Total pieces remaining / 32
        26    King zone attack value / 39
        27    Pawn shield / 3
        28    Open files near king / 3
        29    Doubled pawns / 8
    """

    @property
    def channels(self) -> int:
        return 30

    def encode(self, board: chess.Board) -> torch.Tensor:
        state = np.zeros((30, 8, 8), dtype=np.float32)
        self._piece_planes(board, state, offset=0)
        self._ep_plane(board, state, channel=12)
        self._castling_planes(board, state, offset=13)
        state[17, :, :] = float(board.turn == chess.WHITE)
        state[18, :, :] = float(board.is_check())
        state[19, :, :] = float(board.is_repetition(1) and not board.is_repetition(2))
        me, opp = board.turn, not board.turn
        state[20, :, :] = len(list(board.legal_moves)) / self.MAX_LEGAL
        state[21, :, :] = self._attacked_material_value(board, opp, me) / self.MAX_MATERIAL
        state[22, :, :] = self._attacked_material_value(board, me, opp) / self.MAX_MATERIAL
        state[23, :, :] = self._pieces_around_king(board, me, me) / 8.0
        state[24, :, :] = self._pieces_around_king(board, me, opp) / 8.0
        state[25, :, :] = sum(len(board.pieces(pt, c)) for pt in chess.PIECE_TYPES for c in [chess.WHITE, chess.BLACK]) / 32.0
        state[26, :, :] = self._king_zone_attack_value(board, me, opp) / self.MAX_MATERIAL
        state[27, :, :] = self._pawn_shield(board, me) / 3.0
        state[28, :, :] = self._open_files_near_king(board, me) / 3.0
        state[29, :, :] = self._doubled_pawns(board, me) / 8.0
        return torch.from_numpy(state)

    def decode(self, state) -> chess.Board:
        """Channels 20-29 are derived — only pieces, ep, castling, turn are reconstructed."""
        return self._decode_base(self._to_numpy(state))


# ─────────────────────────────────────────────
# Legacy functional API
# ─────────────────────────────────────────────

_v1 = ChessEncoderV1()
_v2 = ChessEncoderV2()

def encode_chess_state(board: chess.Board) -> torch.Tensor:
    """Backward-compatible wrapper — uses ChessEncoderV1."""
    return _v1.encode(board)

def decode_chess_state(state) -> chess.Board:
    """Backward-compatible wrapper — uses ChessEncoderV1."""
    return _v1.decode(state)


# ─────────────────────────────────────────────
# Move encoding
# ─────────────────────────────────────────────

def move_to_int(move: chess.Move) -> int:
    """Encode a chess.Move as an integer action index (AlphaZero scheme, 4672 actions)."""
    from_sq = move.from_square; to_sq = move.to_square; promo = move.promotion
    from_file = chess.square_file(from_sq); from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq); to_rank = chess.square_rank(to_sq)
    dx = to_file - from_file; dy = to_rank - from_rank
    if promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        direction = 0 if dx == 0 else (1 if dx > 0 else 2)
        piece = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[promo]
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
    """Decode an integer action index back into a chess.Move."""
    from_sq = action // 73; plane_index = action % 73
    from_file = chess.square_file(from_sq); from_rank = chess.square_rank(from_sq)
    promo = None
    if plane_index < 56:
        direction = plane_index // 7; distance = (plane_index % 7) + 1
        dir_map = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        dx, dy = dir_map[direction]
        to_file = from_file + (dx * distance); to_rank = from_rank + (dy * distance)
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



if __name__ == "__main__":
    encoder = Encoder.get("chess")

    print(type(encoder).__name__)