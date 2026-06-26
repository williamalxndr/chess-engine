import chess
import math
import numpy as np
import torch
from abc import ABC, abstractmethod


device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")


# ─────────────────────────────────────────────
# Base encoder
# ─────────────────────────────────────────────

class ChessEncoder(ABC):
    """
    Abstract base class for chess state encoders.
    Subclasses define how many channels they produce and how they encode/decode a board.
    """

    _registry: dict[str, type["ChessEncoder"]] = {}
    LATEST_VERSION = "ChessEncoderV2"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ChessEncoder._registry[cls.__name__] = cls

    @property
    @abstractmethod
    def channels(self) -> int:
        pass

    @abstractmethod
    def encode(self, board: chess.Board) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, state) -> chess.Board:
        pass

    @classmethod
    def create(cls, version: str = "latest") -> "ChessEncoder":
        """
        Factory method.

        Args:
            version: 'latest', 'V1', 'V2', or full class name 'ChessEncoderV1'.

        Returns:
            ChessEncoder instance.
        """
        if version.lower() == "latest":
            name = cls.LATEST_VERSION
        elif version.startswith("ChessEncoder"):
            name = version
        else:
            name = f"ChessEncoder{version}"

        if name not in ChessEncoder._registry:
            raise ValueError(f"Unknown encoder '{name}'. Available: {list(ChessEncoder._registry)}")
        return ChessEncoder._registry[name]()

    # ── shared encode helpers ───────────────────────────────────────────────

    @staticmethod
    def _piece_planes(board: chess.Board, state: np.ndarray, offset: int = 0):
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
        state[offset + 0, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        state[offset + 1, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        state[offset + 2, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        state[offset + 3, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    # ── shared decode helpers ───────────────────────────────────────────────

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

    # ── shared feature helpers ──────────────────────────────────────────────

    # Piece values (king excluded from material calculations)
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.5,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
    }
    MAX_MATERIAL = 39.0   # sum of all non-king piece values
    MAX_LEGAL    = 218.0  # theoretical max legal moves in chess

    @staticmethod
    def _king_zone_squares(board: chess.Board, color: chess.Color) -> list[int]:
        """Return list of squares in the 3x3 area around the king of `color`."""
        king_sq = board.king(color)
        if king_sq is None:
            return []
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        squares = []
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                f, r = king_file + df, king_rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    squares.append(chess.square(f, r))
        return squares

    @classmethod
    def _attacked_material_value(cls, board: chess.Board, attacker_color: chess.Color, target_color: chess.Color) -> float:
        """
        Total value of `target_color` pieces that are attacked by `attacker_color`.
        King is excluded from the value sum.
        """
        total = 0.0
        for piece_type, value in cls.PIECE_VALUES.items():
            for sq in board.pieces(piece_type, target_color):
                if board.is_attacked_by(attacker_color, sq):
                    total += value
        return total

    @classmethod
    def _pieces_around_king(cls, board: chess.Board, king_color: chess.Color, piece_color: chess.Color) -> int:
        """Count pieces of `piece_color` within the 3x3 zone around `king_color`'s king."""
        zone = set(cls._king_zone_squares(board, king_color))
        count = 0
        for sq in zone:
            piece = board.piece_at(sq)
            if piece is not None and piece.color == piece_color:
                count += 1
        return count

    @classmethod
    def _king_zone_attack_value(cls, board: chess.Board, king_color: chess.Color, attacker_color: chess.Color) -> float:
        """
        Total value of `attacker_color` pieces that attack any square
        in the 3x3 zone around `king_color`'s king.
        Each attacker piece is counted once regardless of how many zone squares it attacks.
        """
        zone = set(cls._king_zone_squares(board, king_color))
        attackers_seen = set()
        total = 0.0
        for sq in zone:
            for piece_type, value in cls.PIECE_VALUES.items():
                for attacker_sq in board.attackers(attacker_color, sq):
                    piece = board.piece_at(attacker_sq)
                    if (piece is not None
                            and piece.piece_type == piece_type
                            and attacker_sq not in attackers_seen):
                        attackers_seen.add(attacker_sq)
                        total += value
        return total



    @staticmethod
    def _pawn_shield(board, color) -> int:
        """Ally pawns in 3 squares directly in front of king. Max=3."""
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        import chess as _chess
        king_file = _chess.square_file(king_sq)
        king_rank = _chess.square_rank(king_sq)
        shield_rank = king_rank + (1 if color == _chess.WHITE else -1)
        if not (0 <= shield_rank <= 7):
            return 0
        count = 0
        for df in [-1, 0, 1]:
            f = king_file + df
            if 0 <= f <= 7:
                sq = _chess.square(f, shield_rank)
                piece = board.piece_at(sq)
                if piece is not None and piece.piece_type == _chess.PAWN and piece.color == color:
                    count += 1
        return count

    @staticmethod
    def _open_files_near_king(board, color) -> int:
        """Open files (no pawn of either color) among king_file-1,0,+1. Max=3."""
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        import chess as _chess
        king_file = _chess.square_file(king_sq)
        open_count = 0
        for df in [-1, 0, 1]:
            f = king_file + df
            if not (0 <= f <= 7):
                continue
            file_has_pawn = any(
                board.piece_at(_chess.square(f, r)) is not None and
                board.piece_at(_chess.square(f, r)).piece_type == _chess.PAWN
                for r in range(8)
            )
            if not file_has_pawn:
                open_count += 1
        return open_count

    @staticmethod
    def _doubled_pawns(board, color) -> int:
        """
        Count ally pawns that are on a file shared with at least one other ally pawn.
        Max = 8 (all pawns doubled). Normalize by /8.0.
        """
        import chess as _chess
        file_counts = [0] * 8
        for sq in board.pieces(_chess.PAWN, color):
            file_counts[_chess.square_file(sq)] += 1
        return sum(count for count in file_counts if count >= 2)

# ─────────────────────────────────────────────
# V1: original 21-channel encoder (legacy)
# ─────────────────────────────────────────────

class ChessEncoderV1(ChessEncoder):
    """
    Original 21-channel encoder.

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
# V2: 30 channel
# ─────────────────────────────────────────────

class ChessEncoderV2(ChessEncoder):
    """
    27-channel encoder with richer strategic features.
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
        19    Twofold repetition warning:
                position seen exactly twice total
                (is_repetition(1) and not is_repetition(2))
        20    Current player legal move count / 218
        21    Ally attacked material value / 39
                (value of current player's pieces attacked by opponent)
        22    Enemy attacked material value / 39
                (value of opponent's pieces attacked by current player)
        23    Ally pieces in 3×3 king zone / 8
                (current player's pieces around their own king)
        24    Enemy pieces in 3×3 king zone / 8
                (opponent's pieces around current player's king)
        25    Total pieces remaining / 32
        26    King zone attack value / 39
                (total value of opponent pieces attacking squares
                 in the 3×3 zone around current player's king;
                 each attacker counted once)
        27    Pawn shield / 3
                (ally pawns in 3 squares directly in front of current player's king)
        28    Open files near king / 3
                (files with no pawn of either color among king_file-1, king_file, king_file+1)
        29    Doubled pawns / 8
                (ally pawns on a file shared with at least one other ally pawn)
    """

    @property
    def channels(self) -> int:
        return 30

    def encode(self, board: chess.Board) -> torch.Tensor:
        state = np.zeros((30, 8, 8), dtype=np.float32)

        # 0-11: piece planes
        self._piece_planes(board, state, offset=0)

        # 12: en passant
        self._ep_plane(board, state, channel=12)

        # 13-16: castling
        self._castling_planes(board, state, offset=13)

        # 17: turn
        state[17, :, :] = float(board.turn == chess.WHITE)

        # 18: in check
        state[18, :, :] = float(board.is_check())

        # 19: twofold repetition warning
        state[19, :, :] = float(board.is_repetition(1) and not board.is_repetition(2))

        # perspective
        me  = board.turn
        opp = not board.turn

        # 20: current player legal move count / 218
        state[20, :, :] = len(list(board.legal_moves)) / self.MAX_LEGAL

        # 21: ally attacked material value / 39
        state[21, :, :] = self._attacked_material_value(board, opp, me) / self.MAX_MATERIAL

        # 22: enemy attacked material value / 39
        state[22, :, :] = self._attacked_material_value(board, me, opp) / self.MAX_MATERIAL

        # 23: ally pieces in 3x3 king zone / 8
        state[23, :, :] = self._pieces_around_king(board, me, me) / 8.0

        # 24: enemy pieces in 3x3 king zone / 8
        state[24, :, :] = self._pieces_around_king(board, me, opp) / 8.0

        # 25: total pieces remaining / 32
        total_pieces = sum(
            len(board.pieces(pt, color))
            for pt in chess.PIECE_TYPES
            for color in [chess.WHITE, chess.BLACK]
        )
        state[25, :, :] = total_pieces / 32.0

        # 26: king zone attack value / 39
        state[26, :, :] = self._king_zone_attack_value(board, me, opp) / self.MAX_MATERIAL

        # 27: pawn shield / 3
        state[27, :, :] = self._pawn_shield(board, me) / 3.0

        # 28: open files near king / 3
        state[28, :, :] = self._open_files_near_king(board, me) / 3.0

        # 29: doubled pawns / 8
        state[29, :, :] = self._doubled_pawns(board, me) / 8.0

        return torch.from_numpy(state)

    def decode(self, state) -> chess.Board:
        """
        Decode a (27, 8, 8) V2 state.
        Channels 20-29 are derived features — not restored (would need the
        original board state to recompute anyway). Only pieces, ep,
        castling, and turn are reconstructed.
        """
        state = self._to_numpy(state)
        return self._decode_base(state)


# ─────────────────────────────────────────────
# Legacy functional API (backward compat)
# ─────────────────────────────────────────────

_default_encoder = ChessEncoderV2()

def encode_chess_state(board: chess.Board) -> torch.Tensor:
    """Backward-compatible wrapper — uses ChessEncoderV1."""
    return _default_encoder.encode(board)

def decode_chess_state(state) -> chess.Board:
    """Backward-compatible wrapper — uses ChessEncoderV1."""
    return _default_encoder.decode(state)


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