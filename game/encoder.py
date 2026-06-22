import chess
import math
import numpy as np
import torch


# ENCODER AND DECODER SO NETWORK CAN FORWARD THE STATE

device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

def encode_chess_state(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board into a tensor the network can forward.

    Args:
        board (chess.Board): the board to encode.

    Returns:
        torch.Tensor: float32 tensor of shape (21, 8, 8):
          0-11  : 12 piece type x color planes, one-hot
          12    : en passant target (sparse)
          13-16 : castling rights (W-kingside, W-queenside, B-kingside, B-queenside)
          17    : turn (white to move)
          18    : in check
          19    : halfmove clock / 100
          20    : repetition
    """
    state = np.zeros((21, 8, 8), dtype=np.float32)
    
    # 0-11
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        channel = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        state[channel, row, col] = 1.0
 
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        state[12, row, col] = 1.0
 
    state[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    state[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    state[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    state[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
 
    state[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    state[18, :, :] = 1.0 if board.is_check() else 0.0
    state[19, :, :] = min(board.halfmove_clock, 100) / 100.0
    state[20, :, :] = 1.0 if board.is_repetition() else 0.0

    state = torch.from_numpy(state).float().to(device)
 
    return state

def decode_chess_state(state: np.ndarray) -> chess.Board:
    """
    Decode a (21, 8, 8) encoded state back into a chess.Board.

    Args:
        state (np.ndarray): float32 tensor of shape (21, 8, 8) as produced
            by encode_chess_state.

    Returns:
        chess.Board: the reconstructed board.
    """
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()

    board = chess.Board(fen=None)  # empty board, no pieces
    board.clear()

    # 0-11: piece planes
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for channel in range(12):
        piece_type = piece_types[channel % 6]
        color = chess.WHITE if channel < 6 else chess.BLACK
        for row in range(8):
            for col in range(8):
                if state[channel, row, col] == 1.0:
                    square = chess.square(col, 7 - row)
                    board.set_piece_at(square, chess.Piece(piece_type, color))

    # 12: en passant
    board.ep_square = None
    for row in range(8):
        for col in range(8):
            if state[12, row, col] == 1.0:
                board.ep_square = chess.square(col, 7 - row)
                break

    # 13-16: castling rights
    castling = 0
    if state[13, 0, 0] == 1.0: castling |= chess.BB_H1  # W kingside
    if state[14, 0, 0] == 1.0: castling |= chess.BB_A1  # W queenside
    if state[15, 0, 0] == 1.0: castling |= chess.BB_H8  # B kingside
    if state[16, 0, 0] == 1.0: castling |= chess.BB_A8  # B queenside
    board.castling_rights = castling

    # 17: turn
    board.turn = chess.WHITE if state[17, 0, 0] == 1.0 else chess.BLACK

    # 18: in check — derived from board state, cannot be set directly

    # 19: halfmove clock
    board.halfmove_clock = int(round(state[19, 0, 0] * 100))

    return board

def move_to_int(move):
    """
    Encode a chess.Move as an integer action index.

    Uses the AlphaZero scheme of 73 planes per from-square (56 queen-like
    moves, 8 knight moves, 9 underpromotions), giving 64 * 73 = 4672 actions.

    Args:
        move (chess.Move): the move to encode.

    Returns:
        int: the action index in [0, 4671].
    """
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


def int_to_move(action, board=None):
    """
    Decode an integer action index back into a chess.Move (inverse of move_to_int).

    Args:
        action (int): the action index in [0, 4671].
        board (chess.Board): optional board, used to detect when a forward
            pawn move should be promoted to a queen.

    Returns:
        chess.Move: the decoded move.
    """
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
