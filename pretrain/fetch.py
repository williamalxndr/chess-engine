import chess
import chess.pgn
import sys
import time
import pandas as pd
import torch
import glob
import logging

from core.encoder import Encoder
from core.encoder import move_to_int

logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

encoder = Encoder.get("chess")

RESULT_MAP = {"1-0": -1, "0-1": 1, "1/2-1/2": 0, "*": None}



def color_mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def fetch_pgn(file_path, display=True, skip_moves=10, augment_mirror=True):
    rows = []
    game_stats = []

    with open(file_path, encoding='latin-1') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            result_str = game.headers.get("Result", "*")
            value = RESULT_MAP.get(result_str)
            if value is None:
                continue

            white = game.headers.get("White", "?")
            black = game.headers.get("Black", "?")
            white_elo = game.headers.get("WhiteElo", "?")
            black_elo = game.headers.get("BlackElo", "?")
            event = game.headers.get("Event", "?")
            date = game.headers.get("Date", "?")
            opening = game.headers.get("Opening", "?")

            if display:
                print(f"\n{event} | {date}")
                print(f"White: {white} ({white_elo}) vs Black: {black} ({black_elo})")
                print(f"Opening: {opening}")
                print("-" * 40)

            board = game.board()
            first = True
            move_count = 0

            if display:
                board_str = str(board)
                sys.stdout.write(board_str + "\n")
                sys.stdout.flush()
                first = False

            for move in game.mainline_moves():
                move_count += 1

                if move_count <= skip_moves:
                    board.push(move)
                    continue

                encoded = encoder.encode(board)
                policy = move_to_int(move)

                legal_mask = encoder.legal_action_mask(board)

                rows.append({
                    "encoded_state": encoded,
                    "value": value,
                    "policy": policy,
                    "legal_mask": legal_mask
                })

                if augment_mirror:
                    mirrored_board = board.mirror()
                    mirrored_move = color_mirror_move(move)
                    rows.append({
                        "encoded_state": encoder.encode(mirrored_board),
                        "value": -value, 
                        "policy": move_to_int(mirrored_move),
                        "legal_mask": encoder.legal_action_mask(mirrored_board),
                    })

                board.push(move)

                if display:
                    board_str = str(board)
                    if not first:
                        sys.stdout.write(f"\033[{board_str.count(chr(10)) + 1}A\033[0J")
                    sys.stdout.write(board_str + "\n")
                    sys.stdout.flush()
                    first = False
                    time.sleep(0.0001)

            if display:
                print(result_str)

            game_stats.append({
                "event": event,
                "date": date,
                "white": white,
                "black": black,
                "white_elo": int(white_elo) if white_elo.isdigit() else None,
                "black_elo": int(black_elo) if black_elo.isdigit() else None,
                "result": result_str,
                "moves": move_count,
                "opening": opening,
            })

    return rows, game_stats


def print_statistics(all_game_stats):
    stats_df = pd.DataFrame(all_game_stats)

    total_games = len(stats_df)
    all_players = pd.concat([stats_df["white"], stats_df["black"]]).unique()
    total_players = len(all_players)

    all_elos = pd.concat([
        stats_df["white_elo"].dropna(),
        stats_df["black_elo"].dropna()
    ])

    result_counts = stats_df["result"].value_counts()
    white_wins = result_counts.get("1-0", 0)
    black_wins = result_counts.get("0-1", 0)
    draws = result_counts.get("1/2-1/2", 0)

    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Total games       : {total_games}")
    print(f"Total players     : {total_players}")
    print(f"Avg moves/game    : {stats_df['moves'].mean():.1f}")
    print(f"Min moves/game    : {stats_df['moves'].min()}")
    print(f"Max moves/game    : {stats_df['moves'].max()}")
    print()
    print(f"Results:")
    print(f"  White wins      : {white_wins} ({100*white_wins/total_games:.1f}%)")
    print(f"  Black wins      : {black_wins} ({100*black_wins/total_games:.1f}%)")
    print(f"  Draws           : {draws} ({100*draws/total_games:.1f}%)")
    print()
    if len(all_elos) > 0:
        print(f"Elo stats:")
        print(f"  Mean            : {all_elos.mean():.0f}")
        print(f"  Min             : {all_elos.min():.0f}")
        print(f"  Max             : {all_elos.max():.0f}")
        print(f"  Std             : {all_elos.std():.0f}")
    else:
        print("Elo stats         : not available")
    print()
    print(f"Top 10 players by appearances:")
    player_counts = pd.concat([stats_df["white"], stats_df["black"]]).value_counts().head(10)
    for player, count in player_counts.items():
        print(f"  {player:<30} {count} games")
    print()
    print(f"Top 5 openings:")
    for opening, count in stats_df["opening"].value_counts().head(5).items():
        print(f"  {opening:<40} {count} games")
    print("=" * 50)


def fetch_all(folder_path="pretrain/games", display=True, skip_moves=10, augment_mirror=True):
    all_rows = []
    all_game_stats = []
    pgn_files = sorted(glob.glob(f"{folder_path}/*.pgn"))

    print(f"Found {len(pgn_files)} PGN files")

    for file_path in pgn_files:
        print(f"\n=== {file_path} ===")
        
        try:
            rows, game_stats = fetch_pgn(file_path, display=display, skip_moves=skip_moves, augment_mirror=augment_mirror)
        except Exception as e:
            print(f"FAILED on {file_path}: {e}")
            continue

        all_rows.extend(rows)
        all_game_stats.extend(game_stats)
        print(f"Total rows so far: {len(all_rows)}")

    df = pd.DataFrame(all_rows)
    print_statistics(all_game_stats)
    return df


if __name__ == "__main__":
    _board = chess.Board()
    _board.push_san("e4")  # non-symmetric position so the check isn't trivial
    for _move in _board.legal_moves:
        _mirrored_board = _board.mirror()
        _mirrored_move = color_mirror_move(_move)
        assert _mirrored_board.is_valid(), "Mirrored board is invalid!"
        assert _mirrored_move in _mirrored_board.legal_moves, f"Mirrored move {_mirrored_move} not legal on mirrored board!"

    df = fetch_all(display=False, skip_moves=10, augment_mirror=True)

    states = torch.stack(df["encoded_state"].tolist())  # (N, 30, 8, 8)
    legal_masks = torch.stack(df["legal_mask"].tolist())

    # Save
    torch.save(states, "pretrain/states.pt")
    torch.save(legal_masks, "pretrain/legal_masks.pt")
    df[["value", "policy"]].to_csv("pretrain/target.csv", index=False)