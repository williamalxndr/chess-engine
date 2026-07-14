import chess
import chess.pgn
import sys
import time
import pandas as pd
import torch
import glob
import logging
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

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


def _save_shard(rows, shard_dir, shard_name):
    states      = torch.stack([r["encoded_state"] for r in rows])
    legal_masks = torch.stack([r["legal_mask"] for r in rows])
    values      = torch.tensor([r["value"] for r in rows], dtype=torch.float32)
    policies    = torch.tensor([r["policy"] for r in rows], dtype=torch.long)

    torch.save(states, f"{shard_dir}/{shard_name}_states.pt")
    torch.save(legal_masks, f"{shard_dir}/{shard_name}_masks.pt")
    torch.save({"value": values, "policy": policies}, f"{shard_dir}/{shard_name}_targets.pt")

    return len(rows)


def _resolve_start_index(pgn_files, start_from):
    start_stem = Path(start_from).stem
    for i, f in enumerate(pgn_files):
        if Path(f).stem == start_stem:
            return i
    raise ValueError(f"--start-from '{start_from}' not found in PGN file list")


def fetch_all(folder_path="pretrain/games", shard_dir="pretrain/shards",
              display=True, skip_moves=10, augment_mirror=True,
              push_repo=None, start_from=None):
    Path(shard_dir).mkdir(parents=True, exist_ok=True)
    all_game_stats = []
    total_rows = 0
    pgn_files = sorted(glob.glob(f"{folder_path}/*.pgn"))

    if start_from:
        idx = _resolve_start_index(pgn_files, start_from)
        pgn_files = pgn_files[idx:]

    print(f"Processing {len(pgn_files)} PGN files" + (f" starting at {start_from}" if start_from else ""))

    for file_path in pgn_files:
        print(f"\n=== {file_path} ===")
        try:
            rows, game_stats = fetch_pgn(file_path, display=display, skip_moves=skip_moves, augment_mirror=augment_mirror)
        except Exception as e:
            print(f"FAILED on {file_path}: {e}")
            continue

        all_game_stats.extend(game_stats)

        if not rows:
            continue

        shard_name = Path(file_path).stem
        n = _save_shard(rows, shard_dir, shard_name)
        total_rows += n
        print(f"Shard saved: {shard_name} ({n} rows) | Total rows so far: {total_rows}")

        del rows

        if push_repo:
            _push_shard_to_hub(push_repo, shard_dir, shard_name)

    print_statistics(all_game_stats)
    print(f"\nAll shards written to {shard_dir}/ ({total_rows} total rows across {len(pgn_files)} files)")


def push_to_hub(repo_id, files, repo_type="dataset", private=False, commit_message="Add pretrain data"):
    api = HfApi()

    create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)

    for file_path in files:
        print(f"Uploading {file_path} -> {repo_id} ({repo_type})")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.split("/")[-1],
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )

    print(f"Done. View at: https://huggingface.co/datasets/{repo_id}" if repo_type == "dataset"
          else f"Done. View at: https://huggingface.co/{repo_id}")


def _push_shard_to_hub(repo_id, shard_dir, shard_name, delete_after=True):
    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)

    shard_files = [
        f"{shard_dir}/{shard_name}_states.pt",
        f"{shard_dir}/{shard_name}_masks.pt",
        f"{shard_dir}/{shard_name}_targets.pt",
    ]
    for f in shard_files:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=Path(f).name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add shard {shard_name}",
        )
        if delete_after:
            Path(f).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=str, default=None,
                         help="PGN filename (with or without .pgn) to resume from; processes that file through the end. Omit to process all files.")
    args = parser.parse_args()

    _board = chess.Board()
    _board.push_san("e4")
    for _move in _board.legal_moves:
        _mirrored_board = _board.mirror()
        _mirrored_move = color_mirror_move(_move)
        assert _mirrored_board.is_valid(), "Mirrored board is invalid!"
        assert _mirrored_move in _mirrored_board.legal_moves, f"Mirrored move {_mirrored_move} not legal on mirrored board!"

    fetch_all(
        display=False,
        skip_moves=10,
        augment_mirror=True,
        push_repo="williamalxndr/chess-pretrain-data",
        start_from=args.start_from,
    )