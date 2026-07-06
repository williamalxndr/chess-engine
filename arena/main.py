import os

from arena.arena import Arena
from arena.player import VanillaMCTSPlayer, NetworkMCTSPlayer, RandomPlayer
from core.network import PolicyValueNetwork
from game.env import Chess, TicTacToe


MATCHUPS = {
    "1": ("Vanilla MCTS vs Network", "vanilla", "network"),
    "2": ("Network vs Random", "network", "random"),
    "3": ("Vanilla MCTS vs Random", "vanilla", "random"),
}


def prompt_choice(prompt, options):
    while True:
        raw = input(prompt).strip()
        if raw in options:
            return raw
        print(f"Invalid choice, pick one of {list(options.keys())}")


def prompt_nonempty(prompt):
    while True:
        raw = input(prompt).strip()
        if raw:
            return raw
        print("Input can't be empty, try again.")


def build_player(kind, game, env):
    if kind == "vanilla":
        return VanillaMCTSPlayer()

    if kind == "network":
        path = prompt_nonempty("Path to network checkpoint: ")
        network = PolicyValueNetwork.load(path=path)
        num_rollout_raw = input("num_rollout (default 2000): ").strip()
        num_rollout = int(num_rollout_raw) if num_rollout_raw else 2000
        return NetworkMCTSPlayer(network, game=game, num_rollout=num_rollout)

    if kind == "random":
        return RandomPlayer(env)

    raise ValueError(f"Unknown player kind: {kind}")


def build_env(game):
    if game == "chess":
        return Chess()
    if game == "tictactoe":
        return TicTacToe()
    raise ValueError(f"Unknown game: {game}")


def main():
    print("Select matchup:")
    for key, (label, _, _) in MATCHUPS.items():
        print(f"  {key}. {label}")

    matchup_key = prompt_choice("> ", MATCHUPS)
    _, kind_1, kind_2 = MATCHUPS[matchup_key]

    game_key = prompt_choice("Game? (1=chess, 2=tictactoe): ", {"1": "chess", "2": "tictactoe"})
    game = {"1": "chess", "2": "tictactoe"}[game_key]

    env = build_env(game)

    num_games_raw = input("num_games (default 1): ").strip()
    num_games = int(num_games_raw) if num_games_raw else 1

    player_1 = build_player(kind_1, game, env)
    player_2 = build_player(kind_2, game, env)

    arena = Arena(player_1, player_2, verbose=True, num_games=num_games)
    results = arena.play()

    print("Results:", results)


if __name__ == "__main__":
    main()