from arena.arena import *
import argparse

from core.network import PolicyValueNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A sample script demonstrating python argument parsing.")
    parser.add_argument("--path", type=str, help="network you want to load", default="v1")
    parser.add_argument("--num_games", type=int, help="how many times do you want to play against the bot", default=1)

    args = parser.parse_args()

    env = TicTacToe()
    network = PolicyValueNetwork.load(args.path)

    human_player = HumanPlayer()
    bot_player = NetworkMCTSPlayer(network)

    arena = Arena(env, human_player, bot_player, verbose=True, delay=5, num_games=args.num_games)

    arena.play()