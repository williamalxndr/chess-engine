from arena.arena import *
import argparse

from core.network import PolicyValueNetwork
from game.env import Chess, TicTacToe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A sample script demonstrating python argument parsing.")
    parser.add_argument("--game", type=str, default="chess")
    parser.add_argument("--version", type=str, help="network you want to load", default="v1")
    parser.add_argument("--path", type=str, help="absolute path to load the network", default=None)
    parser.add_argument("--num_games", type=int, help="how many times do you want to play against the bot", default=1)
    parser.add_argument("--parent_dir", type=str, help="where do you want to load the network from", default="checkpoints")

    args = parser.parse_args()

    if args.path:
        network = PolicyValueNetwork.load(path=args.path)
    else:
        network = PolicyValueNetwork.load(game=args.game, version=args.version, parent_dir=args.parent_dir)
        
    if args.game == "chess":
        env = Chess()
    elif args.game == "tictactoe":
        env = TicTacToe()

    human_player = HumanPlayer(env)
    bot_player = NetworkMCTSPlayer(network, game=args.game)

    arena = Arena(human_player, bot_player, verbose=True, num_games=args.num_games)

    arena.play()
