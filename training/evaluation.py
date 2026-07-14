import argparse
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from arena.arena import Arena
from arena.player import NetworkMCTSPlayer
from core.network import PolicyValueNetwork


def _unwrap(network):
    return network.module if isinstance(network, DDP) else network


@dataclass(frozen=True)
class MatchResult:
    new_wins: int
    old_wins: int
    draws: int
    win_rate: float
    promote: bool


@torch.no_grad()
def evaluate(new_network, old_network, game="chess", num_rollout=100,
             num_games=10, win_rate_threshold=0.55, verbose=False,
             pgn_dir="eval_pgns") -> MatchResult:
    new_net = _unwrap(new_network).eval()
    old_net = _unwrap(old_network).eval()

    new_player = NetworkMCTSPlayer(new_net, game=game, num_rollout=num_rollout, name="new")
    old_player = NetworkMCTSPlayer(old_net, game=game, num_rollout=num_rollout, name="old")

    results = Arena(new_player, old_player, num_games=num_games,
                    verbose=verbose, pgn_dir=pgn_dir).play()

    new_wins, old_wins, draws = results[id(new_player)], results[id(old_player)], results["Draw"]
    # Draws count as half a win, standard promotion convention.
    win_rate = (new_wins + 0.5 * draws) / num_games if num_games else 0.0

    return MatchResult(new_wins, old_wins, draws, win_rate, win_rate >= win_rate_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a new checkpoint against an old one.")
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--game", default="chess")
    parser.add_argument("--num_rollout", type=int, default=100)
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    result = evaluate(
        PolicyValueNetwork.load(path=args.new),
        PolicyValueNetwork.load(path=args.old),
        game=args.game, num_rollout=args.num_rollout,
        num_games=args.num_games, win_rate_threshold=args.threshold, verbose=True,
    )
    print(f"new {result.new_wins}-{result.old_wins}-{result.draws} old "
          f"| win_rate {result.win_rate:.2f} | promote={result.promote}")
