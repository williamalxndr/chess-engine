import time
import numpy as np
import torch
from torch import optim
import copy
import argparse
from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn, MofNCompleteColumn
from pathlib import Path
from contextlib import contextmanager

from game.env import TicTacToe
from game.rules import RULES_REGISTRY
from core.tree import NetworkMCTS
from core.network import PolicyValueNetwork, NetworkFactory
from selfplay.replay_buffer import ReplayBuffer
from selfplay.generator import Generator
from training.trainer import Trainer
from arena.arena import *

class Pipeline:
    """
    This is where the real training happens. Each iteration runs this
    cycle, the network the Trainer improves is then reused by the
    Generator for the next round of self-play:

        Generator  -->  ReplayBuffer  -->  Trainer
       (self-play)     (stores s,pi,z)   (optimizes net)
            ^                                  |
            |_______ improved network _________|
    """
    def __init__(self, network: PolicyValueNetwork = None, game="tictactoe", optimizer: optim.Adam = None, train_batch_size=32, replay_buffer_max_size=10000, seed=42, iterations=50, mcts_batch_size=16, num_mcts_rollout=100, steps_per_iter=200, patience=50, verbose=False, version="v1"):
        self.network = network
        self.game = game
        self.train_batch_size = train_batch_size
        self.iterations = iterations
        self.steps_per_iter = steps_per_iter
        self.patience = patience
        self.verbose = verbose
        self.rules = RULES_REGISTRY[game]

        network = PolicyValueNetwork(self.rules) if network is None else network
        self.mcts = NetworkMCTS(network, rules=RULES_REGISTRY[game], num_rollout=num_mcts_rollout, batch_size=mcts_batch_size, seed=seed, add_noise=True)
        buffer_path = Path(f"checkpoints/{self.game}/{version}_buffer.pt")
        if buffer_path.exists():
            self.replay_buffer = ReplayBuffer.load(buffer_path)
        else:
            self.replay_buffer = ReplayBuffer(replay_buffer_max_size)
        self.generator = Generator(self.mcts, seed=seed)
        self.trainer = Trainer(network, optim.Adam(network.parameters(), lr=0.01) if optimizer is None else optimizer, T_max=iterations)
        self.arena = Arena()

    def generate(self):
        num_generate = max(1, self.train_batch_size // self.rules.avg_game_length)
        for _ in range(num_generate):
            trajectory, z = self.generator.generate()
            self.replay_buffer.add(trajectory, z)

    def sample(self):
        """
        Returns a tuple sf, pi, z
        with each of them sized batch_size
        """
        return self.replay_buffer.sample(self.train_batch_size)
    
    def train_step(self):
        s, pi, z = self.sample()
        return self.trainer.step(s, pi, z)

    def train(self, version="v1", duration_hours=None, log_interval=5):
        duration_seconds = duration_hours * 3600 if duration_hours else None
        total_target = self.iterations

        with self._make_progress(duration_hours, total_target) as (progress, task):
            iteration = 0
            start_time = time.time()
            last_save_time = start_time
            save_interval = 1800 
            min_loss = float('inf')
            not_improving = 0
            loss = policy_loss = v_loss = 0.0

            while True:
                current_time = time.time()

                if self._should_stop(current_time, start_time, duration_seconds, iteration):
                    break

                self.network.eval()
                self.generate()
                while len(self.replay_buffer) < self.train_batch_size:
                    self.generate()

                for _ in range(self.steps_per_iter):
                    loss, policy_loss, v_loss = self.train_step()

                min_loss, not_improving = self._update_early_stopping(loss, min_loss, not_improving)
                if not_improving >= self.patience:
                    break

                current_time = time.time()

                if (iteration+1) % log_interval == 0:
                    elapsed = current_time - start_time
                    print(f"iter {iteration+1} | loss: {loss:.4f} | policy: {policy_loss:.4f} | value: {v_loss:.4f} | {elapsed/60:.1f}m elapsed")

                if current_time - last_save_time >= save_interval:
                    self.save(version)
                    last_save_time = current_time

                self._update_progress(progress, task, current_time, start_time,
                                    duration_seconds, loss, policy_loss, v_loss, not_improving)

                self.trainer.scheduler.step()
                iteration += 1

        self.save(version)
        self._print_done(duration_seconds, version)
        return self.get_network()

    @contextmanager
    def _make_progress(self, duration_hours, total_target):
        initial_desc = "loss: 0.0000 | policy loss: 0.0000 | value loss: 0.0000"
        if duration_hours:
            initial_desc = f"Time left: {duration_hours:.2f}h | " + initial_desc

        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            transient=False,
            disable=not self.verbose,
        )
        task = progress.add_task(initial_desc, total=total_target)

        with progress:
            yield progress, task

    def _should_stop(self, current_time, start_time, duration_seconds, iteration):
        if duration_seconds and (current_time - start_time) >= duration_seconds:
            return True
        return iteration >= self.iterations

    def _update_early_stopping(self, loss, min_loss, not_improving):
        if loss < min_loss:
            return loss, 0
        return min_loss, not_improving + 1

    def _update_progress(self, progress, task, current_time, start_time,
                        duration_seconds, loss, policy_loss, v_loss, not_improving):
        if not self.verbose:
            return

        patience_str = f" | ⚠ patience: {not_improving}/{self.patience}" if not_improving > self.patience * 0.5 else ""
        desc = f"loss: {loss:.4f} | policy loss: {policy_loss:.4f} | value loss: {v_loss:.4f}{patience_str}"

        if duration_seconds:
            remaining = max(0, duration_seconds - (current_time - start_time))
            desc = f"Time left: {remaining/3600:.2f}h | " + desc

        progress.update(task, advance=1, description=desc)

    def _print_done(self, duration_seconds, version):
        if not self.verbose:
            print()
            return
        print(f"Training finished! To play against the trained MCTS, run `python3 -m arena.play --game {self.game} --version {version}`")

    def get_network(self):
        return copy.deepcopy(self.network)

    def save(self, version: str):
        self.network.save(self.game, version)
        self.replay_buffer.save(f"checkpoints/{self.game}/{version}_buffer.pt")

    @staticmethod
    def load(network: PolicyValueNetwork, version: str) -> PolicyValueNetwork:
        network.load_state_dict(torch.load(version))
        return network

    def evaluate(self, old_network: PolicyValueNetwork, new_network: PolicyValueNetwork, num_games=100):
        env = TicTacToe()
        old_mcts = NetworkMCTSPlayer(old_network)
        new_mcts = NetworkMCTSPlayer(new_network)
        random_player = RandomPlayer()
        vanilla_mcts = VanillaMCTSPlayer()

        player_1 = new_mcts
        player_2 = vanilla_mcts

        self.arena.__init__(env, player_1=vanilla_mcts, player_2=new_mcts, verbose=False)

        results = self.arena.play(num_games)

        if self.verbose:
            player_1_win = results[id(player_1)] / num_games
            player_2_win = results[id(player_2)] / num_games

            print(results)
            print(f"player_1_win: {player_1_win}")
            print(f"player_2_win: {player_2_win}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, help=f"what game do you want to train? listed games= {list(RULES_REGISTRY.keys())}", default="chess")
    parser.add_argument("--version", type=str, help="version for saving the network OR load network from", default="v1")
    parser.add_argument("--iterations", type=int, help="How many iteration to run?", default=100)
    parser.add_argument("--steps_per_iter", type=int, help="How many steps of optimization per iteration?", default=200)
    parser.add_argument("--train_batch_size", type=int, help="How many sample for replay buffer sampling?", default=32)
    parser.add_argument("--mcts_batch_size", type=int, help="How many batch size for mcts forwarding (in evaluation step)?", default=16)
    parser.add_argument("--num_rollout", type=int, help="How many rollout?", default=100)
    parser.add_argument("--duration", type=float, help="How many hours to train? (Overrides iterations)", default=0.0)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")
    parser.add_argument("--replay_buffer_max_size", type=int, help="Max size of replay buffer", default=100000)
    parser.add_argument("--log_interval", type=int, help="Log every N iterations", default=5)

    args = parser.parse_args()

    # Verify the path points to an actual file
    file_path = Path(f"checkpoints/{args.game}/{args.version}.pt")
    if file_path.is_file():
        # Load the network if it exist
        network = PolicyValueNetwork.load(args.game, args.version)
        print("network loaded")
    else:
        # Create a new network if it doesn't exist yet
        network = NetworkFactory.create(args.game)

    pipeline = Pipeline(
        network=network,
        game=args.game,
        version=args.version,
        iterations=args.iterations,
        steps_per_iter=args.steps_per_iter,
        train_batch_size=args.train_batch_size,
        mcts_batch_size=args.mcts_batch_size,
        num_mcts_rollout=args.num_rollout,
        replay_buffer_max_size=args.replay_buffer_max_size,
        verbose=args.verbose,
    )

    pipeline.train(args.version, duration_hours=args.duration if args.duration > 0 else None, log_interval=args.log_interval)