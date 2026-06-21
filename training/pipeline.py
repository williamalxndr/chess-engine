import time
import numpy as np
import torch
from torch import optim
import copy
import argparse
from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from pathlib import Path

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
    def __init__(self, network: PolicyValueNetwork = None, game="tictactoe", optimizer: optim.Adam = None, batch_size=8, max_size=10000, seed=42, iterations=50, num_mcts_rollout=100, steps_per_iter=200, early_stopping=50):
        self.network = network
        self.game = game
        self.batch_size = batch_size
        self.iterations = iterations
        self.steps_per_iter = steps_per_iter
        self.early_stopping = early_stopping

        network = PolicyValueNetwork(RULES_REGISTRY[game]) if network is None else network
        self.mcts = NetworkMCTS(network, rules=RULES_REGISTRY[game], num_rollout=num_mcts_rollout, seed=seed, add_noise=True)
        self.replay_buffer = ReplayBuffer(max_size)
        self.generator = Generator(self.mcts, seed=seed)
        self.trainer = Trainer(network, optim.Adam(network.parameters(), lr=0.01) if optimizer is None else optimizer, T_max=iterations)
        self.arena = Arena()

    def generate(self):
        num_generate = max(1, self.batch_size // 5)

        for _ in range(num_generate):
            trajectory, z = self.generator.generate()
            self.replay_buffer.add(trajectory, z)

    def sample(self):
        """
        Returns a tuple s, pi, z
        with each of them sized batch_size
        """
        return self.replay_buffer.sample(self.batch_size)
    
    def train_step(self):
        s, pi, z = self.sample()
        return self.trainer.step(s, pi, z)

    def train(self, path="example", duration_hours=None):        
        min_loss = float('inf')   
        not_improving = 0        
        start_time = time.time()
        duration_seconds = duration_hours * 3600 if duration_hours else None

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            total_target = duration_seconds if duration_seconds else self.iterations
            
            initial_desc = "loss: 0.0000 | policy loss: 0.0000 | value loss: 0.0000"
            if duration_seconds:
                initial_desc = f"Time left: {duration_hours:.2f}h | " + initial_desc
                
            task = progress.add_task(
                initial_desc, 
                total=total_target
            )

            iteration = 0
            while True:
                current_time = time.time()
                if duration_seconds and (current_time - start_time) >= duration_seconds:
                    progress.update(task, completed=duration_seconds)
                    break
                if not duration_seconds and iteration >= self.iterations:
                    break

                # Generate self play data, if batch size is greater then the size of replay buffer stored then generate again
                self.generate()
                while len(self.replay_buffer) < self.batch_size:
                    self.generate()

                # Train the network for n_steps_per_iter
                for _ in range(self.steps_per_iter):
                    loss, policy_loss, v_loss = self.train_step()

                # Early stopping check
                if loss < min_loss:
                    min_loss = loss
                    not_improving = 0
                else:
                    not_improving += 1

                if not_improving >= self.early_stopping:
                    break
                    
                # Progress bar display
                patience_str = f" | ⚠ patience: {not_improving}/{self.early_stopping}" if not_improving > self.early_stopping * 0.5 else ""
                
                desc = f"loss: {loss:.4f} | policy loss: {policy_loss:.4f} | value loss: {v_loss:.4f}{patience_str}"
                
                if duration_seconds:
                    current_time = time.time()
                    remaining = max(0, duration_seconds - (current_time - start_time))
                    desc = f"Time left: {remaining/3600:.2f}h | " + desc
                    progress.update(task, completed=(current_time - start_time), description=desc)
                else:
                    progress.update(task, advance=1, description=desc)

                # LR scheduling
                self.trainer.scheduler.step()
                iteration += 1

        self.save(path)

        print(f"Training finished! To play against the trained MCTS, run `python3 -m arena.play --game {self.game} --path {path}`")

        return self.get_network()

    def get_network(self):
        return copy.deepcopy(self.network)
    
    def save(self, path: str):
        # mkdir if not exist
        self.network.save(self.game, path)

    @staticmethod
    def load(network: PolicyValueNetwork, path: str) -> PolicyValueNetwork:
        network.load_state_dict(torch.load(path))
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

        player_1_win = results[id(player_1)] / num_games
        player_2_win = results[id(player_2)] / num_games

        print(results)
        print(f"player_1_win: {player_1_win}")
        print(f"player_2_win: {player_2_win}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, help=f"what game do you want to train? listed games= {list(RULES_REGISTRY.keys())}", default="chess")
    parser.add_argument("--path", type=str, help="path for saving the network OR load network from", default="example")
    parser.add_argument("--iterations", type=int, help="How many iteration to run?", default=100)
    parser.add_argument("--steps_per_iter", type=int, help="How many steps of optimization per iteration?", default=200)
    parser.add_argument("--batch_size", type=int, help="How many batch of data per train step?", default=8)
    parser.add_argument("--num_rollout", type=int, help="How many rollout?", default=100)
    parser.add_argument("--duration", type=float, help="How many hours to train? (Overrides iterations)", default=0.0)

    args = parser.parse_args()

    # Verify the path points to an actual file
    file_path = Path(f"checkpoints/{args.game}/{args.path}.pt")
    if file_path.is_file():
        # Load the network if it exist
        network = PolicyValueNetwork.load(args.game, args.path)
        print("network loaded")
    else:
        # Create a new network if it doesn't exist yet
        network = NetworkFactory.create(args.game)

    pipeline = Pipeline(
        network=network,
        game=args.game, 
        iterations=args.iterations, 
        steps_per_iter=args.steps_per_iter, 
        batch_size=args.batch_size, 
        num_mcts_rollout=args.num_rollout,
    )

    pipeline.train(args.path, duration_hours=args.duration if args.duration > 0 else None)