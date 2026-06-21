
import numpy as np
import torch
from torch import optim
import copy
import argparse
from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn
from pathlib import Path


from game.env import TicTacToe
from game.rules import RULES_REGISTRY
from core.tree import NetworkMCTS
from core.network import PolicyValueNetwork
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
    def __init__(self, network: PolicyValueNetwork = None, game="tictactoe", optimizer: optim.Adam = None, batch_size=64, max_size=10000, seed=42, iterations=200, num_mcts_rollout=1000, steps_per_iter=200, early_stopping=50):
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

    def train(self, path="version_1"):        
        min_loss = float('inf')   
        not_improving = 0        

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("• eta:"),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("loss: 0.0000 | p_loss: 0.0000 | v_loss: 0.0000", total=self.iterations)

            for _ in range(self.iterations):
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
                progress.update(
                    task,
                    advance=1,
                    description=f"loss: {loss:.4f} | policy loss: {policy_loss:.4f} | value loss: {v_loss:.4f}{patience_str}"
                )

                # LR scheduling
                self.trainer.scheduler.step()

        self.save(path)

        print(f"Training finished! To play against the trained MCTS, run `python3 -m arena.play --path {path}`")

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
    parser.add_argument("--path", type=str, help="network you want to load", default="version_2")
    parser.add_argument("--iterations", type=int, help="How many iteration to run?", default=500)
    parser.add_argument("--steps_per_iter", type=int, help="How many steps of optimization per iteration?", default=200)
    parser.add_argument("--batch_size", type=int, help="How many batch of data per train step?", default=64)

    args = parser.parse_args()

    network = PolicyValueNetwork(rules=RULES_REGISTRY[args.game])
    pipeline = Pipeline(network, game=args.game, iterations=args.iterations, steps_per_iter=args.steps_per_iter, batch_size=args.batch_size)

    pipeline.train(args.path)