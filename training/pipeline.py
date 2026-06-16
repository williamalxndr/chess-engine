
import numpy as np
import torch
from torch import optim
import copy

from game.env import TicTacToe
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
    def __init__(self, network: PolicyValueNetwork, optimizer: optim.Adam = None, batch_size=64, max_size=10000, seed=42, iterations=200, mcts_epochs=50, steps_per_iter=200):
        self.network = network
        self.batch_size = batch_size
        self.mcts = NetworkMCTS(network, epochs=mcts_epochs, seed=seed)
        self.replay_buffer = ReplayBuffer(max_size)
        self.generator = Generator(self.mcts, seed=seed)
        self.trainer = Trainer(network, optim.Adam(network.parameters(), lr=0.01) if optimizer is None else optimizer, T_max=iterations)
        self.iterations = iterations
        self.steps_per_iter = steps_per_iter
        self.arena = Arena()

    def generate(self, num=10):
        for _ in range(num):
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
        self.trainer.step(s, pi, z)
            
    
    def train(self):            
        for i in range(self.iterations):
            print(f"Iteration {i}")
            
            old_network = self.get_network()

            self.generate()

            while len(self.replay_buffer) < self.batch_size:
                self.generate()

            for _ in range(self.steps_per_iter):
                self.train_step()
            self.trainer.scheduler.step()
            print()

            new_network = self.get_network()
        self.evaluate(old_network, new_network)

        return self.get_network()

    def get_network(self):
        return copy.deepcopy(self.network)
    
    def evaluate(self, old_network: PolicyValueNetwork, new_network: PolicyValueNetwork, num_simulations=100):
        env = TicTacToe()
        old_mcts = NetworkMCTSPlayer(old_network)
        new_mcts = NetworkMCTSPlayer(new_network)
        random_player = RandomPlayer()
        vanilla_mcts = VanillaMCTSPlayer()

        player_1 = new_mcts
        player_2 = vanilla_mcts

        self.arena.__init__(env, player_1=vanilla_mcts, player_2=new_mcts, verbose=False)

        results = self.arena.play(num_simulations)

        player_1_win = results[id(player_1)] / num_simulations
        player_2_win = results[id(player_2)] / num_simulations

        print(results)
        print(f"player_1_win: {player_1_win}")
        print(f"player_2_win: {player_2_win}")

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    @staticmethod
    def load(network: PolicyValueNetwork, path: str) -> PolicyValueNetwork:
        network.load_state_dict(torch.load(path))
        return network

if __name__ == "__main__":
    network = PolicyValueNetwork()
    pipeline = Pipeline(network)

    pipeline.train()
    pipeline.save("checkpoints/network_final.pt")