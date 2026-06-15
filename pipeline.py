
import numpy as np
from torch import optim

from env import TicTacToe
from tree import NetworkMCTS
from network import PolicyValueNetwork
from replay_buffer import ReplayBuffer
from generator import Generator
from trainer import Trainer

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
    def __init__(self, network: PolicyValueNetwork, optimizer: optim.Adam, max_size=10000, seed=42, iterations=100, mcts_epochs=1000):
        self.network = network
        self.mcts = NetworkMCTS(network, epochs=mcts_epochs, seed=seed)
        self.replay_buffer = ReplayBuffer(max_size)
        self.generator = Generator(self.mcts, seed=seed)
        self.trainer = Trainer(network, optimizer)

        self.iterations = iterations

    def generate(self, num=10):
        for _ in range(num):
            trajectory, z = self.generator.generate()
            self.replay_buffer.add(trajectory, z)

    def sample(self, batch_size=64):
        """
        Returns a tuple s, pi, z
        with each of them sized batch_size
        """
        return self.replay_buffer.sample(batch_size)
    
    def train_step(self, batch_size=64):
        s, pi, z = self.sample(batch_size)
        self.trainer.step(s, pi, z)
            
    
    def train(self):            
        for _ in range(self.iterations):
            self.generate()
            self.train_step()

        return self.get_network()

    def get_network(self):
        return self.network