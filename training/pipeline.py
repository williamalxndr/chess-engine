
import numpy as np
from torch import optim
import copy

from game.env import TicTacToe
from core.tree import NetworkMCTS
from core.network import PolicyValueNetwork
from selfplay.replay_buffer import ReplayBuffer
from selfplay.generator import Generator
from training.trainer import Trainer
from arena.arena import Arena, NetworkMCTSPlayer

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
    def __init__(self, network: PolicyValueNetwork, optimizer: optim.Adam = None, batch_size=64, max_size=10000, seed=42, iterations=50, mcts_epochs=100):
        self.network = network
        self.batch_size = batch_size
        self.mcts = NetworkMCTS(network, epochs=mcts_epochs, seed=seed)
        self.replay_buffer = ReplayBuffer(max_size)
        self.generator = Generator(self.mcts, seed=seed)
        self.trainer = Trainer(network, optim.Adam(network.parameters()) if optimizer is None else optimizer)
        self.iterations = iterations
        self.arena = Arena()

    def generate(self, num=20):
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

            self.train_step()

            new_network = self.get_network()
            self.evaluate(old_network, new_network)

        return self.get_network()

    def get_network(self):
        return copy.deepcopy(self.network)
    
    def evaluate(self, old_network: PolicyValueNetwork, new_network: PolicyValueNetwork):
        env = TicTacToe()
        old_mcts = NetworkMCTSPlayer(old_network)
        new_mcts = NetworkMCTSPlayer(new_network)

        self.arena.__init__(env, player_1=old_mcts, player_2=new_mcts, verbose=False)

        results = self.arena.play(100)

        old_network_win = results[id(old_mcts)] / 100
        new_network_win = results[id(new_mcts)] / 100

        print(f"old_network_win: {old_network_win}")
        print(f"new_network_win: {new_network_win}")


if __name__ == "__main__":
    network = PolicyValueNetwork()
    pipeline = Pipeline(network)

    pipeline.train()