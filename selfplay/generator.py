import numpy as np
import torch
import time
import sys

from core.tree import NetworkMCTS
from core.network import PolicyValueNetwork, NetworkFactory
from game.encoder import *
from selfplay.worker import GameWorker


class SelfPlayGenerator:
    def __init__(self, network: PolicyValueNetwork, num_rollout=100, batch_size=50, seed=42):
        self.network = network
        self.rules = network.rules
        self.num_rollout = num_rollout
        self.batch_size=batch_size

        self.worker: dict = {} 
        self.rng = np.random.default_rng(seed=seed)
        self.valid_seeds = list(range(101))
        self.registered_seeds = []

    @property
    def num_worker(self): 
        return len(self.worker)

    def spawn(self, n):
        spawned_seeds = self.rng.choice(self.valid_seeds, n)
        self.registered_seeds.append(spawned_seeds)
        self.valid_seeds = [item for item in self.valid_seeds if item not in spawned_seeds]
        
        for seed in spawned_seeds:
            self.worker[seed] = GameWorker(
                self.network,
                num_rollout=self.num_rollout,
                batch_size=self.batch_size,
                add_noise=True
            )
    
    def kill(self, n):
        killed_seeds = self.rng.choice(self.registered_seeds, n)
        for seed in killed_seeds:
            del self.worker[seed]
            self.valid_seeds.append(seed)
        self.registered_seeds = [s for s in self.registered_seeds if s not in killed_seeds]
        
    def reset(self):
        self.worker = {}
        self.registered_seeds = []
        self.valid_seeds = list(range(101))


    def generate(self, num_games, display=False):
        diff = num_games - self.num_worker
        if diff > 0: self.spawn(diff)
        elif diff < 0: self.kill(-diff)
        
        all_trajectories = []
        
        for seed, worker in self.worker.items():
            trajectory, result = worker.generate(display=display)
            all_trajectories.append((trajectory, result))
        
        return all_trajectories

        

if __name__ == "__main__": 
    network = PolicyValueNetwork.create("chess")
    gen = SelfPlayGenerator(network=network)

    for _ in range(5):
        gen.generate()

