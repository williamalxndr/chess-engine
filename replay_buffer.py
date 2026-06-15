import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=100, seed=42):
        self.max_size = max_size
        self.seed = seed

        self.ring_buffer = [None] * max_size
        self.rng = np.random.default_rng(seed)
        self.count = 0

    def add(self, trajectory, result):
        for s, pi in trajectory:
            pointer = self.count % self.max_size
            self.ring_buffer[pointer] = (s.copy(), pi, result)
        
            self.count += 1

    def sample(self, batch_size):
        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        return [self.ring_buffer[i] for i in indices]
    
    
    def __len__(self):
        return min(self.count, self.max_size)

    def is_empty(self):
        return len(self) == 0