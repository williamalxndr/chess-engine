import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size=10000, seed=42):
        self.max_size = max_size
        self.seed = seed

        self.ring_buffer = [None] * max_size
        self.rng = np.random.default_rng(seed)
        self.count = 0

    def add(self, trajectory, z):
        z = torch.tensor(z, dtype=torch.float32).unsqueeze(-1)

        for s, pi in trajectory:
            pointer = self.count % self.max_size

            s = s.clone().detach().float()
            pi = pi.clone().detach().float()

            self.ring_buffer[pointer] = (s, pi, z)

            self.count += 1

    def sample(self, batch_size):
        """
        Returns a tuple s, pi, z
        with each of them sized batch_size
        """
        if batch_size > len(self):
            raise ValueError("Batch size > replay buffer size")

        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        data = [self.ring_buffer[i] for i in indices]  # list of (s, pi, z)
        s_list, pi_list, z_list = zip(*data)
        s, pi, z = torch.stack(s_list), torch.stack(pi_list), torch.stack(z_list)
        return s, pi, z
            
    
    def __len__(self):
        return min(self.count, self.max_size)


