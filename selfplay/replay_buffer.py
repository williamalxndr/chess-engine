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
        print(f"[Buffer] Adding trajectory: {len(trajectory)} steps | z={z} | buffer_size={len(self)}/{self.max_size}")
        z = torch.tensor(z, dtype=torch.float32).unsqueeze(-1)

        for s, pi in trajectory:
            pointer = self.count % self.max_size

            s = s.clone().detach().float()
            pi = pi.clone().detach().float()

            self.ring_buffer[pointer] = (s, pi, z)

            self.count += 1
            
        print(f"[Buffer] Done | buffer_size={len(self)}/{self.max_size}")


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
    
    def save(self, path: str):
        torch.save({
            'ring_buffer': self.ring_buffer,
            'count': self.count,
            'max_size': self.max_size,
            'seed': self.seed,
        }, path)

    @staticmethod
    def load(path: str) -> "ReplayBuffer":
        data = torch.load(path, weights_only=False)
        buf = ReplayBuffer(max_size=data['max_size'], seed=data['seed'])
        buf.ring_buffer = data['ring_buffer']
        buf.count = data['count']
        return buf            
    
    def __len__(self):
        return min(self.count, self.max_size)


