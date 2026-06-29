import numpy as np
import torch
from pathlib import Path

class ReplayBuffer:
    def __init__(self, max_size=10000, seed=42):
        self.max_size = max_size

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
    
    def save(self, game: str, version: str, file_name: str, parent_dir: str = "checkpoints", path: str = None):
        dir_path = Path(f"{parent_dir}/{game}/{version}")
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = path or dir_path / f"{file_name}_buffer.pt"

        torch.save({
            'ring_buffer': self.ring_buffer,
            'count': self.count,
            'max_size': self.max_size,
        }, file_path)

    @staticmethod
    def load(game: str = None, version: str = None, file_name: str = None, 
            parent_dir: str = "checkpoints", path: str = None) -> "ReplayBuffer":
        path = path or f"{parent_dir}/{game}/{version}/{file_name}_buffer.pt"
        data = torch.load(path, weights_only=False)
        buf = ReplayBuffer(max_size=data['max_size'])
        buf.ring_buffer = data['ring_buffer']
        buf.count = data['count']
        return buf


    def __len__(self):
        return min(self.count, self.max_size)


