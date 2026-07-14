import numpy as np
import torch
from pathlib import Path
from huggingface_hub import HfApi, create_repo, hf_hub_download

class ReplayBuffer:
    def __init__(self, max_size=10000, seed=42):
        self.max_size = max_size

        self.ring_buffer = [None] * max_size
        self.rng = np.random.default_rng(seed)
        self.count = 0

    def add(self, trajectory, z):
        z = torch.tensor(z, dtype=torch.float32).unsqueeze(-1)

        for s, pi, mask in trajectory:
            pointer = self.count % self.max_size

            s = s.clone().detach().float()
            pi = pi.clone().detach().float()
            mask = mask.clone().detach().bool()

            self.ring_buffer[pointer] = (s, pi, mask, z)

            self.count += 1

    def sample(self, batch_size):
        """
        Returns a tuple s, pi, mask, z
        with each of them sized batch_size
        """
        if batch_size > len(self):
            raise ValueError("Batch size > replay buffer size")

        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        data = [self.ring_buffer[i] for i in indices]  # list of (s, pi, z)
        s_list, pi_list, mask_list, z_list = zip(*data)
        s, pi, mask, z = torch.stack(s_list), torch.stack(pi_list), torch.stack(mask_list), torch.stack(z_list)
        return s, pi, mask, z
    
    @staticmethod
    def push_to_hub(
        file_path:  str,
        repo_id:    str,
        path:       str = None,
        game:       str = None,
        version:    str = None,
        file_name:  str = None,
    ):
        if repo_id is None:
            raise ValueError("push_to_hub requires repo_id (e.g. 'username/repo-name')")

        if path is None and file_name is None:
            raise ValueError("Must provide 'path', or 'file_name' (with 'game'/'version'), for the repo destination.")

        if path is not None and any(v is not None for v in [game, version, file_name]):
            raise ValueError("Use either 'path' or 'game'/'version'/'file_name', not both.")

        if path is None and any(v is None for v in [game, version, file_name]):
            raise ValueError("Either 'path' or all of 'game', 'version', 'file_name' must be provided.")

        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        api = HfApi()
        path_in_repo = path or f"{game}/{version}/{file_name}_buffer.pt"

        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Pushed to https://huggingface.co/{repo_id}")

    @staticmethod
    def pull_from_hub(
        repo_id:    str,
        path:       str = None,
        game:       str = None,
        version:    str = None,
        file_name:  str = None,
        local_dir:  str = None,
        revision:   str = None,
    ) -> str:
        """Download a buffer file from HF Hub. Returns the local file path."""
        if path is None and file_name is None:
            raise ValueError("Must provide 'path', or 'file_name' (with 'game'/'version'), to pull a buffer.")

        if path is not None and any(v is not None for v in [game, version, file_name]):
            raise ValueError("Use either 'path' or 'game'/'version'/'file_name', not both.")

        if path is None and any(v is None for v in [game, version, file_name]):
            raise ValueError("Either 'path' or all of 'game', 'version', 'file_name' must be provided.")

        path_in_repo = path or f"{game}/{version}/{file_name}_buffer.pt"

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="model",
            revision=revision,
            local_dir=local_dir,
        )

        print(f"Pulled https://huggingface.co/{repo_id} to {local_dir}")
        return local_path

    def save(self, game: str, version: str, file_name: str, parent_dir: str = "checkpoints",
            path: str = None, push_to_hf: bool = True, repo_id: str = None):
        dir_path = Path(f"{parent_dir}/{game}/{version}")
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = path or dir_path / f"{file_name}_buffer.pt"

        torch.save({
            'ring_buffer': self.ring_buffer,
            'count': self.count,
            'max_size': self.max_size,
        }, file_path)

        print(f"Buffer saved at {file_path}")

        if push_to_hf:
            ReplayBuffer.push_to_hub(
                file_path=file_path,
                game=game,
                version=version,
                file_name=file_name,
                repo_id=repo_id,
            )

    @staticmethod
    def load(game: str = None, version: str = None, file_name: str = None,
            parent_dir: str = "checkpoints", path: str = None) -> "ReplayBuffer":
        path = path or f"{parent_dir}/{game}/{version}/{file_name}_buffer.pt"
        data = torch.load(path, weights_only=False)
        buf = ReplayBuffer(max_size=data['max_size'])
        buf.ring_buffer = data['ring_buffer']
        buf.count = data['count']
        return buf

    @staticmethod
    def load_from_hub(
        repo_id:    str,
        path:       str = None,
        game:       str = None,
        version:    str = None,
        file_name:  str = None,
        local_dir:  str = None,
        revision:   str = None,
    ) -> "ReplayBuffer":
        """Pull a buffer from HF Hub and load it."""
        local_path = ReplayBuffer.pull_from_hub(
            repo_id=repo_id, path=path, game=game, version=version,
            file_name=file_name, local_dir=local_dir, revision=revision,
        )
        return ReplayBuffer.load(path=local_path)


    def __len__(self):
        return min(self.count, self.max_size)


