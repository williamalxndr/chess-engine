"""Streaming access to the sharded pretrain dataset on the Hugging Face Hub.

`fetch.py` writes one shard per PGN file as three artifacts:

    {name}_states.pt    float32 [N, C, 8, 8]
    {name}_masks.pt     bool    [N, 4672]
    {name}_targets.pt   {"value": float32 [N], "policy": int64 [N]}

The full dataset is far larger than a typical training disk, so shards are
downloaded one at a time, mmap'd, and deleted once consumed.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO_ID = "williamalxndr/chess-pretrain-data"

SUFFIXES = ("_states.pt", "_masks.pt", "_targets.pt")


def list_shards(repo_id: str = DEFAULT_REPO_ID, token: str | None = None) -> list[str]:
    """Return the names of shards that have all three artifacts present."""
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", token=token)

    found = defaultdict(set)
    for file in files:
        name = Path(file).name
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                found[name[: -len(suffix)]].add(suffix)

    return sorted(name for name, suffixes in found.items() if len(suffixes) == len(SUFFIXES))


def split_shards(shards: list[str], holdout: int = 1, seed: int = 0) -> tuple[list[str], list[str]]:
    """Split shard *names* into train/test.

    Splitting by shard rather than by row keeps a single player's games (which
    share openings and, after mirror augmentation, near-duplicate positions)
    entirely on one side of the split.
    """
    if holdout >= len(shards):
        raise ValueError(f"holdout={holdout} leaves no training shards out of {len(shards)}")

    shuffled = list(shards)
    random.Random(seed).shuffle(shuffled)

    return sorted(shuffled[holdout:]), sorted(shuffled[:holdout])


def _default_dir() -> Path:
    kaggle_working = Path("/kaggle/working")
    root = kaggle_working if kaggle_working.exists() else Path(__file__).parent
    return root / "shards"


def download_shard(name: str, repo_id: str = DEFAULT_REPO_ID,
                   local_dir: str | Path | None = None,
                   token: str | None = None,
                   suffixes: tuple[str, ...] = SUFFIXES) -> dict[str, Path]:
    """Download a shard's artifacts into `local_dir` and return their paths."""
    local_dir = Path(local_dir) if local_dir is not None else _default_dir()
    local_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for suffix in suffixes:
        paths[suffix] = Path(hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{name}{suffix}",
            local_dir=local_dir,
            token=token,
        ))

    return paths


@dataclass
class Shard:
    name: str
    states: torch.Tensor
    masks: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    paths: dict[str, Path] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.states)


def load_shard(name: str, repo_id: str = DEFAULT_REPO_ID,
               local_dir: str | Path | None = None,
               token: str | None = None,
               mmap: bool = True) -> Shard:
    """Download (if needed) and open one shard.

    `states` and `masks` are mmap'd by default, so a shard costs page cache
    rather than resident memory; `targets` is small enough to load outright.
    """
    paths = download_shard(name, repo_id=repo_id, local_dir=local_dir, token=token)

    states = torch.load(paths["_states.pt"], mmap=mmap, map_location="cpu", weights_only=True)
    masks = torch.load(paths["_masks.pt"], mmap=mmap, map_location="cpu", weights_only=True)
    targets = torch.load(paths["_targets.pt"], map_location="cpu", weights_only=True)

    shard = Shard(
        name=name,
        states=states,
        masks=masks,
        policy=targets["policy"].long(),
        value=targets["value"].float(),
        paths=paths,
    )

    if not (len(shard.states) == len(shard.masks) == len(shard.policy) == len(shard.value)):
        raise ValueError(
            f"shard {name} is ragged: states={len(shard.states)} masks={len(shard.masks)} "
            f"policy={len(shard.policy)} value={len(shard.value)}"
        )

    return shard


def release_shard(shard: Shard, delete_files: bool = True):
    """Drop a shard's tensors and, by default, its files from disk."""
    shard.states = shard.masks = shard.policy = shard.value = None

    if delete_files:
        for path in shard.paths.values():
            Path(path).unlink(missing_ok=True)

    shard.paths = {}


def iter_shards(names: list[str], repo_id: str = DEFAULT_REPO_ID,
                local_dir: str | Path | None = None,
                token: str | None = None,
                mmap: bool = True,
                keep_files: bool = False):
    """Yield shards one at a time, freeing each before fetching the next."""
    for name in names:
        shard = load_shard(name, repo_id=repo_id, local_dir=local_dir, token=token, mmap=mmap)
        try:
            yield shard
        finally:
            release_shard(shard, delete_files=not keep_files)


def shard_row_counts(names: list[str], repo_id: str = DEFAULT_REPO_ID,
                     local_dir: str | Path | None = None,
                     token: str | None = None,
                     keep_files: bool = True) -> dict[str, int]:
    """Row count per shard, read from the targets files only (a few MB each)."""
    counts = {}
    for name in names:
        paths = download_shard(name, repo_id=repo_id, local_dir=local_dir,
                               token=token, suffixes=("_targets.pt",))
        targets = torch.load(paths["_targets.pt"], map_location="cpu", weights_only=True)
        counts[name] = len(targets["policy"])

        if not keep_files:
            paths["_targets.pt"].unlink(missing_ok=True)

    return counts
