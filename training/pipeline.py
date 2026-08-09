import time
import os
from functools import partial
import numpy as np
import torch
from torch import optim
import copy
import argparse
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from pathlib import Path
from contextlib import contextmanager
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from profiler import timing as prof 

from core.network import PolicyValueNetwork
from selfplay.replay_buffer import ReplayBuffer
from selfplay.generator import SelfPlayGenerator
from training.trainer import Trainer
from training.profiler import profile_pipeline
from core import factory
from core.config import load_config
from training.evaluation import evaluate


class Pipeline:
    """
    Generator  -->  ReplayBuffer  -->  Trainer
   (self-play)     (stores s,pi,z)   (optimizes net)
        ^                                  |
        |_______ improved network _________|
    """

    def __init__(self, 
                 game: str = "chess", version: str = "latest",
                 load_file_name: str = None, save_file_name: str = None,
                 parent_dir: str = "checkpoints", path: str = None,
                 network: PolicyValueNetwork = None,
                 buffer_file_name: str = None, buffer_path: str = None,
                 optimizer: optim.Adam = None,
                 train_batch_size: int = 32,
                 replay_buffer_max_size: int = 10000,
                 iterations: int = 50,
                 mcts_batch_size: int = 16,
                 num_rollout: int = 100,
                 num_selfplay: int = 2,
                 steps_per_iter: int = 200,
                 patience: int = 50,
                 verbose: bool = False,
                 kaggle: bool = False,
                 push_to_hf: bool = False,
                 hf_repo_id: str = "williamalxndr/chess-trained-network",
                 load_from_hf: bool = False,
                 eval_interval_seconds: int = 0,
                 eval_games: int = 4,
                 eval_threshold: float = 0.55,
                 early_stopping: bool = True,
                 ):
        self.game             = game
        self.version          = version
        self.save_file_name   = save_file_name or load_file_name or "fallback"
        self.parent_dir       = parent_dir
        self.train_batch_size = train_batch_size
        self.iterations       = iterations if iterations is not None else 9999
        self.steps_per_iter   = steps_per_iter
        self.num_selfplay     = num_selfplay
        self.patience         = patience
        self.early_stopping   = early_stopping
        self.verbose          = verbose
        self.kaggle           = kaggle
        self.push_to_hf       = push_to_hf
        self.hf_repo_id       = hf_repo_id

        # Validation
        if push_to_hf and hf_repo_id is None:
            raise ValueError("push_to_hf=True requires hf_repo_id (e.g. 'username/repo-name')")
        if load_from_hf and hf_repo_id is None:
            raise ValueError("load_from_hf=True requires hf_repo_id (e.g. 'username/repo-name')")

        # Network
        if network is not None:
            self.network = network
        elif load_from_hf:
            self.network = PolicyValueNetwork.load_from_hub(
                repo_id=hf_repo_id, game=game, version=version, file_name=load_file_name,
            )
            print(f"Network pulled from https://huggingface.co/{hf_repo_id}")
        else:
            self.network = factory.load_or_build_network(
                path=path, game=game, version=version,
                file_name=load_file_name, parent_dir=parent_dir,
            )

        # Replay buffer
        _buffer_path = Path(f"{parent_dir}/{game}/{version}/{buffer_file_name}_buffer.pt") if buffer_file_name else None

        if buffer_path is not None:
            self.replay_buffer = ReplayBuffer.load(path=buffer_path)
            print("Buffer loaded from path")
        elif load_from_hf:
            try:
                self.replay_buffer = ReplayBuffer.load_from_hub(
                    repo_id=hf_repo_id, game=game, version=version, file_name=buffer_file_name,
                )
                print(f"Buffer pulled from https://huggingface.co/{hf_repo_id}")
            except (EntryNotFoundError, RepositoryNotFoundError):
                self.replay_buffer = ReplayBuffer(replay_buffer_max_size)
                print("No buffer on HF yet; buffer created")
        elif _buffer_path and _buffer_path.is_file():
            self.replay_buffer = ReplayBuffer.load(game=game, version=version, file_name=buffer_file_name, parent_dir=parent_dir)
            print(f"Buffer loaded from {parent_dir}")
        else:
            self.replay_buffer = ReplayBuffer(replay_buffer_max_size)
            print("Buffer created")

        # Generator
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.generator = SelfPlayGenerator(
            game=game, version=version,
            file_name=load_file_name, parent_dir=parent_dir, path=path,
            network=network,
            num_rollout=num_rollout, batch_size=mcts_batch_size,
            seed=rank
        )

        # Evaluation (old vs new gating). Matches are timed, not per-iteration:
        # each one costs eval_games full MCTS games.
        self.num_rollout    = num_rollout
        self.eval_interval  = eval_interval_seconds
        self.eval_games     = eval_games
        self.eval_threshold = eval_threshold
        self.is_main        = rank == 0
        self.best_network   = None
        if eval_interval_seconds:
            net = self.network.module if isinstance(self.network, DDP) else self.network
            self.best_network = copy.deepcopy(net)

        # Trainer and optimizer
        self.optimizer = optim.AdamW(self.network.parameters(), lr=0.01, fused=True) if optimizer is None else optimizer
        self.trainer = Trainer(
            self.network,
            self.optimizer,
            T_max=100,
        )        

    # ── Core loop ─────────────────────────────────────────────────────────────

    def generate(self):
        results = self.generator.generate(self.num_selfplay)
        for trajectory, z in results:
            self.replay_buffer.add(trajectory, z)

    def sample(self):
        return self.replay_buffer.sample(self.train_batch_size)

    def train_step(self):
        s, pi, mask, z = self.sample()
        return self.trainer.step(s, pi, mask, z)

    def train(self, duration_hours=None, log_interval=1):
        start = time.time()

        duration_seconds = duration_hours * 3600 if duration_hours else None

        with self._make_progress(duration_hours) as (progress, task):
            iteration      = 0
            start_time     = time.time()
            last_save_time = start_time
            last_eval_time = start_time
            save_interval  = 600
            min_loss       = float('inf')
            not_improving  = 0
            loss = policy_loss = v_loss = 0.0

            while True:
                current_time = time.time()

                if self._should_stop(current_time, start_time, duration_seconds, iteration):
                    break

                # Self-play
                self.network.eval()
                self.generate()
                while len(self.replay_buffer) < self.train_batch_size:
                    self.generate()

                # Optimize
                self.network.train()
                for _ in range(self.steps_per_iter):
                    loss, policy_loss, v_loss = self.train_step()

                # Early stopping
                min_loss, not_improving = self._update_early_stopping(loss, min_loss, not_improving)
                if self.early_stopping and not_improving >= self.patience:
                    break

                current_time = time.time()

                # Logging
                if self.kaggle and iteration % log_interval == 0:
                    elapsed = current_time - start_time
                    remaining_str = ""
                    if duration_seconds:
                        remaining = max(0, duration_seconds - elapsed)
                        remaining_str = f" | {remaining/3600:.2f}h left"
                    patience_str = f" | patience: {not_improving}/{self.patience}" if not_improving > self.patience * 0.5 else ""
                    print(f"iter {iteration} | loss: {loss:.4f} | policy: {policy_loss:.4f} | value: {v_loss:.4f} | {elapsed/60:.1f}m elapsed{remaining_str}{patience_str} | replay buffer: {len(self.replay_buffer)}")

                # Gating: run a match on the eval timer, promoting only a net that
                # beats the incumbent.
                promoted = False
                if self.eval_interval and current_time - last_eval_time >= self.eval_interval:
                    promoted = self.evaluate_and_promote(iteration)
                    last_eval_time = current_time

                # Checkpoint on promotion (never lose a net that just won its match)
                # and on the timer (a killed session must not cost more than one
                # interval). Rank 0 only, so DDP ranks do not race on the same path.
                # Only a promoted net reaches HF; the timer saves stay local.
                if self.is_main and (promoted or current_time - last_save_time >= save_interval):
                    self.save(parent_dir="kaggle" if self.kaggle else "checkpoints", push_to_hf=promoted)
                    last_save_time = current_time

                self._update_progress(progress, task, current_time, start_time, duration_seconds, loss, policy_loss, v_loss, not_improving)
                self.trainer.scheduler.step()
                iteration += 1

        end = time.time()
        elapsed = end - start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        # Final checkpoint. One last match decides whether it earns a push; with
        # gating disabled there is no incumbent to compare against, so push.
        parent_dir = "kaggle" if self.kaggle else "checkpoints"
        if self.is_main:
            promoted = self.evaluate_and_promote(iteration) if self.eval_interval else True
            self.save(parent_dir, push_to_hf=promoted)

        print(f"\nTraining finished after {h}h {m}m {s}s! To play against the trained network, run:")
        print(f"  python3 -m arena.play --game {self.game} --version {self.version} --file_name {self.save_file_name}")
        print(f"  OR")
        print(f"  python3 -m arena.play --path {parent_dir}/{self.game}/{self.version}/{self.save_file_name}.pt")       
        
        return self.get_network()


    # ── Helpers ───────────────────────────────────────────────────────────────

    @contextmanager
    def _make_progress(self, duration_hours):
        initial_desc = "loss: 0.0000 | policy loss: 0.0000 | value loss: 0.0000"
        if duration_hours:
            initial_desc = f"Time left: {duration_hours:.2f}h | " + initial_desc

        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            transient=False,
            disable=not self.verbose or self.kaggle,
        )
        task = progress.add_task(initial_desc, total=self.iterations)
        with progress:
            yield progress, task

    def _should_stop(self, current_time, start_time, duration_seconds, iteration):
        if duration_seconds and (current_time - start_time) >= duration_seconds:
            return True
        return iteration >= self.iterations

    def _update_early_stopping(self, loss, min_loss, not_improving):
        if loss < min_loss:
            return loss, 0
        return min_loss, not_improving + 1

    def _update_progress(self, progress, task, current_time, start_time,
                         duration_seconds, loss, policy_loss, v_loss, not_improving):
        if not self.verbose or self.kaggle:
            return
        patience_str = f" | ⚠ patience: {not_improving}/{self.patience}" if not_improving > self.patience * 0.5 else ""
        desc = f"loss: {loss:.4f} | policy loss: {policy_loss:.4f} | value loss: {v_loss:.4f}{patience_str}"
        if duration_seconds:
            remaining = max(0, duration_seconds - (current_time - start_time))
            desc = f"Time left: {remaining/3600:.2f}h | " + desc
        progress.update(task, advance=1, description=desc)

    def get_network(self):
        return copy.deepcopy(self.network)

    def evaluate_and_promote(self, iteration) -> bool:
        """Match the current net against the incumbent; True if it was promoted.

        The caller owns the cadence; this only refuses to run off the main rank.
        """
        if not self.is_main:
            return False

        result = evaluate(
            self.network, self.best_network,
            game=self.game, num_rollout=self.num_rollout,
            num_games=self.eval_games, win_rate_threshold=self.eval_threshold,
            pgn_dir=f"eval_pgns/{self.game}/{self.version}",
        )
        print(f"[eval] iter {iteration} | new {result.new_wins}-{result.old_wins}-{result.draws} old "
              f"| win_rate {result.win_rate:.2f} | promote={result.promote}")
        # The promoted net becomes the incumbent the next match is scored against.
        if result.promote:
            net = self.network.module if isinstance(self.network, DDP) else self.network
            self.best_network = copy.deepcopy(net)

        return result.promote

    def save(self, parent_dir: str, push_to_hf: bool = False):
        file_name = self.save_file_name
        network = self.network.module if isinstance(self.network, DDP) else self.network

        # Local save
        network.save(self.game, self.version, file_name=file_name, parent_dir=parent_dir, push_to_hf=False)
        self.replay_buffer.save(self.game, self.version, file_name=file_name, parent_dir=parent_dir, push_to_hf=False)

        if push_to_hf and self.push_to_hf:
            base = Path(f"{parent_dir}/{self.game}/{self.version}")
            try:
                PolicyValueNetwork.push_to_hub(
                    file_path=base / f"{file_name}.pt", repo_id=self.hf_repo_id,
                    game=self.game, version=self.version, file_name=file_name,
                )
                ReplayBuffer.push_to_hub(
                    file_path=base / f"{file_name}_buffer.pt", repo_id=self.hf_repo_id,
                    game=self.game, version=self.version, file_name=file_name,
                )
            except Exception as e:
                print(f"[hf] push failed ({type(e).__name__}: {e}); continuing")


def _resolve_network(cfg):
    # Resume from HF when requested and a checkpoint exists; otherwise start fresh.
    if cfg.hf.load_from_hf:
        try:
            network = PolicyValueNetwork.load_from_hub(
                repo_id=cfg.hf.repo_id, game=cfg.game, version=cfg.version, file_name=cfg.file_name,
            )
            print(f"Network pulled from https://huggingface.co/{cfg.hf.repo_id}")
            return network
        except (EntryNotFoundError, RepositoryNotFoundError):
            print("No network checkpoint on HF yet; starting fresh")
    return factory.build_network(cfg.game, cfg.version)


if __name__ == "__main__":
    prof.reset()
    cfg = load_config()

    # Set up DDP
    acc = torch.accelerator.current_accelerator()
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.accelerator.set_device_index(local_rank)
        backend = torch.distributed.get_default_backend_for_device(acc)
        dist.init_process_group(backend)
        rank = dist.get_rank()

        # Wrap the resolved network for DDP (DDP broadcasts rank-0 weights at init)
        device_id = rank % torch.accelerator.device_count()
        network = DDP(_resolve_network(cfg).to(device_id), device_ids=[device_id])
        optimizer = optim.Adam(network.parameters(), lr=0.01, fused=True)

    else:
        network = _resolve_network(cfg)
        optimizer = optim.Adam(network.parameters(), lr=0.01, fused=True)

    buffer_path, buffer_file_name = None, cfg.file_name

    pipeline = Pipeline(
        game=cfg.game,
        version=cfg.version,
        load_file_name=None,
        save_file_name=cfg.file_name,
        path=None,
        buffer_file_name=buffer_file_name,
        buffer_path=buffer_path,
        network=network,
        optimizer=optimizer,
        iterations=cfg.training.iterations,
        steps_per_iter=cfg.training.steps_per_iter,
        train_batch_size=cfg.training.train_batch_size,
        mcts_batch_size=cfg.mcts.mcts_batch_size,
        num_rollout=cfg.mcts.num_rollout,
        num_selfplay=cfg.training.num_selfplay,
        replay_buffer_max_size=cfg.training.replay_buffer_max_size,
        verbose=cfg.logging.verbose,
        kaggle=cfg.logging.kaggle,
        push_to_hf=cfg.hf.push_to_hf,
        load_from_hf=cfg.hf.load_from_hf,
        hf_repo_id=cfg.hf.repo_id,
        eval_interval_seconds=3600,
        eval_games=4,
        early_stopping=False,
    )

    pipeline.train(
        duration_hours=cfg.training.duration,
        log_interval=cfg.logging.log_interval,
    )

    if dist.is_available() and dist.is_initialized(): 
        dist.destroy_process_group()
    